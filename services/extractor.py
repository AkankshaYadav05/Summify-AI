import os
import re
import subprocess
import tempfile
from collections import Counter
from io import BytesIO

import fitz  # PyMuPDF
from docx import Document


# A PDF page with fewer than this many extracted characters is treated as
# "likely image-only" — used to detect scanned PDFs that need OCR.
MIN_CHARS_PER_PAGE = 20


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_document_text(file, filename: str | None = None, max_pages: int | None = None) -> str:
    """Read an uploaded file and return clean, summarizer-ready text."""
    if hasattr(file, "file"):
        raw_bytes = file.file.read()
    else:
        raw_bytes = file.read()

    if not raw_bytes:
        raise ValueError("Uploaded file is empty.")

    filename = (filename or getattr(file, "filename", "") or "").lower()
    filename = os.path.basename(filename)

    if filename.endswith(".pdf"):
        return extract_pdf_text(raw_bytes, max_pages=max_pages)

    if filename.endswith(".docx"):
        return extract_docx_text(raw_bytes)

    if filename.endswith(".doc"):
        return extract_doc_text(raw_bytes)

    if filename.endswith((".txt", ".md")):
        return extract_plain_text(raw_bytes)

    raise ValueError("Unsupported file type. Please upload a PDF, Word (.doc/.docx), or text file.")


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------

def extract_pdf_text(raw_bytes: bytes, max_pages: int | None = None) -> str:
    """Extract text from a PDF, stripping repeated headers/footers and page numbers."""
    try:
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Could not open PDF: {e}")

    if doc.page_count == 0:
        doc.close()
        raise ValueError("PDF has no pages.")

    pages = doc if max_pages is None else doc[:max_pages]

    page_texts = []
    readable_pages = 0
    for page in pages:
        page_text = _extract_page_text(page)
        page_texts.append(page_text)
        if len(page_text) >= MIN_CHARS_PER_PAGE:
            readable_pages += 1

    doc.close()

    if readable_pages == 0:
        raise ValueError(
            "No extractable text found. This PDF looks like a scanned or "
            "image-only document and needs OCR before it can be summarized."
        )

    text = _strip_repeated_lines(page_texts)
    text = _clean_text(text)

    if not text:
        raise ValueError("PDF text extraction produced no usable content.")

    return text


def _extract_page_text(page) -> str:
    """Pull text blocks from a page in reading order (top-to-bottom, left-to-right)."""
    blocks = page.get_text("blocks")
    blocks = sorted(blocks, key=lambda b: (round(b[1] / 20), b[0]))
    lines = [b[4].strip() for b in blocks if b[4].strip()]
    return "\n".join(lines)


def _strip_repeated_lines(page_texts: list[str], min_repeat_frac: float = 0.4) -> str:
    """Remove lines that repeat across many pages — usually headers, footers, or watermarks."""
    if len(page_texts) < 3:
        return "\n\n".join(page_texts)

    line_counts: Counter = Counter()
    per_page_lines = []
    for text in page_texts:
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        per_page_lines.append(lines)
        for ln in set(lines):
            line_counts[ln] += 1

    threshold = max(2, int(len(page_texts) * min_repeat_frac))
    boilerplate = {ln for ln, count in line_counts.items() if count >= threshold}

    cleaned_pages = []
    for lines in per_page_lines:
        kept = [ln for ln in lines if ln not in boilerplate]
        cleaned_pages.append("\n".join(kept))

    return "\n\n".join(cleaned_pages)


# ---------------------------------------------------------------------------
# Word documents
# ---------------------------------------------------------------------------

def extract_docx_text(raw_bytes: bytes) -> str:
    """Extract text from a Word (.docx) file."""
    try:
        document = Document(BytesIO(raw_bytes))
    except Exception as e:
        raise ValueError(f"Could not open Word document: {e}")

    lines = [p.text.strip() for p in document.paragraphs if p.text.strip()]
    text = _clean_text("\n".join(lines))

    if not text:
        raise ValueError("No readable text found in Word document.")

    return text


# ---------------------------------------------------------------------------
# Word (.doc)
# ---------------------------------------------------------------------------

def extract_doc_text(raw_bytes: bytes) -> str:
    """Extract text from a legacy Word (.doc) file using antiword if available."""
    with tempfile.NamedTemporaryFile(suffix=".doc", delete=False) as temp_file:
        temp_file.write(raw_bytes)
        temp_path = temp_file.name

    try:
        result = subprocess.run(
            ["antiword", temp_path],
            capture_output=True,
            text=True,
            check=False,
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    if result.returncode == 0 and result.stdout.strip():
        text = _clean_text(result.stdout)
        if text:
            return text

    raise ValueError("Could not read .doc file. Please save it as .docx or .txt instead.")


# ---------------------------------------------------------------------------
# Plain text
# ---------------------------------------------------------------------------

def extract_plain_text(raw_bytes: bytes) -> str:
    """Extract text from a .txt or .md file, handling common encodings."""
    for encoding in ("utf-8-sig", "utf-16", "utf-8", "latin-1"):
        try:
            text = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw_bytes.decode("utf-8", errors="ignore")

    text = _clean_text(text)
    if not text:
        raise ValueError("Text extraction produced no usable content.")

    return text


# ---------------------------------------------------------------------------
# Shared cleanup
# ---------------------------------------------------------------------------

def _clean_text(text: str) -> str:
    """Normalize whitespace and fix common extraction artifacts."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)            # rejoin hyphenated words split across lines
    text = re.sub(r"(?<![.!?:;])\n(?!\n)", " ", text)       # turn soft line-wraps into spaces
    text = re.sub(r"\n{3,}", "\n\n", text)                  # collapse 3+ blank lines into 2
    text = re.sub(r"[ \t]+", " ", text)                     # collapse repeated spaces/tabs
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)            # remove space before punctuation
    text = re.sub(r"(?mi)^\s*(page\s*)?\d+(\s*of\s*\d+)?\s*$", "", text)  # drop standalone page numbers
    return text.strip()


__all__ = ["extract_document_text", "extract_pdf_text", "extract_docx_text", "extract_doc_text", "extract_plain_text"]