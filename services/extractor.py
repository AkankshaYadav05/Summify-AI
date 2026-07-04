"""
Text extraction utilities.

  extract_pdf_text(file, max_pages)  — UploadFile → clean text
  extract_article(url)               — URL → clean text

Both raise ValueError for user-fixable problems (bad file, unreachable URL, etc.)
so callers can turn those into 400 HTTP responses without catching Exception broadly.
"""

import io
import os
import re
import zipfile
from collections import Counter
from xml.etree import ElementTree as ET

import fitz          # PyMuPDF
import trafilatura


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# A PDF page with fewer than this many extracted characters is treated as
# "likely image-only" for that page — used to detect scanned PDFs.
_MIN_CHARS_PER_PAGE = 20


# ---------------------------------------------------------------------------
# Document extraction
# ---------------------------------------------------------------------------

def _read_uploaded_bytes(file) -> tuple[bytes, str]:
    raw_bytes = file.file.read()
    if not raw_bytes:
        raise ValueError("Uploaded file is empty.")
    filename = getattr(file, "filename", "") or ""
    return raw_bytes, os.path.basename(filename).lower()


def extract_document_text(file, filename: str | None = None, max_pages: int | None = None) -> str:
    """Extract clean, summarizer-ready text from an uploaded PDF or document."""
    raw_bytes, uploaded_name = _read_uploaded_bytes(file)
    name = (filename or uploaded_name).lower()

    if name.endswith(".pdf"):
        return extract_pdf_text_from_bytes(raw_bytes, max_pages=max_pages)

    if name.endswith(".docx"):
        return _extract_docx_text(raw_bytes)

    if name.endswith((".txt", ".md", ".csv", ".json", ".log")):
        return _extract_plain_text(raw_bytes)

    raise ValueError("Unsupported file type. Please upload a PDF, Word document, or text file.")


def extract_pdf_text(file, max_pages: int | None = None) -> str:
    """Extract clean, summarizer-ready text from an uploaded PDF."""
    raw_bytes, _ = _read_uploaded_bytes(file)
    return extract_pdf_text_from_bytes(raw_bytes, max_pages=max_pages)


def extract_pdf_text_from_bytes(raw_bytes: bytes, max_pages: int | None = None) -> str:
    try:
        doc = fitz.open(stream=raw_bytes, filetype="pdf")
    except Exception as e:
        raise ValueError(f"Could not open PDF: {e}")

    if doc.page_count == 0:
        raise ValueError("PDF has no pages.")

    pages_to_read = doc if max_pages is None else doc[:max_pages]

    page_texts: list[str] = []
    chars_per_page: list[int] = []
    for page in pages_to_read:
        blocks = page.get_text("blocks")
        blocks = sorted(blocks, key=lambda b: (round(b[1] / 20), b[0]))
        page_text = "\n".join(b[4].strip() for b in blocks if b[4].strip())
        page_texts.append(page_text)
        chars_per_page.append(len(page_text))

    doc.close()

    text_bearing_pages = sum(1 for c in chars_per_page if c >= _MIN_CHARS_PER_PAGE)
    if text_bearing_pages == 0:
        raise ValueError(
            "No extractable text found. This PDF appears to be scanned or "
            "image-only and requires OCR before it can be summarized."
        )

    full_text = _strip_repeated_headers_footers(page_texts)
    full_text = _clean_extracted_text(full_text)

    if not full_text.strip():
        raise ValueError("PDF text extraction produced no usable content.")

    return full_text


def _extract_docx_text(raw_bytes: bytes) -> str:
    try:
        with zipfile.ZipFile(io.BytesIO(raw_bytes)) as archive:
            document_xml = archive.read("word/document.xml")
    except Exception as exc:
        raise ValueError(f"Could not read DOCX content: {exc}")

    try:
        root = ET.fromstring(document_xml)
    except ET.ParseError as exc:
        raise ValueError(f"Could not parse DOCX content: {exc}")

    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", namespace):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", namespace)]
        if texts:
            paragraphs.append("".join(texts))

    text = "\n".join(paragraphs)
    text = _clean_extracted_text(text)
    if not text.strip():
        raise ValueError("DOCX text extraction produced no usable content.")
    return text


def _extract_plain_text(raw_bytes: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-16", "latin-1"):
        try:
            text = raw_bytes.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        text = raw_bytes.decode("utf-8", errors="ignore")

    text = _clean_extracted_text(text)
    if not text.strip():
        raise ValueError("Text extraction produced no usable content.")
    return text


def _strip_repeated_headers_footers(page_texts: list[str], min_repeat_frac: float = 0.4) -> str:
    """Remove lines that repeat across many pages (headers, footers, watermarks)."""
    if len(page_texts) < 3:
        return "\n\n".join(page_texts)

    line_counts: Counter = Counter()
    per_page_lines: list[list[str]] = []
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


def _clean_extracted_text(text: str) -> str:
    """Normalize whitespace and fix common PDF-extraction artifacts."""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)          # hyphenated line-break splits
    text = re.sub(r"(?<![.!?:;])\n(?!\n)", " ", text)     # soft line-wraps → space
    text = re.sub(r"\n{3,}", "\n\n", text)                 # 3+ newlines → 2
    text = re.sub(r"[ \t]+", " ", text)                    # collapse spaces/tabs
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)           # space before punctuation
    text = re.sub(r"(?m)^\s*(page\s*)?\d+(\s*of\s*\d+)?\s*$", "", text, flags=re.IGNORECASE)
    return text.strip()


__all__ = ["extract_pdf_text", "extract_document_text", "extract_pdf_text_from_bytes"]
