import fitz
import trafilatura


def extract_pdf_text(file) -> str:
	# `file` is expected to be a FastAPI UploadFile-like object
	doc = fitz.open(stream=file.file.read(), filetype="pdf")
	return "".join(page.get_text() for page in doc)


def extract_article(url: str) -> str:
	downloaded = trafilatura.fetch_url(url)

	if not downloaded:
		raise ValueError("Could not download the article from URL.")

	text = trafilatura.extract(
		downloaded,
		include_comments=False,
		include_tables=False,
		output_format="txt",
	)

	if not text or not text.strip():
		raise ValueError("Could not extract article text from URL.")

	return text.strip()


__all__ = ["extract_pdf_text", "extract_article"]

