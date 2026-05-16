from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
# from pydantic import BaseModel
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import re
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import fitz
import trafilatura
# from youtube_transcript_api import YouTubeTranscriptApi
from urllib.parse import urlparse, parse_qs
# from typing import Optional


app = FastAPI(
    title="Summify",
    description="AI Summarizer using fine-tuned dialogue model + DistilBART",
    version="1.0",
)

# ── Device ────────────────────────────────────────────────────────
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

# ── Models ────────────────────────────────────────────────────────
# Your fine-tuned model  →  used for the "Dialogues" tab only
dialogue_tokenizer = AutoTokenizer.from_pretrained("./text_summary_model")
dialogue_model     = AutoModelForSeq2SeqLM.from_pretrained("./text_summary_model")
dialogue_model.to(device)

# DistilBART  →  used for Text / PDF / YouTube / Article tabs
# sshleifer/distilbart-cnn-12-6 is a great default: fast, accurate,
# handles long-ish documents well via our chunking strategy.
document_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
document_model     = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
document_model.to(device)

# ── Static & templates ────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")


# ── Helpers ───────────────────────────────────────────────────────
def clean_data(text: str) -> str:
    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\[[0-9]*\]", "", text)
    text = re.sub(r"[^a-zA-Z0-9.,!? ]", "", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 200) -> list[str]:
    words  = text.split()
    return [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]


# ── Dialogue summarizer (your model) ─────────────────────────────
# FIX: simpler, cleaner prompts so the fine-tuned model actually
#      follows the instruction instead of ignoring it.
def summarize_dialogue(text: str, summary_type: str = "short") -> str:
    text = clean_data(text)

    # Keep the prompt minimal — fine-tuned models work best with the
    # exact phrasing they saw during training.  We prepend a short
    # instruction token style prefix and let the model do the rest.
    if summary_type == "bullets":
        prompt = f"summarize as bullet points: {text}"
    elif summary_type == "notes":
        prompt = f"summarize key topics decisions and action items: {text}"
    else:  # "short"
        prompt = f"summarize: {text}"

    inputs = dialogue_tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    outputs = dialogue_model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_length=60,
        min_length=15,
        num_beams=8,
        length_penalty=2.5,
        repetition_penalty=2.5,
        no_repeat_ngram_size=4,
        early_stopping=True
    )

    summary = dialogue_tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Post-process: if bullets were requested but the model didn't add
    # bullet markers, split sentences and add them ourselves.
    if summary_type == "bullets" and not summary.strip().startswith("•"):
        sentences = re.split(r"(?<=[.!?])\s+", summary.strip())
        summary = "\n".join(f"• {s}" for s in sentences if s)

    # Post-process for notes: ensure section headers exist
    if summary_type == "notes":
        if "key topics" not in summary.lower():
            summary = f"Key Topics:\n{summary}"

    return summary


# ── Document summarizer (DistilBART) ─────────────────────────────
# FIX: DistilBART is a seq2seq model trained on news summarization.
#      It ignores instruction text in the input; instead we shape its
#      output in post-processing based on summary_type.
def summarize_document(text: str, summary_type: str = "short") -> str:
    text   = clean_data(text)

    if len(text.split()) < 400:
        chunks = [text]
    else:
        chunks = chunk_text(text)

    chunk_summaries = []

    for chunk in chunks:
        # DistilBART works best with plain text — no instruction prefix
        inputs = document_tokenizer(
            chunk,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(device)

        # Tune length per summary type
        if summary_type == "short":
            max_new = 50
            min_new = 30
        elif summary_type == "bullets":
            max_new = 120
            min_new = 40
        else:  # notes
            max_new = 150
            min_new = 50

        outputs = document_model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new,
            min_new_tokens=min_new,
            num_beams=4,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

        summary = document_tokenizer.decode(outputs[0], skip_special_tokens=True)
        chunk_summaries.append(summary.strip())

    combined = " ".join(chunk_summaries)

    # Remove repeated sentences
    sentences = re.split(r'(?<=[.!?]) +', combined)

    seen = set()
    filtered = []

    for s in sentences:
        key = s.strip().lower()
        if key not in seen and len(s.split()) > 4:
            seen.add(key)
            filtered.append(s)

    combined = " ".join(filtered)
    

    # ── Post-process to match the requested format ────────────────
    if summary_type == "short":
        # Keep only the first 2–3 sentences
        sentences = re.split(r"(?<=[.!?])\s+", combined)
        combined = " ".join(sentences[:3])

    elif summary_type == "bullets":
        # Split into sentences and bullet them
        sentences = re.split(r"(?<=[.!?])\s+", combined)
        # Deduplicate while preserving order
        seen, unique = set(), []
        for s in sentences:
            key = s.lower().strip()
            if key not in seen and len(s) > 10:
                seen.add(key)
                unique.append(s)
        combined = "\n".join(f"• {s}" for s in unique[:8])

    elif summary_type == "notes":
        # Split sentences and group into sections heuristically
        sentences = re.split(r"(?<=[.!?])\s+", combined)
        n = max(1, len(sentences) // 3)
        topics    = sentences[:n]
        decisions = sentences[n:2*n]
        actions   = sentences[2*n:]

        def fmt(lst): return "\n".join(f"  - {s}" for s in lst if len(s) > 5)

        combined = (
            f"Key Topics:\n{fmt(topics)}\n\n"
            f"Decisions:\n{fmt(decisions)}\n\n"
            f"Action Items:\n{fmt(actions)}"
        )

    return combined.strip()


# ── Extractors ────────────────────────────────────────────────────
def extract_pdf_text(file) -> str:
    doc  = fitz.open(stream=file.file.read(), filetype="pdf")
    return "".join(page.get_text() for page in doc)




def extract_article(url: str) -> str:
    downloaded = trafilatura.fetch_url(url)

    if not downloaded:
        raise ValueError("Could not download the article from URL.")

    text = trafilatura.extract(
        downloaded,
        include_comments=False,
        include_tables=False,
        output_format="text",
    )

    if not text or not text.strip():
        raise ValueError("Could not extract article text from URL.")

    return text.strip()


def extract_video_id(url: str) -> str:
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower()

    if hostname in ("youtu.be", "www.youtu.be"):
        return parsed.path.lstrip("/")

    if "youtube" in hostname:
        params = parse_qs(parsed.query)
        if "v" in params and params["v"]:
            return params["v"][0]

    raise ValueError("Invalid YouTube URL.")


def get_youtube_transcript(url: str) -> str:
    video_id = extract_video_id(url)
    from youtube_transcript_api import YouTubeTranscriptApi

    transcript = None
    if hasattr(YouTubeTranscriptApi, "get_transcript"):
        transcript = YouTubeTranscriptApi.get_transcript(video_id)
    else:
        ytt = YouTubeTranscriptApi()
        if hasattr(ytt, "fetch"):
            transcript = ytt.fetch(video_id)
        elif hasattr(ytt, "get_transcript"):
            transcript = ytt.get_transcript(video_id)
        else:
            raise RuntimeError("Unsupported youtube_transcript_api version")

    parts = []
    for snippet in transcript:
        if isinstance(snippet, dict):
            parts.append(snippet.get("text", ""))
        else:
            parts.append(getattr(snippet, "text", ""))

    return " ".join(p for p in parts if p)


# ── API endpoint ──────────────────────────────────────────────────
@app.post("/summarize")
async def summarize(
    request: Request,
    pdf: UploadFile = File(None),
):
    try:
        content_type = request.headers.get("content-type", "")

        if content_type.startswith("application/json"):
            payload      = await request.json()
            source       = payload.get("source")
            text         = payload.get("text", "")
            link         = payload.get("link", "")
            summary_type = payload.get("summary_type", "short")
        else:
            form         = await request.form()
            source       = form.get("source")
            text         = form.get("text", "")
            link         = form.get("link", "")
            summary_type = form.get("summary_type", "short")

        if not source:
            raise HTTPException(status_code=400, detail="Missing 'source' field.")

        # ── Route to the right model ──────────────────────────────
        # "dialogue" tab  →  your fine-tuned model
        # everything else →  DistilBART
        if source == "dialogue":
            if not text:
                raise ValueError("No dialogue text provided.")
            summary = summarize_dialogue(text, summary_type)

        elif source == "text":
            if not text:
                raise ValueError("No text provided.")
            summary = summarize_document(text, summary_type)

        elif source == "pdf":
            if pdf is None:
                raise ValueError("No PDF file uploaded.")
            pdf_text = extract_pdf_text(pdf)
            summary  = summarize_document(pdf_text, summary_type)

        elif source == "youtube":
            if not link:
                raise ValueError("No YouTube URL provided.")

            try:
                transcript = get_youtube_transcript(link)
                summary    = summarize_document(transcript, summary_type)
            except ValueError:
                # If a non-YouTube URL was pasted into the Links tab,
                # treat it as an article link instead of failing.
                article_text = extract_article(link)
                summary      = summarize_document(article_text, summary_type)

        elif source == "article":
            if not link:
                raise ValueError("No article URL provided.")
            article_text = extract_article(link)
            summary      = summarize_document(article_text, summary_type)

        else:
            raise ValueError(f"Unknown source: '{source}'")

        return {"summary": summary}

    except Exception as e:
        # Return as JSON error (not 500) so the frontend shows it nicely
        return {"error": str(e)}


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)