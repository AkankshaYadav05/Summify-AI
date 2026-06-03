from fastapi import FastAPI, Request, HTTPException, UploadFile, File, Form
# from pydantic import BaseModel
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from services.dialogue_service import summarize_dialogue
from services.summarizer import summarize_document
from services.docs_service import extract_pdf_text, extract_article
from services.links_service import get_youtube_transcript


app = FastAPI(
    title="Summify",
    description="AI Summarizer using fine-tuned dialogue model + DistilBART",
    version="1.0",
)


# ── Static & templates ───────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")



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