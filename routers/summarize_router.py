from fastapi import APIRouter, Request, UploadFile, File, HTTPException

from services.dialogue_service import summarize_dialogue
from services.summarizer import summarize_text, summarize_extracted_text
from services.extractor import extract_document_text


router = APIRouter()


@router.post("/summarize")
async def summarize(
    request: Request,
    pdf: UploadFile = File(None),
    file: UploadFile = File(None),
):
    try:
        content_type = request.headers.get("content-type", "")

        if content_type.startswith("application/json"):
            payload      = await request.json()
            source       = payload.get("source")
            text         = payload.get("text", "")
            summary_type = payload.get("summary_type", "short")
        else:
            form         = await request.form()
            source       = form.get("source")
            text         = form.get("text", "")
            summary_type = form.get("summary_type", "short")

        if not source:
            raise HTTPException(status_code=400, detail="Missing 'source' field.")

        # ── Dispatch by source ────────────────────────────────────────────

        if source == "dialogue":
            if not text:
                raise ValueError("No dialogue text provided.")
            summary = summarize_dialogue(text, summary_type)

        elif source == "text":
            print("Content-Type:", content_type)
            print("Source:", source)
            print("Text:", repr(text))
            print("Summary Type:", summary_type)
            if not text:
                raise ValueError("No text provided.")
            summary = summarize_text(text, summary_type)

        elif source in {"pdf", "document", "file"}:
            uploaded_file = pdf or file
            if uploaded_file is None:
                raise ValueError("No document file uploaded.")

            extracted_text = extract_document_text(
                uploaded_file,
                filename=getattr(uploaded_file, "filename", None),
            )
            if not extracted_text or not extracted_text.strip():
                raise ValueError("No text could be extracted from the uploaded document.")

            summary = summarize_extracted_text(extracted_text, summary_type)

        else:
            raise ValueError(f"Unknown source: '{source}'")

        return {"summary": summary}

    except HTTPException:
        raise
    except Exception as e:
        return {"error": str(e)}
