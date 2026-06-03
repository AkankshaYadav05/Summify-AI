import re
from typing import List

from models.dialogue_model import dialogue_tokenizer, dialogue_model, device
from services.utils import clean_data, chunk_text


def summarize_dialogue(text: str, summary_type: str = "short") -> str:
    text = clean_data(text)

    if len(text.split()) > 800:
        condensed = []
        for chunk in chunk_text(text, chunk_size=280):
            inputs = dialogue_tokenizer(
                chunk,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            ).to(device)
            outputs = dialogue_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=80,
                min_new_tokens=30,
                num_beams=5,
                repetition_penalty=2.5,
                no_repeat_ngram_size=4,
                length_penalty=2.0,
                temperature=0.8,
                top_k=40,
                top_p=0.92,
                early_stopping=True,
            )
            condensed.append(dialogue_tokenizer.decode(outputs[0], skip_special_tokens=True).strip())
        text = " ".join(condensed)

    if summary_type == "bullets":
        prompt = (
            "Summarize this conversation as clear bullet points by speaker, "
            "including main topics, decisions, and next steps:\n\n" + text
        )
        max_new = 110
        min_new = 40
    elif summary_type == "notes":
        prompt = (
            "Summarize this conversation into notes with key topics, "
            "decisions, and action items:\n\n" + text
        )
        max_new = 140
        min_new = 50
    else:
        prompt = (
            "Summarize this conversation in a concise, high-level summary, "
            "preserving speaker roles and main outcomes:\n\n" + text
        )
        max_new = 90
        min_new = 35

    inputs = dialogue_tokenizer(
        prompt,
        truncation=True,
        max_length=1024,
        return_tensors="pt",
    ).to(device)

    outputs = dialogue_model.generate(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
        max_new_tokens=max_new,
        min_new_tokens=min_new,
        num_beams=6,
        length_penalty=2.0,
        repetition_penalty=2.5,
        no_repeat_ngram_size=4,
        temperature=0.85,
        top_k=50,
        top_p=0.93,
        early_stopping=True,
    )

    summary = dialogue_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    if summary_type == "bullets" and not summary.strip().startswith("•"):
        sentences = re.split(r"(?<=[.!?])\s+", summary)
        summary = "\n".join(f"• {s.strip()}" for s in sentences if s.strip())

    if summary_type == "notes":
        if "key topics" not in summary.lower():
            summary = f"Key Topics:\n{summary}"

    return summary


__all__ = ["summarize_dialogue"]
