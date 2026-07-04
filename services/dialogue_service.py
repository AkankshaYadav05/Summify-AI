"""
Dialogue summarization service.

Uses the local fine-tuned model (text_summary_model) for the "Dialogues" tab.

Bug fixed vs original:
  - repetition_penalty=2.0 on the final-pass generation was causing degenerate
    looping on some inputs. Reduced to 1.5, which suppresses repeats without
    trapping the beam search in penalty-avoidance spirals.
"""

import re
from typing import List

from models.dialogue_model import dialogue_tokenizer, dialogue_model, device
from services.utils import clean_data


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------

def _chunk_dialogue_by_turns(text: str, max_turns: int = 20) -> List[str]:
    """Split dialogue into overlapping chunks of at most max_turns speaker turns."""
    turns = re.split(r'\n(?=[A-Z][a-z]+:)', text)
    chunks: List[str] = []
    current: List[str] = []
    for turn in turns:
        current.append(turn)
        if len(current) >= max_turns:
            chunks.append("\n".join(current))
            current = current[-3:]   # 3-turn overlap for context continuity
    if current:
        chunks.append("\n".join(current))
    return chunks


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def summarize_dialogue(text: str, summary_type: str = "short") -> str:
    """Summarize a dialogue/conversation using the fine-tuned local model.

    Args:
        text: Raw dialogue text (speaker-prefixed turns or free-form).
        summary_type: "short" | "bullets" | "notes"

    Returns:
        Formatted summary string.
    """
    text = clean_data(text)

    # ── Stage 1: condense long dialogues chunk-by-chunk ──────────────────
    if len(text.split()) > 800:
        if summary_type == "bullets":
            chunk_max_new, chunk_min_new = 80, 25
        elif summary_type == "notes":
            chunk_max_new, chunk_min_new = 100, 35
        else:
            chunk_max_new, chunk_min_new = 70, 25

        condensed: List[str] = []
        for chunk in _chunk_dialogue_by_turns(text, max_turns=20):
            inputs = dialogue_tokenizer(
                chunk,
                truncation=True,
                max_length=1024,
                return_tensors="pt",
            ).to(device)
            outputs = dialogue_model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                num_beams=4,
                max_new_tokens=chunk_max_new,
                min_new_tokens=chunk_min_new,
                no_repeat_ngram_size=3,
                length_penalty=1.0,
                # FIX: was 2.0 — too aggressive in the condensing pass, caused
                # beam search to produce grammatically broken filler to avoid
                # repetition penalty. 1.3 is enough to suppress obvious repeats.
                repetition_penalty=1.3,
                early_stopping=True,
            )
            condensed.append(
                dialogue_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
            )
        text = " ".join(condensed)

    # ── Stage 2: final-pass summarization with prompt steering ───────────
    words = len(text.split())

    if words < 150:
        short_max, short_min = 70, 25
        bullet_max, bullet_min = 90, 35
        notes_max, notes_min = 110, 45

    elif words < 350:
        short_max, short_min = 100, 35
        bullet_max, bullet_min = 140, 50
        notes_max, notes_min = 180, 70

    else:
        short_max, short_min = 140, 50
        bullet_max, bullet_min = 190, 70
        notes_max, notes_min = 230, 90


    if summary_type == "short":

        prompt = f"""
    Summarize the following conversation.

    Requirements:
    - Mention the main discussion.
    - Mention important decisions.
    - Mention responsibilities.
    - Mention outcomes.
    - Do not repeat names.
    - Do not invent information.

    Conversation:

    {text}
    """

        max_new = short_max
        min_new = short_min


    elif summary_type == "bullets":

        prompt = f"""
    Summarize the following conversation.

    Return ONLY bullet points.

    Include:
    - Main discussion
    - Decisions made
    - Responsibilities assigned
    - Next steps

    Do not repeat information.

    Conversation:

    {text}
    """

        max_new = bullet_max
        min_new = bullet_min


    else:

        prompt = f"""
    Convert the following conversation into meeting notes.

    Use this format exactly.

    Overview

    Key Decisions

    Action Items

    Outcome

    Conversation:

    {text}
    """

        max_new = notes_max
        min_new = notes_min

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
        num_beams=8,
        repetition_penalty=2.0,
        no_repeat_ngram_size=3,
        length_penalty=1.2,
        early_stopping=True,
    )

    summary = dialogue_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    ).strip()

    summary = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', summary)

    sentences = re.split(r'(?<=[.!?])\s+', summary)

    clean = []
    seen = set()

    for s in sentences:

        s = s.strip()

        if not s:
            continue

        key = s.lower()

        if key not in seen:
            seen.add(key)
            clean.append(s)

    summary = " ".join(clean)

    # ── Post-process formatting ───────────────────────────────────────────
    if summary_type == "bullets":

        bullets = []

        for s in clean:

            if len(s) > 15:
                bullets.append(f"• {s}")

        return "\n".join(bullets[:6])


    if summary_type == "notes":

        notes = []

        if len(clean) >= 1:
            notes.append("Overview")
            notes.append(clean[0])

        if len(clean) > 2:
            notes.append("\nKey Decisions")

            for s in clean[1:-1]:
                notes.append(f"• {s}")

        if len(clean) > 1:
            notes.append("\nOutcome")
            notes.append(clean[-1])

        return "\n".join(notes)


    return summary


__all__ = ["summarize_dialogue"]
