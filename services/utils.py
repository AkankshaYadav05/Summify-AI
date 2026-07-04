import re
from typing import List


def clean_data(text: str) -> str:
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"\[[0-9]*\]", "", text)
    text = re.sub(r"[^\w\s\.,!?;:\-\(\)']", "", text)
    text = re.sub(r"\n{2,}", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _split_into_sentences(text: str) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    cleaned = re.sub(r"(?<!\.)\s*([.!?])\s*", r"\1 ", cleaned)
    cleaned = re.sub(r"\s+([.,!?])", r"\1", cleaned)
    sentences = [s.strip(" -•") for s in re.split(r"(?<=[.!?])\s+", cleaned) if s.strip()]
    return sentences or [cleaned]


def _ensure_sentence(text: str) -> str:
    text = text.strip().rstrip(" -•")
    if not text:
        return ""
    if text[-1] not in ".!?":
        text += "."
    return text


def format_summary_for_type(summary: str, summary_type: str = "short") -> str:
    """Format a raw summary into a clearer structure for the UI."""
    if not summary:
        return ""

    summary_type = (summary_type or "short").lower()
    clean = re.sub(r"\s+", " ", summary).strip()
    sentences = _split_into_sentences(clean)

    if summary_type == "bullets":
        bullets: List[str] = []
        for sentence in sentences[:5]:
            item = _ensure_sentence(sentence)
            if item:
                bullets.append(f"• {item}")

        if len(bullets) < 4:
            clauses = [part.strip(" -•") for part in re.split(r"[,;]|\s+and\s+", clean) if part.strip()]
            for clause in clauses[:5]:
                cleaned_clause = _ensure_sentence(clause)
                if cleaned_clause and len(bullets) < 5:
                    bullets.append(f"• {cleaned_clause}")

        return "\n".join(bullets[:5])

    if summary_type == "notes":
        key_points = [_ensure_sentence(s) for s in sentences[:2] if s.strip()]
        action_items = [_ensure_sentence(s) for s in sentences[2:4] if s.strip()]
        outcome = [_ensure_sentence(s) for s in sentences[4:6] if s.strip()]

        if not key_points:
            key_points = [_ensure_sentence(clean[:80])]
        if not action_items:
            action_items = ["Follow through on the agreed next steps."]
        if not outcome:
            outcome = ["The work is progressing with clear ownership and direction."]

        lines = ["Key Points"]
        lines.extend(f"• {item}" for item in key_points)
        lines.extend(["", "Action Taken"])
        lines.extend(f"• {item}" for item in action_items)
        lines.extend(["", "Outcome"])
        lines.extend(f"• {item}" for item in outcome)
        return "\n".join(lines)

    if len(sentences) <= 4:
        return "\n".join(_ensure_sentence(s) for s in sentences[:4])

    return "\n".join(
        [_ensure_sentence(s) for s in sentences[:3]]
        + [_ensure_sentence(" ".join(sentences[3:5]))]
    )


__all__ = ["clean_data", "format_summary_for_type"]
