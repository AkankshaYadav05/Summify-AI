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


def chunk_text(text: str, chunk_size: int = 350) -> List[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        words = sentence.split()
        if current_len + len(words) > chunk_size and current:
            chunks.append(" ".join(current).strip())
            current = []
            current_len = 0

        current.append(sentence)
        current_len += len(words)

    if current:
        chunks.append(" ".join(current).strip())

    return chunks


__all__ = ["clean_data", "chunk_text"]
