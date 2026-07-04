"""
FLAN-T5 text summarizer.

Single public function:
    summarize_text(text, summary_type) -> str

summary_type: "short" | "bullets" | "notes"

The TextSummarizer class is kept internal; callers should only need
summarize_text().  Import the class directly only if you need batch
inference (TextSummarizer.summarize_batch).
"""

import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class TextSummarizer:
    def __init__(self, model_name: str = "google/flan-t5-base", quantize: bool = False):
        """Initialize the summarizer.

        Args:
            model_name: Pre-trained seq2seq model.
                - "google/flan-t5-base" / any "*t5*" model: instruction-tuned,
                  REQUIRES a "summarize: " task prefix — handled automatically.
                - "facebook/bart-large-cnn" / "sshleifer/distilbart-cnn-12-6":
                  summarization-finetuned, no prefix needed.
            quantize: Apply dynamic int8 quantization (CPU only).
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        # T5-family models need an explicit task prefix on the input text.
        self.requires_task_prefix = "t5" in model_name.lower()
        self.task_prefix = "summarize: " if self.requires_task_prefix else ""

        if quantize and self.device == "cpu":
            self.model = torch.quantization.quantize_dynamic(self.model, dtype=torch.qint8)

        self.model.to(self.device)
        self.model.eval()

    # ------------------------------------------------------------------
    # Pre / post processing
    # ------------------------------------------------------------------

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize input text."""
        text = re.sub(r"\s+", " ", text.strip())
        text = re.sub(r"https?://\S+|www\.\S+", "", text)
        text = re.sub(r"[^\w\s.,!?-]", "", text)
        text = re.sub(r'\b(\w+)(\s+\1\b)+', r'\1', text, flags=re.IGNORECASE)
        return text

    @staticmethod
    def clean_summary(summary: str) -> str:
        """Remove degenerate word/phrase repetition from model output."""
        # Collapse repeated words: "tool tool tool" → "tool"
        summary = re.sub(r'\b(\w+)(\s+\1\b){1,}', r'\1', summary, flags=re.IGNORECASE)
        # Collapse repeated 2-4 word phrases
        summary = re.sub(r'\b((?:\w+\s+){1,4}\w+)\s+\1\b', r'\1', summary, flags=re.IGNORECASE)
        summary = re.sub(r'\s+([.,!?])', r'\1', summary)
        summary = re.sub(r'\s+', ' ', summary).strip()
        summary = TextSummarizer._dedupe_sentences(summary)
        if summary and summary[-1] not in ".!?":
            summary += "."
        return summary

    @staticmethod
    def _dedupe_sentences(text: str, overlap_threshold: float = 0.7) -> str:
        """Drop near-duplicate sentences (by word overlap) from a summary."""
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        kept: list[str] = []
        kept_word_sets: list[set] = []
        for sent in sentences:
            words = {w.lower() for w in re.findall(r"\w+", sent)}
            if not words:
                continue
            is_dupe = any(
                len(words & prev) / min(len(words), len(prev)) >= overlap_threshold
                for prev in kept_word_sets if prev
            )
            if not is_dupe:
                kept.append(sent.strip())
                kept_word_sets.append(words)
        return " ".join(kept)

    # ------------------------------------------------------------------
    # Chunking
    # ------------------------------------------------------------------

    def split_long_text(self, text: str, max_tokens: int = 1024) -> list[str]:
        """Split text into chunks that fit within the model's token window."""
        tokens = self.tokenizer.tokenize(text)
        chunks = [tokens[i:i + max_tokens] for i in range(0, len(tokens), max_tokens)]
        return [self.tokenizer.convert_tokens_to_string(chunk) for chunk in chunks]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    def _generate(self, text: str, max_length: int, min_length: int,
                  length_penalty: float, repetition_penalty: float,
                  num_beams: int, early_stopping: bool) -> str:
        """Generate a summary for a single chunk."""
        try:
            model_input = self.task_prefix + text if self.requires_task_prefix else text
            inputs = self.tokenizer(
                model_input, max_length=1024, truncation=True, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length,
                    min_length=min_length,
                    length_penalty=length_penalty,
                    repetition_penalty=repetition_penalty,
                    num_beams=num_beams,
                    early_stopping=early_stopping,
                    no_repeat_ngram_size=3,
                )
            summary = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
            return self.clean_summary(summary)
        except Exception as e:
            print(f"Error during summarization: {e}")
            return text

    def summarize_batch(self, texts: list[str], batch_size: int = 4,
                        max_length: int = 130, min_length: int = 30,
                        length_penalty: float = 1.2, repetition_penalty: float = 1.3,
                        num_beams: int = 4, early_stopping: bool = True,
                        **extra_kwargs) -> list[str]:
        """Summarize multiple texts in batches."""
        summaries: list[str] = []
        for i in range(0, len(texts), batch_size):
            batch = [self.preprocess_text(t) for t in texts[i:i + batch_size]]
            if self.requires_task_prefix:
                batch = [self.task_prefix + t for t in batch]
            inputs = self.tokenizer(
                batch, max_length=1024, truncation=True,
                padding=True, return_tensors="pt"
            ).to(self.device)
            with torch.no_grad():
                summary_ids = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_length=max_length, min_length=min_length,
                    length_penalty=length_penalty, repetition_penalty=repetition_penalty,
                    num_beams=num_beams, early_stopping=early_stopping,
                    no_repeat_ngram_size=3, **extra_kwargs,
                )
            summaries.extend(
                self.clean_summary(self.tokenizer.decode(ids, skip_special_tokens=True))
                for ids in summary_ids
            )
        return summaries

    def summarize(self, text: str, max_length: int = 130, min_length: int = 30,
                  length_penalty: float = 1.0, repetition_penalty: float = 1.3,
                  num_beams: int = 6, early_stopping: bool = True) -> dict[str, str]:
        """Summarize text, chunking automatically for long inputs.

        length_penalty defaults to 1.0 (neutral) because T5 tends to
        under-generate; push higher (1.2-1.4) if you swap to a BART model.
        """
        cleaned = self.preprocess_text(text)
        chunks = self.split_long_text(cleaned)
        chunk_summaries = []
        for chunk in chunks:
            chunk_max = max(max_length // len(chunks), 40)
            chunk_min = max(min(min_length // len(chunks), chunk_max - 5), 10)
            chunk_summaries.append(self._generate(
                chunk,
                max_length=chunk_max, min_length=chunk_min,
                length_penalty=length_penalty, repetition_penalty=repetition_penalty,
                num_beams=num_beams, early_stopping=early_stopping,
            ))
        return {
            "original_text": text,
            "cleaned_text": cleaned,
            "summary": " ".join(chunk_summaries),
        }


# ---------------------------------------------------------------------------
# Module-level singleton — one model load for the process lifetime
# ---------------------------------------------------------------------------

adv_summarizer = TextSummarizer(model_name="google/flan-t5-base", quantize=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

import re

def summarize_text(text: str, summary_type: str = "short") -> str:
    """
    Summarize text using FLAN-T5.

    summary_type:
        short   -> concise paragraph
        bullets -> bullet point summary
        notes   -> structured notes
    """

    if not text or not text.strip():
        return ""

    summary_type = (summary_type or "short").lower()

    # ----------------------------
    # Dynamic summary length
    # ----------------------------
    words = len(text.split())

    if words < 150:
        short_max, short_min = 70, 25
        bullet_max, bullet_min = 90, 35
        notes_max, notes_min = 110, 45

    elif words < 400:
        short_max, short_min = 120, 45
        bullet_max, bullet_min = 160, 60
        notes_max, notes_min = 200, 80

    else:
        short_max, short_min = 170, 60
        bullet_max, bullet_min = 220, 90
        notes_max, notes_min = 280, 110

    common_args = {
        "num_beams": 6,
        "length_penalty": 1.2,
        "repetition_penalty": 2.0,
        # "no_repeat_ngram_size": 3,
        "early_stopping": True,
    }

    # ----------------------------
    # SHORT SUMMARY
    # ----------------------------
    if summary_type == "short":

        return adv_summarizer.summarize(
            text,
            max_length=short_max,
            min_length=short_min,
            **common_args,
        )["summary"]

    # ----------------------------
    # BULLET SUMMARY
    # ----------------------------
    elif summary_type == "bullets":

        summary = adv_summarizer.summarize(
            text,
            max_length=bullet_max,
            min_length=bullet_min,
            **common_args,
        )["summary"]

        sentences = re.split(r'(?<=[.!?])\s+', summary)

        bullets = []
        seen = set()

        for s in sentences:

            s = s.strip()

            if len(s) < 15:
                continue

            key = s.lower()

            if key not in seen:
                seen.add(key)
                bullets.append(f"• {s}")

        return "\n".join(bullets[:6])

    # ----------------------------
    # NOTES SUMMARY
    # ----------------------------
    elif summary_type == "notes":

        summary = adv_summarizer.summarize(
            text,
            max_length=notes_max,
            min_length=notes_min,
            **common_args,
        )["summary"]

        sentences = [
            s.strip()
            for s in re.split(r'(?<=[.!?])\s+', summary)
            if s.strip()
        ]

        output = []

        if sentences:
            output.append("Overview")
            output.append(sentences[0])

        if len(sentences) > 2:
            output.append("\nKey Points")

            for s in sentences[1:-1]:
                output.append(f"• {s}")

        if len(sentences) > 1:
            output.append("\nConclusion")
            output.append(sentences[-1])

        return "\n".join(output)

    else:
        return adv_summarizer.summarize(
            text,
            max_length=short_max,
            min_length=short_min,
            **common_args,
        )["summary"]


def summarize_extracted_text(text: str, summary_type: str = "short") -> str:
    """Summarize text that has already been extracted from a document upload."""
    return summarize_text(text, summary_type)


__all__ = ["summarize_text", "summarize_extracted_text", "TextSummarizer"]
