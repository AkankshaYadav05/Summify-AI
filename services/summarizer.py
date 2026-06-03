
import re
from typing import List

from models.document_model import document_tokenizer, document_model, device
from services.utils import clean_data, chunk_text


def _has_large_overlap(summary: str, text: str, min_chars: int = 50) -> bool:
	# crude check: if a substring of summary appears verbatim in text
	s = summary.replace("\n", " ")
	t = text.replace("\n", " ")
	# check for any substring of length >= min_chars from summary in text
	for i in range(0, max(1, len(s) - min_chars)):
		fragment = s[i:i+min_chars]
		if fragment and fragment in t:
			return True
	return False


def summarize_document(text: str, summary_type: str = "short") -> str:
	text = clean_data(text)

	if len(text.split()) < 400:
		chunks = [text]
	else:
		chunks = chunk_text(text)

	chunk_summaries = []

	for chunk in chunks:
		inputs = document_tokenizer(
			chunk,
			truncation=True,
			max_length=1024,
			return_tensors="pt",
		).to(device)

		if summary_type == "short":
			max_new = 50
			min_new = 30
		elif summary_type == "bullets":
			max_new = 120
			min_new = 40
		else:
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

		summary = document_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
		chunk_summaries.append(summary)

	combined = " ".join(chunk_summaries)

	if len(chunk_summaries) > 1:
		inputs = document_tokenizer(
			combined,
			truncation=True,
			max_length=1024,
			return_tensors="pt",
		).to(device)

		outputs = document_model.generate(
			input_ids=inputs["input_ids"],
			attention_mask=inputs["attention_mask"],
			max_new_tokens=100,
			min_new_tokens=45,
			num_beams=4,
			no_repeat_ngram_size=3,
			early_stopping=True,
		)

		combined = document_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

	# If the model simply copied long parts of the input, try a sampling pass
	if _has_large_overlap(combined, text, min_chars=60):
		inputs = document_tokenizer(
			combined,
			truncation=True,
			max_length=1024,
			return_tensors="pt",
		).to(device)
		outputs = document_model.generate(
			input_ids=inputs["input_ids"],
			attention_mask=inputs["attention_mask"],
			max_new_tokens= max(40, min(120, len(combined.split())//2)),
			do_sample=True,
			top_k=50,
			top_p=0.95,
			temperature=0.7,
			repetition_penalty=2.0,
			no_repeat_ngram_size=3,
			early_stopping=True,
		)
		combined = document_tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

	sentences = re.split(r"(?<=[.!?])\s+", combined)
	seen = set()
	filtered = []

	for s in sentences:
		key = s.strip().lower()
		if not key or key in seen or len(s.split()) < 5:
			continue
		seen.add(key)
		filtered.append(s.strip())

	combined = " ".join(filtered)

	if summary_type == "short":
		sentences = re.split(r"(?<=[.!?])\s+", combined)
		combined = " ".join(sentences[:3])

	elif summary_type == "bullets":
		sentences = re.split(r"(?<=[.!?])\s+", combined)
		seen, unique = set(), []
		for s in sentences:
			key = s.lower().strip()
			if key not in seen and len(s) > 10:
				seen.add(key)
				unique.append(s)
		combined = "\n".join(f"• {s}" for s in unique[:8])

	elif summary_type == "notes":
		sentences = re.split(r"(?<=[.!?])\s+", combined)
		n = max(1, len(sentences) // 3)
		topics = sentences[:n]
		decisions = sentences[n:2 * n]
		actions = sentences[2 * n:]

		def fmt(lst):
			return "\n".join(f"  - {s}" for s in lst if len(s) > 5)

		combined = (
			f"Key Topics:\n{fmt(topics)}\n\n"
			f"Decisions:\n{fmt(decisions)}\n\n"
			f"Action Items:\n{fmt(actions)}"
		)

	return combined.strip()


__all__ = ["summarize_document"]

