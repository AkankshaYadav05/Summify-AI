from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .dialogue_model import device

MODEL_NAME = "google/flan-t5-base"

# Load tokenizer
document_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load summarization model
document_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

# Move model to CPU/GPU
document_model.to(device)

__all__ = ["device", "document_tokenizer", "document_model"]