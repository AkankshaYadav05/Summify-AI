from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .dialogue_model import device


# DistilBART model for documents (Text / PDF / YouTube / Article tabs)
document_tokenizer = AutoTokenizer.from_pretrained("sshleifer/distilbart-cnn-12-6")
document_model = AutoModelForSeq2SeqLM.from_pretrained("sshleifer/distilbart-cnn-12-6")
document_model.to(device)


__all__ = ["device", "document_tokenizer", "document_model"]

