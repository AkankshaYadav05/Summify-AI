import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Device detection
if torch.backends.mps.is_available():
	device = torch.device("mps")
elif torch.cuda.is_available():
	device = torch.device("cuda")
else:
	device = torch.device("cpu")


# Load fine-tuned dialogue model (used for the "Dialogues" tab)
dialogue_tokenizer = AutoTokenizer.from_pretrained("./text_summary_model")
dialogue_model = AutoModelForSeq2SeqLM.from_pretrained("./text_summary_model")
dialogue_model.to(device)


__all__ = ["device", "dialogue_tokenizer", "dialogue_model"]

