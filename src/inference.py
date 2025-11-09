# src/inference.py

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# pick a real model so tests don't fail
MODEL_NAME = "distilbert-base-uncased"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
model.to(DEVICE)
model.eval()


def predict(text: str):
    """
    Runs inference on a single text string and returns:
    - predicted_class: int
    - logits: torch.Tensor
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    )
    # move to device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits  # shape: (1, num_labels)
        predicted_class = torch.argmax(logits, dim=1).item()

    return predicted_class, logits
