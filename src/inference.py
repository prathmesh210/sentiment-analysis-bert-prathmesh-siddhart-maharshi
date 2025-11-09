# src/inference.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from config import DEVICE  # optional: if you want to use GPU

# Load model + tokenizer
MODEL_PATH = "path/to/saved/model"  # replace with actual path after training
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
model.eval()  # set to eval mode
model.to(DEVICE)  # optional, move to GPU if available

def predict(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}  # move tensors to device
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class, logits