# tests/unit/test_inference.py
import pytest
from src.inference import predict

def test_predict_returns_label_and_logits():
    text = "I love this product!"
    label, logits = predict(text)
    assert isinstance(label, int)
    assert logits.shape[1] == 2  # assuming binary classification
