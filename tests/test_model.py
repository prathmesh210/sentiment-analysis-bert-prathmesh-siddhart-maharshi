import pytest
import torch

from src.model import get_model, get_tokenizer, MODEL_NAME, NUM_LABELS

def test_model_instantiation():
    try:
        model = get_model(MODEL_NAME, NUM_LABELS)
    except OSError:
        pytest.skip("Model cannot be downloaded in this environment.")
    assert model is not None
    assert model.config.num_labels == NUM_LABELS

def test_model_forward_pass():
    try:
        tokenizer = get_tokenizer(MODEL_NAME)
        model = get_model(MODEL_NAME, NUM_LABELS)
    except OSError:
        pytest.skip("Model cannot be downloaded in this environment.")

    texts = ["i love this", "this is bad"]
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**encodings)

    logits = outputs.logits
    assert logits.shape[0] == len(texts)
    assert logits.shape[1] == NUM_LABELS

@pytest.mark.skipif(not torch.cuda.is_available(), reason="Skip on CI if no GPU (slow downloads)")
def test_model_instantiation():
    try:
        model = get_model(MODEL_NAME, NUM_LABELS)
    except Exception:
        pytest.skip("Model not available in CI environment.")
    assert model.config.num_labels == NUM_LABELS


def test_model_forward_pass():
    try:
        tokenizer = get_tokenizer(MODEL_NAME)
        model = get_model(MODEL_NAME, NUM_LABELS)
    except Exception:
        pytest.skip("Model not available in CI environment.")

    texts = ["good", "bad"]
    encodings = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        outputs = model(**encodings)
    logits = outputs.logits
    assert logits.shape == (2, NUM_LABELS)

