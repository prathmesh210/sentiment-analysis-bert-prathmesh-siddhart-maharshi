# tests/unit/test_data_processing.py
import pandas as pd
from src import data_processing as dp


class FakeTokenizer:
    """
    Minimal fake tokenizer for unit tests.
    It returns predictable input_ids and attention_mask shapes.
    """
    def __call__(self, texts, padding=True, truncation=True, max_length=128, return_attention_mask=True):
        # produce a token id for each char to keep it simple; pad to length 5 for test predictability
        input_ids = []
        attention_mask = []
        for t in texts:
            tokens = [ord(c) % 100 for c in str(t)][:3]  # tiny pseudo-token list
            attention = [1] * len(tokens)
            # pad to length 5
            tokens += [0] * (5 - len(tokens))
            attention += [0] * (5 - len(attention))
            input_ids.append(tokens)
            attention_mask.append(attention)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_clean_text_basic():
    assert dp.clean_text("  HELLO  WORLD!! ") == "hello world!!"
    assert dp.clean_text(None) == ""
    # remove weird chars
    assert "@" not in dp.clean_text("hi@there#") 
    # collapse spaces
    assert dp.clean_text("a    b") == "a b"


def test_tokenizer_creates_input_ids_and_attention_mask():
    s = pd.Series(["hello", "test"])
    fake = FakeTokenizer()
    enc = dp.tokenize_series(s, tokenizer=fake, max_length=10)
    assert "input_ids" in enc and "attention_mask" in enc
    assert isinstance(enc["input_ids"], list)
    assert len(enc["input_ids"]) == 2
    assert len(enc["attention_mask"][0]) == 5  # our fake pads to length 5


def test_train_val_split_sizes_and_labels():
    # make 10 rows
    df = pd.DataFrame({"text": [f"t{i}" for i in range(10)], "label": [i % 2 for i in range(10)]})
    fake = FakeTokenizer()
    train, val = dp.train_val_split(df, text_col="text", label_col="label", tokenizer=fake, test_size=0.2, random_state=0)
    assert len(train["input_ids"]) == 8
    assert len(val["input_ids"]) == 2
    assert "labels" in train and "labels" in val
    assert len(train["labels"]) == 8
    assert len(val["labels"]) == 2