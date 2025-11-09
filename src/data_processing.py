# src/data_processing.py
import re
from typing import Optional, Tuple, Dict, Any, List

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def clean_text(text: Optional[str]) -> str:
    """
    Simple cleaning:
    - convert to str
    - lowercase
    - collapse whitespace
    - remove most non-alphanumeric characters except simple punctuation
    """
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    s = str(text).lower()
    s = re.sub(r"\s+", " ", s)  # collapse whitespace
    # keep letters, numbers, and common punctuation . , ' " ! ?
    s = re.sub(r"[^a-z0-9\s\.\,\'\"!\?]", "", s)
    return s.strip()


def load_tokenizer(name: str = "bert-base-uncased") -> PreTrainedTokenizerBase:
    """
    Load and return a Hugging Face tokenizer.
    """
    return AutoTokenizer.from_pretrained(name, use_fast=True)


def tokenize_series(
    texts: pd.Series,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int = 128,
) -> Dict[str, List[List[int]]]:
    """
    Tokenize a pandas Series of raw texts and return a dict of lists:
    {'input_ids': [[...], [...]], 'attention_mask': [[...], [...]]}
    This returns python lists (not tensors) to keep tests simple and avoid device issues.
    """
    texts_list = texts.fillna("").astype(str).tolist()
    enc = tokenizer(
        texts_list,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True,
    )
    # enc fields are usually lists of lists (or tensors), convert to lists explicitly
    return {
        "input_ids": [list(x) for x in enc["input_ids"]],
        "attention_mask": [list(x) for x in enc["attention_mask"]],
    }


def train_val_split(
    df: pd.DataFrame,
    text_col: str,
    label_col: Optional[str],
    tokenizer: PreTrainedTokenizerBase,
    test_size: float = 0.2,
    random_state: int = 42,
    max_length: int = 128,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Tokenize texts and split into train/validation dicts.
    Each returned dict contains at least 'input_ids' and 'attention_mask' as lists of lists.
    If label_col provided, also returns 'labels' as a list.
    """
    if text_col not in df.columns:
        raise ValueError(f"text_col '{text_col}' not found in df")

    enc = tokenize_series(df[text_col].apply(clean_text), tokenizer, max_length=max_length)

    indices = list(range(len(df)))
    train_idx, val_idx = train_test_split(indices, test_size=test_size, random_state=random_state, shuffle=True)

    def subset(encodings: Dict[str, List[List[int]]], idxs: List[int]) -> Dict[str, Any]:
        return {k: [encodings[k][i] for i in idxs] for k in encodings}

    train = subset(enc, train_idx)
    val = subset(enc, val_idx)

    if label_col:
        if label_col not in df.columns:
            raise ValueError(f"label_col '{label_col}' not found in df")
        labels = df[label_col].astype(int).tolist()
        train["labels"] = [labels[i] for i in train_idx]
        val["labels"] = [labels[i] for i in val_idx]

    return train, val


if __name__ == "__main__":
    # tiny demo if you run the file directly
    import pandas as pd

    sample = pd.DataFrame(
        {
            "text": [
                "I LOVE this movie!!!   ",
                "This is trash... not good",
                None,
                "Ok-ish, could be better :)",
            ],
            "label": [1, 0, 0, 1],
        }
    )

    tok = load_tokenizer("bert-base-uncased")
    train, val = train_val_split(sample, text_col="text", label_col="label", tokenizer=tok, test_size=0.25)
    print("Train sizes:", len(train["input_ids"]), "Val sizes:", len(val["input_ids"]))