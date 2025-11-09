from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
from torch.utils.data import Dataset
import torch


MODEL_NAME = "distilbert-base-uncased"  # you can change to "bert-base-uncased"
NUM_LABELS = 2


def get_tokenizer(model_name: str = MODEL_NAME):
    return AutoTokenizer.from_pretrained(model_name)


def get_model(model_name: str = MODEL_NAME, num_labels: int = NUM_LABELS):
    return AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )


class SimpleTextDataset(Dataset):
    """
    Tiny dataset wrapper for training.
    Expects a list of tokenized inputs (dicts) and list of labels.
    """
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def train_model(train_texts, train_labels, val_texts=None, val_labels=None, output_dir: str = "./model_out"):
    """
    Fine-tune a pretrained model on provided texts and labels.
    This is a simple example — you can expand it later.
    """
    tokenizer = get_tokenizer()
    train_encodings = tokenizer(
        train_texts,
        truncation=True,
        padding=True
    )

    train_dataset = SimpleTextDataset(train_encodings, train_labels)

    model = get_model()

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,
        num_train_epochs=1,
        logging_steps=10,
        save_strategy="no"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset
    )

    trainer.train()
    return model

