"""Wikitext dataset [https://huggingface.co/datasets/wikitext]."""
from typing import Dict

import torch
from transformers import AutoTokenizer

import datasets
from datasets import load_dataset

_dataset = "wikitext"
_subset = "wikitext-2-v1"
_tokenizer_name = "bert-base-uncased"

tokenizer = AutoTokenizer.from_pretrained(_tokenizer_name)


def _filter(row: Dict):
    if row["text"] == "":
        return False
    return True


def _transform(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def create_train_dataset():
    """Create the train dataset."""
    dataset = (
        load_dataset(_dataset, _subset, split="train", streaming=False)
        .filter(_filter)
        .map(_transform, batched=True)
    )

    if not isinstance(dataset, datasets.Dataset):
        raise ValueError("Expected a Dataset.")

    dataset.set_format(
        type="torch", columns=["input_ids", "token_type_ids", "attention_mask"]
    )

    return dataset


def create_validation_dataset():
    """Create the validation dataset."""
    dataset = (
        load_dataset(_dataset, _subset, split="validation", streaming=False)
        .filter(_filter)
        .map(_transform, batched=True)
    )
    if not isinstance(dataset, datasets.Dataset):
        raise ValueError("Expected a Dataset.")

    dataset.set_format(
        "torch", columns=["input_ids", "token_type_ids", "attention_mask"]
    )

    return dataset


class _WikitextCollator:
    def __call__(self, batch):
        # Extract input_ids from the batch
        input_ids = [example["input_ids"] for example in batch]

        # Prepare input and target sequences
        # For input, take all tokens except the last
        # For target, take all tokens except the first
        input_seqs = [ids[:-1] for ids in input_ids]
        target_seqs = [ids[1:] for ids in input_ids]

        # Pad sequences to the same length and stack them
        input_seqs = torch.nn.utils.rnn.pad_sequence(
            input_seqs, batch_first=True, padding_value=0
        )
        target_seqs = torch.nn.utils.rnn.pad_sequence(
            target_seqs, batch_first=True, padding_value=0
        )

        return input_seqs, target_seqs


wikitext_collator = _WikitextCollator()
