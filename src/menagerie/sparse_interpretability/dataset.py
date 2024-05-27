"""Dataset utilities for sparse interpretability experiments."""

import einops
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer

from datasets import IterableDataset
from menagerie.sparse_interpretability.tracer import StopForward, Trace


def compute_activations(
    dataset: IterableDataset,
    model: nn.Module,
    layer_name: str,
    batch_size: int,
) -> IterableDataset:
    """Compute activations from a dataset.

    This function computes activations from input data at a given layer in the
    provided model. The activations

    Args:
        dataset: Dataset to compute activations from
        model: Model to compute activations from
        layer_name: Name of the layer to compute activations from
        batch_size: Batch size for computing activations

    Returns:
        precomputed_dataset: Dataset with activations computed
    """
    device = "cuda"
    model.to(device)
    model.eval()

    @torch.no_grad()
    def _get_activations(batch):
        tokens = batch["tokens"].to(device)
        try:
            with Trace(model, layer_name, stop=True, detach=True) as ret:
                _ = model(tokens)
                mlp_activation_data = ret.output.to(device)  # type: ignore
        except StopForward:
            ...
        mlp_activation_data = ret.output.to(device)  # type: ignore
        mlp_activation_data = nn.functional.gelu(mlp_activation_data)

        _, seq_len = tokens.shape
        activations = mlp_activation_data.reshape(batch_size, seq_len, -1)

        return {
            "activations": list(torch.unbind(activations.clone().detach(), dim=0)),
            "tokens": list(torch.unbind(tokens.clone().detach(), dim=0)),
        }

    columns_to_remove = [
        column
        for column in (dataset.column_names if dataset.column_names is not None else [])
        if column not in ["tokens", "activations"]
    ]

    precomputed_dataset = dataset.map(
        _get_activations,
        batched=True,
        batch_size=batch_size,
        remove_columns=columns_to_remove,
    ).with_format("torch")

    return precomputed_dataset


def tokenize_and_concat(
    dataset: IterableDataset,
    tokenizer: AutoTokenizer,
    *,
    max_length: int = 1024,
    column_name: str = "text",
    add_bos_token: bool = True,
) -> IterableDataset:
    """Helper function to tokenize and concatenate text columns in a dataset.

    This function after tokenizing the texts concatenates them (separated by EOS tokens)
    and then reshapes them into a 2D array of shape (____, sequence_length), dropping
    the last batch. Tokenizers are much faster if parallelised, so we chop the string
    into 20, feed it into the tokenizer, in parallel with padding, then remove padding
    at the end.

    This tokenization is useful for training language models, as it allows us to
    efficiently train on a large corpus of text of varying lengths (without,
    eg, a lot of truncation or padding). Further, for models with absolute positional
    encodings, this avoids privileging early tokens (eg, news articles often begin
    with CNN, and models may learn to use early positional encodings to predict these)

    This is adapted from transformer_lens:
    https://github.com/TransformerLensOrg/TransformerLens

    Args:
        dataset: Dataset to tokenize
        tokenizer: Tokenizer to use
        streaming: Whether to stream the tokenization
        max_length: Maximum length of the concatenated sequence
        column_name: Name of the column to tokenize
        add_bos_token: Whether to add a BOS token to the beginning of each text
        num_proc: Number of processes to use for tokenization
    """
    column_names = dataset.column_names
    assert column_names is not None, "Dataset must have column names"
    dataset = dataset.remove_columns(
        [col for col in column_names if col != column_name]
    )
    if tokenizer.pad_token is None:  # type: ignore
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})  # type: ignore

    seq_len = max_length - 1 if add_bos_token else max_length

    def tokenize(examples: dict[str, list[str]]) -> dict[str, np.ndarray]:
        """Tokenize a batch of examples."""
        text = examples[column_name]
        full_text = tokenizer.eos_token.join(text)  # type: ignore
        num_chunks = 20
        chunk_lengths = (len(full_text) - 1) // num_chunks + 1
        chunks = [
            full_text[i * chunk_lengths : (i + 1) * chunk_lengths]
            for i in range(num_chunks)
        ]

        tokens = tokenizer(chunks, return_tensors="np", padding=True)[  # type: ignore
            "input_ids"
        ].flatten()
        tokens = tokens[tokens != tokenizer.pad_token_id]  # type: ignore

        num_tokens = len(tokens)
        num_batches = num_tokens // seq_len

        tokens = tokens[: seq_len * num_batches]
        tokens = einops.rearrange(
            tokens, "(batch seq) -> batch seq", batch=num_batches, seq=seq_len
        )

        if add_bos_token:
            prefix = np.full((num_batches, 1), tokenizer.bos_token_id)  # type: ignore
            tokens = np.concatenate([prefix, tokens], axis=1)
        return {"tokens": tokens}

    tokenized_dataset = dataset.map(
        tokenize,
        batched=True,
        remove_columns=[column_name],
    ).with_format("torch")
    return tokenized_dataset
