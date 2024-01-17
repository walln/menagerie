"""HF simplified code parrot dataset."""

from transformers import AutoTokenizer, DataCollatorForLanguageModeling

from datasets import Dataset, DatasetDict, load_dataset


def create_datasets(context_length=128, seed=42, *, sanity_check_data: bool = False):
    """Create code parrot datasets."""
    ds_train = load_dataset("huggingface-course/codeparrot-ds-train", split="train")
    ds_valid = load_dataset(
        "huggingface-course/codeparrot-ds-valid", split="validation"
    )

    if not isinstance(ds_train, Dataset) or not isinstance(ds_valid, Dataset):
        raise ValueError("Expected a Dataset.")

    raw_datasets = DatasetDict(
        {
            "train": ds_train.shuffle(seed=seed).select(range(50000)),
            "valid": ds_valid.shuffle(seed=seed).select(range(500)),
        }
    )

    # Sanity check example metadata
    if sanity_check_data:
        for key in raw_datasets["train"][0]:
            print(f"{key.upper()}: {raw_datasets['train'][0][key][:200]}")

    tokenizer = AutoTokenizer.from_pretrained(
        "huggingface-course/code-search-net-tokenizer"
    )

    outputs = tokenizer(
        raw_datasets["train"][:2]["content"],
        truncation=True,
        max_length=context_length,
        return_overflowing_tokens=True,
        return_length=True,
    )

    if sanity_check_data:
        print(f"Input IDs length: {len(outputs['input_ids'])}")  # type: ignore
        print(f"Input chunk lengths: {(outputs['length'])}")
        print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")

    def _tokenize(element):
        outputs = tokenizer(
            element["content"],
            truncation=True,
            max_length=context_length,
            return_overflowing_tokens=True,
            return_length=True,
        )
        input_batch = []
        for length, input_ids in zip(outputs["length"], outputs["input_ids"]):  # type: ignore
            if length == context_length:
                input_batch.append(input_ids)
        return {"input_ids": input_batch}

    tokenized_datasets = raw_datasets.map(
        _tokenize, batched=True, remove_columns=raw_datasets["train"].column_names
    )

    if sanity_check_data:
        print(tokenized_datasets)

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    return tokenized_datasets, tokenizer, data_collator
