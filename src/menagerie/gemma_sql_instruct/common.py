"""Common variables and functions for the gemma_sql_instruct module."""

from datasets import Dataset, load_dataset

# Convert dataset to OAI messages
system_message = """You are an text to SQL query translator. Users will ask you questions in English and you will generate a SQL query based on the provided SCHEMA.
SCHEMA:
{schema}"""


def create_conversation(sample):
    """Convert a sample to a conversation in the OpenAI format.

    Args:
      sample: A sample from the dataset.

    Returns:
      An OpenAI conversation.

    """
    return {
        "messages": [
            {
                "role": "system",
                "content": system_message.format(schema=sample["context"]),
            },
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]
    }


def create_dataset(dataset_size: int = 4000, test_split_size: int = 1000):
    """Create a training and test dataset for the gemma_sql_instruct model.

    Args:
      dataset_size: The size of the dataset.
      test_split_size: The size of the test split.

    Returns:
    A tuple of the training and test datasets.
    """
    assert dataset_size > test_split_size

    dataset = load_dataset("b-mc2/sql-create-context", split="train")
    assert isinstance(dataset, Dataset)

    dataset = dataset.shuffle().select(range(dataset_size))

    feature_names = list(dataset.features.keys())
    # Convert dataset to OAI messages
    dataset = dataset.map(
        create_conversation, remove_columns=feature_names, batched=False
    )
    dataset = dataset.train_test_split(test_size=test_split_size / dataset_size)

    # interleave_datasets() is used to combine the training and test datasets

    train_dataset = dataset["train"]
    test_dataset = dataset["test"]

    return train_dataset, test_dataset
