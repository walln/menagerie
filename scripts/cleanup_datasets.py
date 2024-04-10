"""Clean up datasets in the cache directory."""

import os
import shutil


def list_datasets(cache_dir):
    """List all datasets in the cache directory."""
    if not os.path.exists(cache_dir):
        print("Cache directory does not exist.")
        return []

    datasets = [
        d for d in os.listdir(cache_dir) if os.path.isdir(os.path.join(cache_dir, d))
    ]
    return datasets


def delete_dataset(cache_dir, dataset):
    """Delete a specific dataset."""
    dataset_path = os.path.join(cache_dir, dataset)
    if os.path.exists(dataset_path):
        shutil.rmtree(dataset_path)
        print(f"Deleted dataset: {dataset}")
    else:
        print(f"Dataset not found: {dataset}")


def main():
    """Clean up datasets in the cache directory."""
    CACHE_DIR = os.path.expanduser("~/.cache/huggingface/datasets")

    print("Listing datasets in cache...\n")
    datasets = list_datasets(CACHE_DIR)

    if not datasets:
        print("No datasets found in cache.")
        return

    print("Found the following datasets:")
    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset}")

    choice = input("\nDo you want to delete any datasets? (yes/no): ").lower()
    if choice != "yes":
        return

    to_delete = input(
        "Enter the numbers of the datasets to delete (separated by space): "
    )
    to_delete_indices = [int(x) - 1 for x in to_delete.split()]

    for index in to_delete_indices:
        if 0 <= index < len(datasets):
            delete_dataset(CACHE_DIR, datasets[index])
        else:
            print(f"Invalid dataset number: {index + 1}")


if __name__ == "__main__":
    main()
