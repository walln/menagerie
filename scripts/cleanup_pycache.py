"""Delete all cache directories in the current directory."""

import os
import shutil


def find_cache_directories(start_path, cache_names, exclude_dirs):
    """Find all cache directories in the given path."""
    cache_dirs = []
    for root, dirs, _ in os.walk(start_path):
        # Skip excluded directories
        if any(excluded in root for excluded in exclude_dirs):
            continue
        for name in cache_names:
            if name in dirs:
                full_path = os.path.join(root, name)
                cache_dirs.append(full_path)
    return cache_dirs


def delete_all_at_once(directories):
    """Delete all directories at once."""
    for directory in directories:
        shutil.rmtree(directory)
        print(f"Deleted {directory}")


if __name__ == "__main__":
    current_directory = os.getcwd()
    cache_names = ["__pycache__", ".pytest_cache"]
    exclude_dirs = [".git", ".venv", "data", "logs", "wandb"]

    print(f"Scanning for cache directories in {current_directory}...")
    cache_dirs = find_cache_directories(current_directory, cache_names, exclude_dirs)

    if cache_dirs:
        print("The following cache directories will be deleted:")
        for directory in cache_dirs:
            print(directory)
        confirm = input("Do you want to delete all these directories? [y/N] ")
        if confirm.lower() == "y":
            delete_all_at_once(cache_dirs)
            print("All specified cache directories have been deleted.")
        else:
            print("Deletion cancelled.")
    else:
        print("No cache directories found.")
