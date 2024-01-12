"""Dataset utilities."""
from menagerie.utils.console import progress_columns
from rich.progress import Progress
from torch.utils.data.dataloader import DataLoader


def sanity_check_data(train_dataloader: DataLoader, valid_dataloader: DataLoader):
    """Sanity check the data."""
    with Progress(*progress_columns) as progress:
        for _ in progress.track(
            train_dataloader, description="[purple]Training Data Sanity Check"
        ):
            pass

        for _ in progress.track(
            valid_dataloader, description="[purple]Validation Data Sanity Check"
        ):
            pass
