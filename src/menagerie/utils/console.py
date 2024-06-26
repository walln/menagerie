"""Utilities for rich console output."""

from rich.console import Console
from rich.progress import MofNCompleteColumn, Progress, SpinnerColumn
from rich.traceback import install

install(show_locals=False)

console = Console()

progress_columns = [
    SpinnerColumn(),
    *Progress.get_default_columns(),
    MofNCompleteColumn(),
]
