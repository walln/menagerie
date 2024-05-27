"""Rich console manager for the sparse interpretability module."""

from rich.console import Console, Group
from rich.panel import Panel
from rich.progress import MofNCompleteColumn, Progress, SpinnerColumn, TimeElapsedColumn

console = Console()

activation_buffer_refresh_progress = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
    MofNCompleteColumn(),
    console=console,
)

sae_training_progress = Progress(
    SpinnerColumn(),
    *Progress.get_default_columns(),
    TimeElapsedColumn(),
    MofNCompleteColumn(),
    console=console,
)

# group of progress bars;
# some are always visible, others will disappear when progress is complete
progress_group = Group(
    Panel(Group(activation_buffer_refresh_progress, sae_training_progress)),
)
