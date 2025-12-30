"""EDA CLI - Exploratory Data Analysis Command Line Interface."""

from .core import (
    summarize_dataset,
    missing_table,
    correlation_matrix,
    compute_quality_flags,
)

__version__ = "0.1.0"
__all__ = [
    "summarize_dataset",
    "missing_table",
    "correlation_matrix",
    "compute_quality_flags",
]
