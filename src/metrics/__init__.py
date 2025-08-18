"""Metrics package initialization."""

from .quantile_loss import (
    pinball_loss,
    compute_quantile_losses,
    aggregate_losses,
    create_loss_summary_table,
    compare_model_losses
)
from .diebold_mariano import (
    diebold_mariano_test,
    dm_test_by_groups,
    aggregate_dm_tests,
    multiple_testing_correction
)

__all__ = [
    "pinball_loss",
    "compute_quantile_losses", 
    "aggregate_losses",
    "create_loss_summary_table",
    "compare_model_losses",
    "diebold_mariano_test",
    "dm_test_by_groups",
    "aggregate_dm_tests",
    "multiple_testing_correction"
]
