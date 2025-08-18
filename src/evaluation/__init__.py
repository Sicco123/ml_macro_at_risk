"""Evaluation package initialization."""

from .aggregator import (
    aggregate_loss_by_dimension,
    create_performance_ranking,
    create_summary_statistics,
    create_model_comparison_table,
    compute_coverage_statistics,
    create_comprehensive_report
)
from .plotting import (
    plot_training_curves,
    plot_forecast_paths,
    plot_loss_comparison,
    plot_calibration,
    plot_factor_correlations,
    create_evaluation_dashboard
)

__all__ = [
    "aggregate_loss_by_dimension",
    "create_performance_ranking",
    "create_summary_statistics",
    "create_model_comparison_table",
    "compute_coverage_statistics",
    "create_comprehensive_report",
    "plot_training_curves",
    "plot_forecast_paths",
    "plot_loss_comparison",
    "plot_calibration",
    "plot_factor_correlations",
    "create_evaluation_dashboard"
]
