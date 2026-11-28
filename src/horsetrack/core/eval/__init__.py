"""Evaluation: walk-forward folds, the locked holdout and its use log, segment ROI shrinkage,
the tail-return curve by claimed value, the run registry, the report writer, and scoring with
the paired bootstrap (problem statement section 3)."""

from horsetrack.core.eval.tail import tail_curve

__all__ = ["tail_curve"]
