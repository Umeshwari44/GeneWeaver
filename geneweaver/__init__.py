"""GeneWeaver: GPU-accelerated CRISPR alignment (off-target search) engine."""
__version__ = "0.1.0"

from .backend import get_backend
from .engine import Hit, OffTargetSearcher
from .scoring import guide_specificity, mit_off_target_score

__all__ = ["get_backend", "Hit", "OffTargetSearcher", "mit_off_target_score", "guide_specificity"]
