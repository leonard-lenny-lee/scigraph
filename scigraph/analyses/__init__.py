"""Analyses available for scigraph data tables."""

from scigraph.analyses._desc_stats import DescriptiveStatistics
from scigraph.analyses._normalize import Normalize
from scigraph.analyses._row_stats import RowStatistics

__all__ = ["DescriptiveStatistics", "Normalize", "RowStatistics"]
