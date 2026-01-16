# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .cumulative_return import cumulative_return_graph
from .score_ic import score_ic_graph
from .report import report_graph
from .rank_label import rank_label_graph
from .risk_analysis import risk_analysis_graph

# Pyecharts versions (optional import)
try:
    from .risk_analysis_pyecharts import risk_analysis_graph_pyecharts
except ImportError:
    # pyecharts is not installed
    risk_analysis_graph_pyecharts = None


__all__ = [
    "cumulative_return_graph",
    "score_ic_graph",
    "report_graph",
    "rank_label_graph",
    "risk_analysis_graph",
    "risk_analysis_graph_pyecharts"
]
