"""
Graph builder registry.

Pipelines call `get_graph_builder(name)` to obtain a function that converts
an image (or superpixels) into a graph. Builders self-register on import.

Built-in builders (imported below to trigger registration):
- grid: 8-neighbor pixel grid with optional edge filtering (lbp/gray/rgb)
- affinity: Gaussian affinity within a radius using L*a*b* or intensity
- prob4: initial 4-connected pixel graph (exp(-||ΔI||^2 / σ_I^2))
- contrast4: initial 4-connected pixel graph (exp(-α·|ΔI|))
- superpixel_adjacency: adjacency graph between SLICO segments
"""

from __future__ import annotations

from typing import Callable, Dict, Optional

from igraph import Graph


GraphBuilderFunc = Callable[..., Graph]

_REGISTRY: Dict[str, GraphBuilderFunc] = {}


def register_graph_builder(name: str, func: GraphBuilderFunc) -> None:
    key = name.strip().lower()
    _REGISTRY[key] = func


def get_graph_builder(name: str) -> Optional[GraphBuilderFunc]:
    key = name.strip().lower()
    return _REGISTRY.get(key)


def available_graph_builders() -> Dict[str, GraphBuilderFunc]:
    return dict(_REGISTRY)
def available_graph_builders() -> Dict[str, GraphBuilderFunc]:
    return dict(_REGISTRY)


# Import built-in builders so they register themselves
from . import grid  # noqa: E402,F401
from . import affinity  # noqa: E402,F401
from . import prob4  # noqa: E402,F401
from . import contrast4  # noqa: E402,F401
# superpixel_adjacency is now subsumed by node_mode='superpixel' in the above builders


