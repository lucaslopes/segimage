"""
Graph builder registry (lazy-load).

We avoid importing heavy dependencies (e.g., python-igraph) at module import
time so ComfyUI can list nodes even if optional deps are missing. Builders are
loaded on demand when `get_graph_builder(name)` is first called.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # only for typing; avoid runtime import
    from igraph import Graph  # pragma: no cover


GraphBuilderFunc = Callable[..., 'Graph']

_REGISTRY: Dict[str, GraphBuilderFunc] = {}


def register_graph_builder(name: str, func: GraphBuilderFunc) -> None:
    key = name.strip().lower()
    _REGISTRY[key] = func


def get_graph_builder(name: str) -> Optional[GraphBuilderFunc]:
    key = name.strip().lower()
    return _REGISTRY.get(key)


def available_graph_builders() -> Dict[str, GraphBuilderFunc]:
    return dict(_REGISTRY)


_LAZY_LOADERS = {
    "grid": lambda: __import__(__name__ + ".grid", fromlist=["build_grid_pixel_graph"]).grid.build_grid_pixel_graph,  # type: ignore[attr-defined]
    "affinity": lambda: __import__(__name__ + ".affinity", fromlist=["build_affinity_pixel_graph"]).affinity.build_affinity_pixel_graph,  # type: ignore[attr-defined]
    "prob4": lambda: __import__(__name__ + ".prob4", fromlist=["build_prob4_pixel_graph"]).prob4.build_prob4_pixel_graph,  # type: ignore[attr-defined]
    "contrast4": lambda: __import__(__name__ + ".contrast4", fromlist=["build_contrast4_pixel_graph"]).contrast4.build_contrast4_pixel_graph,  # type: ignore[attr-defined]
}


def _ensure_builder_loaded(name: str) -> None:
    key = name.strip().lower()
    if key in _REGISTRY:
        return
    loader = _LAZY_LOADERS.get(key)
    if loader is None:
        return
    try:
        func = loader()
        register_graph_builder(key, func)  # type: ignore[arg-type]
    except Exception:
        # Leave unregistered; caller will handle unknown builder
        pass


def get_graph_builder(name: str) -> Optional[GraphBuilderFunc]:
    _ensure_builder_loaded(name)
    key = name.strip().lower()
    return _REGISTRY.get(key)


