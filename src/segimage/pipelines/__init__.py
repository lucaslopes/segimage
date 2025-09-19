"""
Pipelines package for segimage.

Defines a simple registry for pluggable processing pipelines. Each pipeline
implements a `run(input_path, output_path, **options) -> bool` method.
"""

from typing import Callable, Dict


PipelineFunc = Callable[..., bool]


_REGISTRY: Dict[str, PipelineFunc] = {}


def register_pipeline(name: str, func: PipelineFunc) -> None:
    key = name.strip().lower()
    _REGISTRY[key] = func


def get_pipeline(name: str) -> PipelineFunc:
    key = name.strip().lower()
    return _REGISTRY.get(key)  # type: ignore[return-value]


def available_pipelines() -> Dict[str, PipelineFunc]:
    return dict(_REGISTRY)


# Import built-in pipelines to ensure they are registered on package import
from . import slico_graph  # noqa: E402,F401
from . import graph_hedonic  # noqa: E402,F401
from . import slico_graph_hedonic  # noqa: E402,F401
from . import graph_view  # noqa: E402,F401



