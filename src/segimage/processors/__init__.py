"""
Processors package for segimage.

Defines a simple registry for pluggable processing strategies. Each processor
implements a `run(input_path, output_path, **options) -> bool` method.
"""

from typing import Callable, Dict


ProcessorFunc = Callable[..., bool]


_REGISTRY: Dict[str, ProcessorFunc] = {}


def register_processor(name: str, func: ProcessorFunc) -> None:
    key = name.strip().lower()
    _REGISTRY[key] = func


def get_processor(name: str) -> ProcessorFunc:
    key = name.strip().lower()
    return _REGISTRY.get(key)  # type: ignore[return-value]


def available_processors() -> Dict[str, ProcessorFunc]:
    return dict(_REGISTRY)


# Import built-in processors to ensure they are registered on package import
from . import color_cluster  # noqa: E402,F401

# Optional processors with extra dependencies should be imported lazily
try:  # noqa: SIM105
    from . import slico  # noqa: E402,F401
except Exception:  # If scikit-image is not installed, skip registration silently
    pass


