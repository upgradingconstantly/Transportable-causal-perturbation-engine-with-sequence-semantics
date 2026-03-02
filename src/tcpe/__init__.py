"""Transportable Causal Perturbation Engine package."""

from importlib.metadata import PackageNotFoundError, version
from typing import Any

from .runtime.seed import set_global_seed

try:
    __version__ = version("tcpe")
except PackageNotFoundError:
    __version__ = "0.1.0"

def validate_anndata_schema(*args: Any, **kwargs: Any) -> Any:
    """Lazy export to avoid importing optional schema dependencies at package import time."""
    from .anndata_schema import validate_anndata_schema as _validate_anndata_schema

    return _validate_anndata_schema(*args, **kwargs)


__all__ = ["__version__", "set_global_seed", "validate_anndata_schema"]
