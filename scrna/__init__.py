import importlib as _importlib
from . import _config

__version__ = '0.1'

_submodules = [
    'rmt',
    'prc',
    'data',
    'metrics',
    'utils',
    'datasets'
]

__all__ = _submodules + []


def __dir__():
    return __all__


def __getattr__(name):
    if name in _submodules:
        return _importlib.import_module(f"scrna.{name}")
    else:
        try:
            return globals()[name]
        except KeyError:
            raise AttributeError(f"Module 'scrna' has no attribute '{name}'")
