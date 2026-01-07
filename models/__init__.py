"""Top-level package for the project's model modules.

This file makes the ``models`` directory a proper Python package so
imports like ``from models import layers`` work consistently.
"""

# Expose commonly used submodules for convenience
__all__ = [
    "layers",
    "sparse_embedding",
    "common",
    "losses",
    "recursive_reasoning",
]
