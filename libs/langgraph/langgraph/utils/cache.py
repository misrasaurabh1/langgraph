from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any


def _freeze(obj: Any, depth: int = 10) -> Hashable:
    # Optimize fast path for common immutable/primitive types
    if obj is None or isinstance(obj, (int, float, complex, str, bytes)):
        return obj
    # Early exit if depth exhausted
    if depth <= 0:
        return obj
    # Avoid boolean being treated as an int (since bool is subclass of int)
    if isinstance(obj, bool):
        return obj
    # Fast path for pre-hashable types
    if isinstance(obj, Hashable):
        return obj
    # Optimize Mapping path
    if isinstance(obj, Mapping):
        # Sort keys so {"a":1,"b":2} == {"b":2,"a":1}
        # Don't use generator+sorted+tuple (costly on small dicts)
        out = []
        for k, v in obj.items():
            out.append((k, _freeze(v, depth - 1)))
        out.sort()
        return tuple(out)
    # Optimize Sequence but avoid str, bytes, bytearray
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes, bytearray)):
        out = [_freeze(x, depth - 1) for x in obj]
        return tuple(out)
    # numpy / pandas etc. can provide their own .tobytes()
    if hasattr(obj, "tobytes"):
        return (
            type(obj).__name__,
            obj.tobytes(),
            getattr(obj, "shape", None),
        )
    return obj  # strings, ints, dataclasses with frozen=True, etc.


def default_cache_key(*args: Any, **kwargs: Any) -> str | bytes:
    """Default cache key function that uses the arguments and keyword arguments to generate a hashable key."""
    import pickle

    # protocol 5 strikes a good balance between speed and size
    return pickle.dumps((_freeze(args), _freeze(kwargs)), protocol=5, fix_imports=False)
