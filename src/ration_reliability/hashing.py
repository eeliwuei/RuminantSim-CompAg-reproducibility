"""Deterministic content hashing used for input hashes and run records.

The hash covers types, shapes, dtypes and exact bytes of numpy arrays, and a canonical
JSON-like encoding of scalars / mappings / sequences.  Floats are encoded with ``repr`` so
that two different values never collide through rounding.
"""

from __future__ import annotations

import dataclasses
import enum
import hashlib
from pathlib import Path
from typing import Any, Iterable

import numpy as np

__all__ = ["stable_hash", "file_sha256", "files_sha256"]


def _update(h: "hashlib._Hash", obj: Any) -> None:
    if obj is None:
        h.update(b"N;")
    elif isinstance(obj, bool):
        h.update(b"B" + (b"1" if obj else b"0") + b";")
    elif isinstance(obj, enum.Enum):
        _update(h, obj.value)
    elif isinstance(obj, (int, np.integer)):
        h.update(b"I" + str(int(obj)).encode() + b";")
    elif isinstance(obj, (float, np.floating)):
        h.update(b"F" + repr(float(obj)).encode() + b";")
    elif isinstance(obj, str):
        b = obj.encode("utf-8")
        h.update(b"S" + str(len(b)).encode() + b":" + b + b";")
    elif isinstance(obj, bytes):
        h.update(b"Y" + str(len(obj)).encode() + b":" + obj + b";")
    elif isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        h.update(b"A" + str(arr.dtype.str).encode() + str(arr.shape).encode() + b":")
        if arr.dtype == object:
            for item in arr.ravel():
                _update(h, item)
        else:
            h.update(arr.tobytes())
        h.update(b";")
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        h.update(b"D" + type(obj).__name__.encode() + b"{")
        for f in dataclasses.fields(obj):
            _update(h, f.name)
            _update(h, getattr(obj, f.name))
        h.update(b"}")
    elif isinstance(obj, dict) or hasattr(obj, "items"):
        items = sorted(((str(k), v) for k, v in obj.items()), key=lambda kv: kv[0])
        h.update(b"M{")
        for k, v in items:
            _update(h, k)
            _update(h, v)
        h.update(b"}")
    elif isinstance(obj, (list, tuple)):
        h.update(b"L[")
        for item in obj:
            _update(h, item)
        h.update(b"]")
    elif isinstance(obj, (set, frozenset)):
        h.update(b"T[")
        for item in sorted(obj, key=repr):
            _update(h, item)
        h.update(b"]")
    elif isinstance(obj, Path):
        _update(h, str(obj))
    else:
        raise TypeError(f"stable_hash: unsupported type {type(obj)!r}")


def stable_hash(*objs: Any) -> str:
    """Return the sha256 hex digest of the canonical encoding of ``objs``."""
    h = hashlib.sha256()
    for o in objs:
        _update(h, o)
    return h.hexdigest()


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    """sha256 hex digest of a file's bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def files_sha256(paths: Iterable[str | Path], root: str | Path | None = None) -> str:
    """Hash of a set of files: sha256 over sorted ``relative_path\\0file_sha256\\n`` lines."""
    rootp = Path(root) if root is not None else None
    lines = []
    for p in paths:
        pp = Path(p)
        rel = str(pp.relative_to(rootp)) if rootp is not None else str(pp)
        lines.append(f"{rel}\0{file_sha256(pp)}\n")
    h = hashlib.sha256()
    for line in sorted(lines):
        h.update(line.encode("utf-8"))
    return h.hexdigest()
