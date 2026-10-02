"""Named, independent random streams (contract T5/T6).

Every random quantity is drawn from a :class:`numpy.random.Generator` obtained from a
:class:`numpy.random.SeedSequence` whose ``spawn_key`` encodes the *purpose* of the stream::

    SeedSequence(entropy=root_seed, spawn_key=(stream_index, *sub_keys))

Core stream names and fixed indexes (never renumber; they are part of the run record):

============  =====  ==========================================================
name          index  purpose
============  =====  ==========================================================
``opt``       0      optimisation draws (SAA scenarios, draw-mean coefficients)
``validation``1      development selection (e.g. safety-margin calibration)
``test``      2      frozen hold-out simulation stream (after protocol freeze)
``outer``     3      outer parameter draws (parameter-estimation uncertainty)
============  =====  ==========================================================

Other names (e.g. an assay-signal stream) get a stable index derived from the name
(``2**32 + first 4 bytes of sha256(name)``), so adding a stream never shifts the core four.
Sub-keys distinguish replicates (e.g. ``generator("outer", 17)`` for outer draw 17).
Streams are independent of the order in which they are created.

Reserved formal-evaluation roots (official-run plan batch 4; streams policy SP-3 / SP-4; closes SB-05)
----------------------------------------------------------------------------------------------------
``configs/streams_policy.yaml`` ``reserved_formal_streams`` reserves three roots for the official evaluation, fixed by
a label rule before any result existed: ``root_k = int.from_bytes(sha256(label_k).digest()[:8], "big") >> 1`` with
``label_k = RESERVED_ROOT_LABEL_TEMPLATE.format(k=k)``.  :func:`reserved_formal_roots` derives them from that rule
(sha256 only; no configuration file is read; a test compares them with the policy file by integer comparison).  A
reserved root is spent on its first draw (SP-4), so :class:`RandomStreams` refuses one **at construction** unless it is
given an :class:`OfficialStreamToken` for that root -- and only ``official_v2.require_protocol_frozen()``
(``experiments/E1_cost_reliability/official_v2.py``) can mint a token (:func:`mint_official_stream_token` checks its
caller; a token carries an HMAC seal under a per-process key, so a hand-built or copied-in token is refused).  The
refusal messages never contain a root value.  This is a guard against accidents and shortcuts inside this code base,
not a security boundary against deliberate circumvention.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np

__all__ = ["CORE_STREAMS", "stream_index", "RandomStreams", "RESERVED_ROOT_LABEL_TEMPLATE", "N_RESERVED_ROOTS",
           "derive_reserved_root", "reserved_formal_roots", "is_reserved_root", "reserved_root_index",
           "ReservedRootError", "OfficialStreamToken", "mint_official_stream_token", "OFFICIAL_TOKEN_MINTER"]

#: Core stream name -> fixed spawn index.
CORE_STREAMS: dict[str, int] = {"opt": 0, "validation": 1, "test": 2, "outer": 3}


def stream_index(name: str) -> int:
    """Spawn index of a stream name (core names fixed; others hashed, >= 2**32)."""
    if not name or "/" in name:
        raise ValueError(f"illegal stream name {name!r}")
    if name in CORE_STREAMS:
        return CORE_STREAMS[name]
    return (1 << 32) + int.from_bytes(hashlib.sha256(name.encode("utf-8")).digest()[:4], "big")


# ---------------------------------------------------------------------------------------------------------------------
# reserved formal-evaluation roots (streams policy reserved_formal_streams; official-run plan batch 4, F3)
# ---------------------------------------------------------------------------------------------------------------------
#: label rule of configs/streams_policy.yaml reserved_formal_streams.generation (fixed 2026-09-25, before any result)
RESERVED_ROOT_LABEL_TEMPLATE = "ration_reliability_study/formal_evaluation/root/v1/k={k}"
#: number of reserved roots (k = 0, 1, 2)
N_RESERVED_ROOTS = 3
#: the one function allowed to mint an :class:`OfficialStreamToken`: (function name, file path suffix)
OFFICIAL_TOKEN_MINTER = ("require_protocol_frozen", "experiments/E1_cost_reliability/official_v2.py")


class ReservedRootError(PermissionError):
    """A reserved formal-evaluation root was requested without a valid official token (the value is withheld)."""


def derive_reserved_root(k: int) -> int:
    """Reserved root ``k`` from the label rule (sha256 only; never printed by this module)."""
    if isinstance(k, bool) or not isinstance(k, int) or not 0 <= k < N_RESERVED_ROOTS:
        raise ValueError(f"reserved root index must be one of 0..{N_RESERVED_ROOTS - 1}")
    label = RESERVED_ROOT_LABEL_TEMPLATE.format(k=k).encode("ascii")
    return int.from_bytes(hashlib.sha256(label).digest()[:8], "big") >> 1


def reserved_formal_roots() -> tuple[int, ...]:
    """The reserved roots in index order k = 0, 1, 2, derived from the label rule (no configuration file is read).

    Looked up at call time by the guard (tests replace it with a synthetic set; the real roots are never used to
    create a generator outside the official path)."""
    return tuple(derive_reserved_root(k) for k in range(N_RESERVED_ROOTS))


def is_reserved_root(seed: Any) -> bool:
    """Whether ``seed`` is one of :func:`reserved_formal_roots` (integer comparison)."""
    try:
        s = int(seed)
    except (TypeError, ValueError):
        return False
    return s in set(reserved_formal_roots())


def reserved_root_index(seed: Any) -> Optional[int]:
    """Index k of a reserved root, ``None`` for any other seed."""
    try:
        s = int(seed)
    except (TypeError, ValueError):
        return None
    roots = reserved_formal_roots()
    return roots.index(s) if s in roots else None


_SEAL_KEY = secrets.token_bytes(32)          # per process: a token never crosses a process boundary
_TOKEN_FIELDS = ("head_commit", "freeze_record_sha256", "protocol_sha256", "spec_digest", "roots_k", "issued_utc")


def _seal(values: dict[str, Any]) -> str:
    msg = json.dumps([values[k] if k != "roots_k" else [int(x) for x in values[k]] for k in _TOKEN_FIELDS],
                     sort_keys=True, ensure_ascii=True).encode("ascii")
    return hmac.new(_SEAL_KEY, msg, hashlib.sha256).hexdigest()


@dataclass(frozen=True)
class OfficialStreamToken:
    """Permission to open the reserved roots ``roots_k`` in this process, issued by
    ``official_v2.require_protocol_frozen()`` after every freeze gate passed (official-run plan F3).  Carries the
    evidence it was issued on (git HEAD, sha256 of ``configs/protocol_freeze.json``, protocol sha256, specification
    digest) and an HMAC seal; constructing one by hand raises :class:`ReservedRootError`."""

    head_commit: str
    freeze_record_sha256: str
    protocol_sha256: str
    spec_digest: str
    roots_k: tuple[int, ...]
    issued_utc: str
    seal: str = field(repr=False, compare=False, default="")

    def __post_init__(self) -> None:
        if not self.valid():
            raise ReservedRootError("OfficialStreamToken: not issued by official_v2.require_protocol_frozen() "
                                    "(seal missing or wrong)")

    def _values(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in _TOKEN_FIELDS}

    def valid(self) -> bool:
        """The seal matches this process's key and the token's fields."""
        return isinstance(self.seal, str) and hmac.compare_digest(self.seal, _seal(self._values()))

    def authorises(self, seed: Any) -> bool:
        """Whether this token opens ``seed``: a reserved root whose index k is in ``roots_k``."""
        k = reserved_root_index(seed)
        return k is not None and k in tuple(int(x) for x in self.roots_k) and self.valid()

    def public_record(self) -> dict[str, Any]:
        """The evidence fields (no root value; the seal is not exported)."""
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self._values().items()}


def mint_official_stream_token(*, head_commit: str, freeze_record_sha256: str, protocol_sha256: str,
                               spec_digest: str, roots_k: tuple[int, ...], issued_utc: str) -> OfficialStreamToken:
    """Mint a token.  Only ``require_protocol_frozen`` of ``experiments/E1_cost_reliability/official_v2.py`` may call
    this (checked on the caller's frame); any other caller gets :class:`ReservedRootError`."""
    caller = sys._getframe(1)
    fn_name, suffix = OFFICIAL_TOKEN_MINTER
    path = Path(caller.f_code.co_filename).as_posix()
    if caller.f_code.co_name != fn_name or not path.endswith("/" + suffix):
        raise ReservedRootError("only official_v2.require_protocol_frozen() may issue an official stream token")
    ks = tuple(int(k) for k in roots_k)
    if not ks or any(not 0 <= k < N_RESERVED_ROOTS for k in ks) or len(set(ks)) != len(ks):
        raise ValueError(f"roots_k must be distinct indices in 0..{N_RESERVED_ROOTS - 1}")
    vals = {"head_commit": str(head_commit), "freeze_record_sha256": str(freeze_record_sha256),
            "protocol_sha256": str(protocol_sha256), "spec_digest": str(spec_digest), "roots_k": ks,
            "issued_utc": str(issued_utc)}
    return OfficialStreamToken(**vals, seal=_seal(vals))


@dataclass(frozen=True)
class RandomStreams:
    """Factory of named generators under one root seed (e.g. development seeds 1103/2207/3301).

    A reserved formal-evaluation root (:func:`reserved_formal_roots`) is refused at construction -- before any draw --
    unless ``official_token`` is a valid :class:`OfficialStreamToken` that authorises it (official-run plan batch 4;
    streams policy SP-3)."""

    root_seed: int
    official_token: Optional[OfficialStreamToken] = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        if int(self.root_seed) < 0:
            raise ValueError("root_seed must be a non-negative integer")
        if is_reserved_root(self.root_seed):
            tok = self.official_token
            if not isinstance(tok, OfficialStreamToken) or not tok.authorises(self.root_seed):
                raise ReservedRootError(
                    "RandomStreams: the root is a reserved formal-evaluation root (configs/streams_policy.yaml SP-3); "
                    "it is opened only with a token from official_v2.require_protocol_frozen() that names its index "
                    "(value withheld; nothing was drawn)")

    def seed_sequence(self, stream: str, *sub: int) -> np.random.SeedSequence:
        """SeedSequence of ``stream`` / ``sub`` keys."""
        key = (stream_index(stream),) + tuple(int(s) for s in sub)
        if any(k < 0 for k in key):
            raise ValueError("sub keys must be non-negative")
        return np.random.SeedSequence(entropy=int(self.root_seed), spawn_key=key)

    def generator(self, stream: str, *sub: int) -> np.random.Generator:
        """A fresh PCG64 generator for ``stream`` / ``sub`` keys."""
        return np.random.Generator(np.random.PCG64(self.seed_sequence(stream, *sub)))

    def stream_id(self, stream: str, *sub: int) -> str:
        """Printable id recorded with draws and in run records, e.g. ``root=1103/opt/0``."""
        tail = "/".join(str(int(s)) for s in sub)
        return f"root={int(self.root_seed)}/{stream}" + (f"/{tail}" if tail else "")
