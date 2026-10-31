"""Holographic Reduced Representations (HRR) with phase encoding. Each concept is a vector of angles in [0, 2π):
bind = circular convolution (phase addition), unbind = circular correlation (phase subtraction), bundle =
superposition (circular mean). Phase encoding avoids the magnitude collapse of complex-number HRRs and maps cleanly
to cosine similarity; atoms derive deterministically from SHA-256 so representations are identical across processes,
machines, and Python versions. References: Plate (1995) HRRs; Gayler (2004) Vector Symbolic Architectures."""

import hashlib
import importlib.util
import logging
import math
import struct
import sys

np = None  # numpy is loaded on first use (see _numpy): this module is imported from inside the
# plugin loader's lock while a managed worker starts its reader thread, and a top-level numpy
# import there deadlocked Kanban workers inside numpy._core._multiarray_umath's create_module.
_TRIED = False  # _numpy() has tried already; np alone cannot say (None is both "not yet" and "missing")
_ERROR = ""  # why a numpy that is installed did not load; "" when it loaded, is missing, or was not tried yet
logger = logging.getLogger(__name__)


def _numpy():
    """numpy, imported on first use; None when it is not installed or does not load."""
    global np, _TRIED, _ERROR
    if not _TRIED:
        try:
            import numpy as np
        except Exception as exc:  # noqa: BLE001 - installed but broken (a wheel without its DLL: OSError too)
            np = None
            if numpy_findable():  # missing is a setup and stays silent, as before; broken is said once
                _ERROR = f"{type(exc).__name__}: {exc}"[:200]
                logger.warning("holographic: numpy is installed but did not load (%s); structural recall and search "
                               "by meaning stay off in this process, recall goes on by words", _ERROR)
        # Publish only a completed attempt. Concurrent callers must reach Python's
        # import lock rather than mistake an in-progress import for missing numpy.
        _TRIED = True
    return np


def has_numpy() -> bool:
    """Whether numpy can be imported (tried once, then cached): the gates that used to read a
    module-level flag ask this instead, so "not loaded yet" is never read as "not installed"."""
    return _numpy() is not None


def numpy_installed() -> bool:
    """Whether numpy is there, WITHOUT importing it: the answer of the import once one was tried, else whether
    Python can find the package. For checks made while the provider starts (the system prompt line, the store's
    vector repair), where an import would only load numpy early: on Windows a Kanban worker hangs on that load
    (NousResearch/hermes-agent#145156). A package found here may still fail to import; the first real use
    (has_numpy) then says so."""
    if _TRIED:
        return np is not None
    return numpy_findable()


def numpy_findable() -> bool:
    """Whether Python can find numpy, without importing it and whatever an earlier import attempt said."""
    loaded = sys.modules.get("numpy", ...)
    if loaded is not ...:  # imported elsewhere, or blocked with sys.modules["numpy"] = None
        return loaded is not None
    try:
        return importlib.util.find_spec("numpy") is not None
    except (ImportError, ValueError):
        return False


def numpy_error() -> str:
    """Why an installed numpy did not load, once an import was tried; "" otherwise."""
    return _ERROR

_TWO_PI = 2.0 * math.pi
_FLOAT32_BLOB_PREFIX = b"HRR1"
_F32, _F64 = 4, 8  # itemsizes of np.float32 / np.float64
ROLE_CONTENT, ROLE_ENTITY = "__hrr_role_content__", "__hrr_role_entity__"  # role atoms used by encode_fact


def _require_numpy() -> None:
    if _numpy() is None:
        raise RuntimeError("numpy is required for holographic operations")


def encode_atom(word: str, dim: int = 1024) -> "np.ndarray":
    """Deterministic phase vector: SHA-256 counter blocks of f"{word}:{i}" -> uint16 -> [0, 2π).
    hashlib rather than numpy RNG so atoms are reproducible across platforms."""
    _require_numpy()
    uint16_values = [v for i in range(math.ceil(dim / 16))  # 32-byte digest = 16 uint16 values
                     for v in struct.unpack("<16H", hashlib.sha256(f"{word}:{i}".encode()).digest())]
    return np.array(uint16_values[:dim], dtype=np.float64) * (_TWO_PI / 65536.0)


def bind(a: "np.ndarray", b: "np.ndarray") -> "np.ndarray":
    """Circular convolution = phase addition; result is quasi-orthogonal to both inputs."""
    _require_numpy()
    return (a + b) % _TWO_PI


def unbind(memory: "np.ndarray", key: "np.ndarray") -> "np.ndarray":
    """Circular correlation = phase subtraction; unbind(bind(a, b), a) ≈ b."""
    _require_numpy()
    return (memory - key) % _TWO_PI


def bundle(*vectors: "np.ndarray") -> "np.ndarray":
    """Superposition via circular mean; holds O(sqrt(dim)) items before similarity degrades."""
    _require_numpy()
    return np.angle(np.sum([np.exp(1j * v) for v in vectors], axis=0)) % _TWO_PI


def similarity(a: "np.ndarray", b: "np.ndarray") -> float:
    """Phase cosine similarity in [-1, 1]; ~0 for unrelated vectors."""
    _require_numpy()
    return float(np.mean(np.cos(a - b)))


def encode_text(text: str, dim: int = 1024) -> "np.ndarray":
    """Bag-of-words bundle of token atoms; empty text -> encode_atom("__hrr_empty__")."""
    _require_numpy()
    tokens = [t for t in (tok.strip(".,!?;:\"'()[]{}") for tok in text.lower().split()) if t]
    return bundle(*[encode_atom(token, dim) for token in tokens]) if tokens else encode_atom("__hrr_empty__", dim)


def encode_fact(content: str, entities: list[str], dim: int = 1024) -> "np.ndarray":
    """bundle(bind(text, ROLE_CONTENT), bind(entity_i, ROLE_ENTITY)...), so
    unbind(fact, bind(entity, ROLE_ENTITY)) ≈ content_vector."""
    _require_numpy()
    role_content, role_entity = encode_atom(ROLE_CONTENT, dim), encode_atom(ROLE_ENTITY, dim)
    return bundle(bind(encode_text(content, dim), role_content),
                  *[bind(encode_atom(entity.lower(), dim), role_entity) for entity in entities])


def phases_to_bytes(phases: "np.ndarray", dim: int | None = None) -> bytes:
    """Serialize as prefixed float32 (half the size of legacy float64 blobs). At dim=1 the prefixed float32 blob
    and a raw float64 blob are both 8 bytes, so write legacy float64 there to keep ``bytes_to_phases`` unambiguous."""
    _require_numpy()
    dim = int(phases.shape[0]) if dim is None else dim
    if len(_FLOAT32_BLOB_PREFIX) + dim * _F32 == dim * _F64:
        return np.asarray(phases, dtype=np.float64).tobytes()
    return _FLOAT32_BLOB_PREFIX + np.asarray(phases, dtype=np.float32).tobytes()


def bytes_to_phases(data: bytes, dim: int | None = None) -> "np.ndarray":
    """Deserialize prefixed float32 or legacy raw float64 blobs (always returns float64). With ``dim`` given, a
    prefixed blob whose size equals the float64 size (dim=1) is read as legacy float64: ``phases_to_bytes`` never
    writes a prefixed blob at that size."""
    _require_numpy()
    plen = len(_FLOAT32_BLOB_PREFIX)
    prefixed = data.startswith(_FLOAT32_BLOB_PREFIX)
    f32 = lambda payload: np.frombuffer(payload, dtype=np.float32).astype(np.float64)  # noqa: E731
    f64 = lambda payload: np.frombuffer(payload, dtype=np.float64).copy()  # noqa: E731
    if dim is None:
        payload, size, what = (data[plen:], _F32, "float32 vector blob has invalid payload") if prefixed else (data, _F64, "legacy vector blob has invalid")
        if len(payload) % size != 0:
            raise ValueError(f"HRR {what} byte length: {len(payload)}")
        return f32(payload) if prefixed else f64(payload)
    float32_blob_bytes, float64_bytes = plen + dim * _F32, dim * _F64
    collides = float32_blob_bytes == float64_bytes
    if not collides and prefixed and len(data) == float32_blob_bytes:
        return f32(data[plen:])
    if len(data) == float64_bytes:
        return f64(data)
    if not prefixed:
        raise ValueError(f"HRR legacy vector blob has {len(data)} bytes; expected {float64_bytes} (float64) for dim={dim}")
    expected = f"{float64_bytes} (legacy float64)" if collides else f"{float32_blob_bytes} (prefixed float32) or {float64_bytes} (legacy float64)"
    raise ValueError(f"HRR vector blob has {len(data)} bytes ({len(data) - plen} payload bytes after the float32 prefix); "
                     f"expected {expected} for dim={dim}")
