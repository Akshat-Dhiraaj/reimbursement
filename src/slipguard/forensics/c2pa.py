"""C2PA / Content Credentials provenance — the cryptographic-provenance sibling of the
EXIF (:mod:`slipguard.forensics.image`) and PDF (:mod:`slipguard.forensics.pdf`) inspectors.

A C2PA manifest is a *cryptographically signed* provenance record some AI tools and cameras
embed in an image:

* **AI generators** (Adobe Firefly, DALL-E, Sora, Google Imagen) stamp a ``c2pa.actions``
  assertion whose ``digitalSourceType`` is ``trainedAlgorithmicMedia`` (or a composite) —
  signed evidence the image was AI-generated/edited. This is the one IMAGE signal that
  yields a *trustworthy positive* rather than a heuristic.
* **Some cameras** (Pixel 10 signs every photo; Galaxy S25 signs AI edits) stamp a
  ``digitalCapture`` source — evidence of a genuine capture (weak exoneration).

**Honest limits the detector reflects:** HIGH-PRECISION / NEAR-ZERO-RECALL. Most real
receipt photos carry no manifest (iPhone / most Android don't sign by default), and any
screenshot or re-save strips it — so **absence is not evidence** (we abstain). It is
strippable, so a clean camera read only weakly exonerates.

``c2pa-python`` (MIT/Apache, the optional ``[c2pa]`` extra) is imported lazily;
:func:`c2pa_available` lets the detector check without importing and abstain cleanly when
it is absent — the same discipline as the Pillow and pikepdf layers.
"""

from __future__ import annotations

import importlib.util
import json
import os
from dataclasses import dataclass
from typing import Optional

# IPTC digitalSourceType codes (matched as lowercase substrings of the URI) meaning the
# media was produced/edited by a generative model — the AI-fraud signal.
_AI_SOURCE_TOKENS = (
    "trainedalgorithmicmedia",
    "compositewithtrainedalgorithmicmedia",
    "algorithmicmedia",
)
# ... meaning a genuine camera/sensor capture — weak exoneration.
_CAMERA_SOURCE_TOKENS = ("digitalcapture", "computationalcapture")

_MIME = {
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png",
    ".webp": "image/webp", ".tif": "image/tiff", ".tiff": "image/tiff",
    ".heic": "image/heic", ".heif": "image/heif", ".avif": "image/avif",
    ".gif": "image/gif", ".pdf": "application/pdf",
}


@dataclass
class C2paProvenance:
    has_manifest: bool
    source_type: str = "unknown"           # 'ai' | 'camera' | 'unknown'
    source_uris: tuple[str, ...] = ()
    generator: Optional[str] = None        # claim_generator — the signing tool
    validation_state: Optional[str] = None


def c2pa_available() -> bool:
    """Whether c2pa-python can be imported, checked without importing it."""
    return importlib.util.find_spec("c2pa") is not None


def _mime_for(path: str) -> str:
    return _MIME.get(os.path.splitext(path)[1].lower(), "application/octet-stream")


def _collect_values(obj, key: str) -> list[str]:
    """Every string value stored under ``key`` anywhere in a nested manifest. C2PA nests
    ``digitalSourceType`` differently across versions/assertions, so we search rather than
    assume a fixed path (robust to schema drift)."""
    out: list[str] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == key and isinstance(v, str):
                out.append(v)
            else:
                out.extend(_collect_values(v, key))
    elif isinstance(obj, list):
        for v in obj:
            out.extend(_collect_values(v, key))
    return out


def classify_source_types(manifest: dict) -> tuple[str, list[str]]:
    """Classify a manifest's ``digitalSourceType`` assertions as ``'ai'``, ``'camera'`` or
    ``'unknown'`` (with the raw URIs). ``'ai'`` wins over ``'camera'`` when both appear — a
    trained-algorithmic edit over a captured base is still an AI edit."""
    uris = _collect_values(manifest, "digitalSourceType")
    low = [u.lower() for u in uris]
    if any(tok in u for u in low for tok in _AI_SOURCE_TOKENS):
        return "ai", uris
    if any(tok in u for u in low for tok in _CAMERA_SOURCE_TOKENS):
        return "camera", uris
    return "unknown", uris


def _active_manifest(store: dict) -> dict:
    """The active manifest from a ManifestStore JSON, or the store itself if the shape is
    flat/unknown — :func:`classify_source_types` searches recursively either way."""
    manifests = store.get("manifests")
    if isinstance(manifests, dict) and manifests:
        active = store.get("active_manifest")
        if active in manifests:
            return manifests[active]
        if len(manifests) == 1:
            return next(iter(manifests.values()))
    return store


def _generator(manifest: dict) -> Optional[str]:
    info = manifest.get("claim_generator_info")
    if isinstance(info, list) and info and isinstance(info[0], dict):
        name = info[0].get("name")
        if isinstance(name, str):
            return name
    cg = manifest.get("claim_generator")
    return cg if isinstance(cg, str) else None


def inspect_c2pa(path: str) -> C2paProvenance:
    """Read a file's C2PA manifest and classify its ``digitalSourceType``. Returns
    ``has_manifest=False`` for any file with no / unreadable manifest (the common case) so
    the caller treats absence as a non-signal. Requires c2pa-python (gate with
    :func:`c2pa_available`)."""
    import c2pa

    mime = _mime_for(path)
    try:
        with open(path, "rb") as fh, c2pa.Reader(mime, fh) as reader:
            store = json.loads(reader.json())
            try:
                state = str(reader.get_validation_state())
            except Exception:
                state = None
    except Exception:
        # no manifest, unreadable, or unsupported type -> nothing to judge
        return C2paProvenance(has_manifest=False)

    manifest = _active_manifest(store)
    source_type, uris = classify_source_types(manifest)
    return C2paProvenance(
        has_manifest=True,
        source_type=source_type,
        source_uris=tuple(uris),
        generator=_generator(manifest),
        validation_state=state,
    )
