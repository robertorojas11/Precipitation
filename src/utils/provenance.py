"""Shared provenance helpers: git revision capture and hashed manifests.

Generalizes the manifest pattern already used by the archived deterministic
pipeline (``legacy/deterministic_unet/train.py``'s ``_source_revision`` and
``src/data_preprocessing/quality.py``'s ``file_sha256``/``manifest_sha256``) so
every new acquisition artifact (downloaded ERA5/IMERG/Oya files) carries the
same traceability metadata. See
``docs/decisions/0002-implementacion-v2.md`` PASO 3/4.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any

from src.data_preprocessing.quality import file_sha256, manifest_sha256

__all__ = ["file_sha256", "source_revision", "write_manifest", "load_manifest"]


def source_revision() -> dict[str, Any]:
    """Return the current git commit and dirty state, or ``None`` values."""
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except Exception:
        return {"commit": None, "dirty": None}


def write_manifest(
    path: str | Path,
    *,
    kind: str,
    records: list[dict],
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write a manifest with a stable content hash and provenance block.

    ``records`` is the authoritative list (one entry per artifact, each with
    its own ``sha256``); ``manifest_sha256`` hashes exactly that list so a
    manifest is comparable across machines. The write is atomic (temp file +
    rename), matching the pipeline's existing atomic-write convention.
    """
    manifest = {
        "kind": kind,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_revision(),
        "record_count": len(records),
        "manifest_sha256": manifest_sha256(records),
        "records": records,
    }
    if extra:
        manifest.update(extra)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, default=str) + "\n", encoding="utf-8")
    tmp.replace(path)
    return manifest


def load_manifest(path: str | Path) -> dict[str, Any] | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
