"""Health checks for local and network-backed pipeline storage."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import secrets
import time

from src.utils.config import Config


@dataclass(frozen=True)
class StorageProbe:
    path: str
    resolved_path: str
    mounted: bool
    mount_point: str
    filesystem: str
    writable: bool
    free_gib: float
    latency_ms: float
    integrity_verified: bool
    error: str | None = None

    @property
    def healthy(self) -> bool:
        return (
            self.mounted
            and self.writable
            and self.integrity_verified
            and self.error is None
        )


def probe_directory(path: Path | str, *, minimum_free_gib: float = 5.0) -> StorageProbe:
    """Verify a directory using a durable write, rename, read, and cleanup cycle."""
    directory = Path(path)
    started = time.perf_counter()
    temporary = directory / f".precipitation-storage-probe-{os.getpid()}.tmp"
    renamed = temporary.with_suffix(".verified")
    mounted = False
    mount_point = ""
    filesystem = ""
    writable = False
    integrity_verified = False
    free_gib = 0.0
    error = None
    try:
        candidate_path = directory.absolute()
        mount_candidates = []
        with Path("/proc/self/mountinfo").open() as stream:
            for line in stream:
                left, right = line.rstrip().split(" - ", maxsplit=1)
                fields = left.split()
                mounted_at = Path(fields[4].replace("\\040", " "))
                if candidate_path == mounted_at or mounted_at in candidate_path.parents:
                    mount_candidates.append((len(mounted_at.parts), mounted_at, right.split()[0]))
        if not mount_candidates:
            raise OSError(f"Cannot identify filesystem mount for {candidate_path}")
        _, detected_mount, filesystem = max(mount_candidates)
        mount_point = str(detected_mount)
        mounted = True
        if str(candidate_path).startswith("/mnt/") and detected_mount == Path("/"):
            raise OSError(
                f"{candidate_path} is not backed by a mounted filesystem under /mnt"
            )

        directory.mkdir(parents=True, exist_ok=True)
        resolved = directory.resolve(strict=True)
        statistics = os.statvfs(resolved)
        free_gib = statistics.f_bavail * statistics.f_frsize / 1024**3
        if free_gib < minimum_free_gib:
            raise OSError(
                f"Only {free_gib:.2f} GiB free; minimum is {minimum_free_gib:.2f} GiB"
            )

        payload = secrets.token_bytes(1024 * 1024)
        expected_hash = sha256(payload).hexdigest()
        with temporary.open("wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        writable = True
        temporary.replace(renamed)
        with renamed.open("rb") as stream:
            actual_hash = sha256(stream.read()).hexdigest()
        integrity_verified = actual_hash == expected_hash
        if not integrity_verified:
            raise OSError("Storage probe checksum mismatch")
        resolved_text = str(resolved)
    except Exception as exception:
        resolved_text = str(directory.absolute())
        error = f"{type(exception).__name__}: {exception}"
    finally:
        for candidate in (temporary, renamed):
            try:
                candidate.unlink(missing_ok=True)
            except OSError as cleanup_error:
                error = error or f"Probe cleanup failed: {cleanup_error}"
    latency_ms = (time.perf_counter() - started) * 1000.0
    return StorageProbe(
        path=str(directory),
        resolved_path=resolved_text,
        mounted=mounted,
        mount_point=mount_point,
        filesystem=filesystem,
        writable=writable,
        free_gib=round(free_gib, 3),
        latency_ms=round(latency_ms, 3),
        integrity_verified=integrity_verified,
        error=error,
    )


def probe_pipeline_storage(minimum_free_gib: float = 5.0) -> dict:
    """Probe every distinct externally configured pipeline storage directory."""
    paths = {
        Path(Config.LOCAL_DATA_DIR),
        Path(Config.RAW_DATA_DIR),
        Path(Config.PROCESSED_DATA_DIR),
    }
    probes = [
        probe_directory(path, minimum_free_gib=minimum_free_gib)
        for path in sorted(paths, key=str)
    ]
    return {
        "healthy": all(probe.healthy for probe in probes),
        "minimum_free_gib": minimum_free_gib,
        "probes": [asdict(probe) | {"healthy": probe.healthy} for probe in probes],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimum-free-gib", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = probe_pipeline_storage(args.minimum_free_gib)
    payload = json.dumps(result, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n")
    raise SystemExit(0 if result["healthy"] else 1)


if __name__ == "__main__":
    main()
