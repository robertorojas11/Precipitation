"""PyTorch ``Dataset`` over stage 4/5's prepared, normalized hourly tensors
(stage 6 -- training, PASO 5 of docs/decisions/0002-implementacion-v2.md;
full narrative in docs/stages_v2/06-train.md).

One sample is one ``(day, hour)`` pair from a **single region's** prepared
split -- this project's training decision is "single-region start"
(docs/decisions/0002-implementacion-v2.md §1.1, reaffirming decision 0001;
``docs/architecture_v3.md`` B.4 marks the hierarchical region-union
mechanism DEFERRED). Each region has its own canonical grid shape
(``build_hourly_dataset.region_grid``), so pooling multiple regions into one
batch would require padding or a region-homogeneous batch sampler neither
of which is built yet -- training one region at a time sidesteps the issue
entirely and matches the deferred-union decision already on record.

Every tensor returned here matches the channel layout the three candidate
models (``src/models/{generator,discriminator,regression_stage,
deterministic_resnet}.py``) expect directly, with no further reshaping
needed by a training loop: ``inputs`` (lr_channels, H, W), ``invariants``
(2, H, W), ``target`` (1, H, W).

**NaN sanitization is a forward-pass concern only, never a loss/metric
concern.** Stage 4 deliberately leaves invalid pixels as ``NaN`` after the
log transform + normalization (docs/stages_v2/README.md's "Shared
experimental contract": "never silently converted to zero rain"). Feeding
a NaN into a convolution would poison every pixel in its receptive field,
not just the invalid one, so ``__getitem__`` replaces NaN with ``0.0``
(the distribution's own mean in normalized z-score space) **only** in the
tensors handed to the network; the original validity masks are returned
unchanged and must still be used to exclude those exact pixels from every
loss and metric -- the substitution never reaches a loss function.

**Per-day caching, and why it needs ``DayGroupedSampler``.** This module
used to re-open and fully decompress a day's NPZ on *every* ``__getitem__``
call -- 24x redundant work, since all 24 hours of a day live in the same
file -- flagged as a known limitation in `docs/stages_v2/06-train.md`
("revisit once real training throughput is actually measured"). It has
now been measured, on real data: ~71ms/day to decompress from the network-
mounted store, which at ~3,650 accepted training days x 24 redundant reads
each is **~104 minutes of pure I/O per epoch**, in a single process, before
a single gradient step -- the actual cause of training appearing to hang
with an empty log for over 40 minutes. `__getitem__` now keeps a one-slot
cache of the most recently loaded day, cutting that to ~104 minutes *once*
(amortized) instead of every epoch -- **but only if consecutive
``__getitem__`` calls tend to share a day**. A plain
``DataLoader(..., shuffle=True)`` draws a uniform random permutation over
the full ``days * 24`` index range, so consecutive calls almost never
share a day and the cache's hit rate would stay near zero. ``
DayGroupedSampler`` below shuffles *day order* (and, independently, *hour
order within each day*) instead of shuffling every index globally -- still
a different, fully randomized visiting order each epoch, just one that
keeps each day's 24 hours contiguous, which is exactly what the cache
needs. The default (non-shuffled) sequential order `DataLoader(shuffle=
False)` already uses for validation/test already has this property for
free, by construction (index = day_index * 24 + hour), so only the
*training* loader needs the custom sampler.

**Full in-RAM preload -- the one-slot cache's limit, and why this goes
further.** The one-slot cache only amortizes I/O *within* an epoch (each
day paid for once, not 24x); every new epoch still re-opens every day's
NPZ from scratch over the network mount (``/mnt/data-r2``, confirmed CIFS/
SMB, not local disk). Measured on real data, one region's full training
split is ~11-12GB across ~3,650 files -- trivially small next to this
machine's 188GB RAM -- so ``preload=True`` loads every day once in
``__init__`` (via a bounded ``ThreadPoolExecutor``: the bottleneck is
network latency, not CPU, and ``np.load`` releases the GIL during the
socket read/decompress, so threads overlap real network waits) and keeps
every array in memory for the rest of the process's life. This removes
per-epoch network I/O entirely after the one upfront cost, independent of
sampler order -- ``DayGroupedSampler`` is still used for shuffling, just no
longer load-bearing for cache locality once preloaded. Default is
``False`` so existing callers (tests with tiny synthetic fixtures, or
memory-constrained runs) are unaffected; real training scripts opt in.
Only the resulting plain ``dict`` is kept as state, never the executor
itself, so a preloaded dataset stays picklable for ``DataLoader`` worker
processes.

**Measured on real data: decompression inflates size ~8x, not ~1x.**
``prepare_hourly_dataset.py`` writes each day with ``np.savez_compressed``,
and precipitation grids compress well (mostly zeros/repeated values) -- one
real day measured at ~1.3MB on disk unpacks to ~11MB in memory. A region's
full ~11-12GB *on-disk* training split is therefore ~90GB *decompressed*,
not ~11GB -- this is per **process**, not shared, so running several
models for the *same* region concurrently with ``preload=True`` each pay
this cost independently (3 models x ~90GB blew well past this machine's
188GB RAM and triggered silent kills with no dmesg/journalctl trace, the
hard way this was found). Preload one process per region at a time, or
leave ``preload=False`` (the per-day cache still helps) when running
several models/regions concurrently on shared hardware.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

from src.data_preprocessing.prepare_hourly_dataset import _prepared_root
from src.utils.provenance import load_manifest

HOURS_PER_DAY = 24


def _nan_to_zero(array: np.ndarray) -> np.ndarray:
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)


class PreparedHourlyDataset(Dataset):
    """``len()`` is ``accepted_days * 24``. Keeps a one-slot cache of the
    most recently loaded day's raw arrays (see module docstring) -- correct
    under any access pattern (a cache miss just reloads), fast only when
    consecutive indices share a day, which `DayGroupedSampler` guarantees
    for training and sequential (non-shuffled) iteration already guarantees
    for free. Not safe to share across ``DataLoader`` worker processes --
    each worker forks its own copy of the dataset (and therefore its own
    cache), so this is only effective with ``num_workers=0`` (the default
    throughout this project's training scripts); with workers > 0 each
    worker still benefits individually, just without the shared RAM. Pass
    ``preload=True`` to load every day up front instead (see module
    docstring) -- then no per-worker cache behavior matters at all, because
    every day already lives in memory.
    """

    def __init__(
        self,
        region: str,
        split: str,
        *,
        prepared_root: Path | None = None,
        preload: bool = False,
        preload_workers: int = 16,
    ):
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}")
        self.region = region
        self.split = split
        self.prepared_root = Path(prepared_root) if prepared_root is not None else _prepared_root()
        manifest_path = self.prepared_root / "metadata" / f"manifest_prepared_{region}.json"
        manifest = load_manifest(manifest_path)
        if manifest is None:
            raise FileNotFoundError(f"No prepared manifest at {manifest_path}")
        self.records = [r for r in manifest.get("records", []) if r["split"] == split]
        if not self.records:
            raise ValueError(f"No prepared records for region={region!r} split={split!r} in {manifest_path}")
        self._cache_day_index: int | None = None
        self._cache_payload: dict[str, np.ndarray] | None = None
        self._preloaded: dict[int, dict[str, np.ndarray]] | None = None
        if preload:
            self._preloaded = self._preload_all(preload_workers)

    @staticmethod
    def _read_day(record: dict) -> dict[str, np.ndarray]:
        with np.load(record["path"]) as data:
            return {
                "inputs_norm": data["inputs_norm"],
                "target_norm": data["target_norm"],
                "invariants_norm": data["invariants_norm"],
                "input_valid_mask": data["input_valid_mask"],
                "target_valid_mask": data["target_valid_mask"],
                "basin_mask": data["basin_mask"],
            }

    def _preload_all(self, max_workers: int) -> dict[int, dict[str, np.ndarray]]:
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as pool:
            payloads = list(pool.map(self._read_day, self.records))
        return dict(enumerate(payloads))

    def __len__(self) -> int:
        return len(self.records) * HOURS_PER_DAY

    def _load_day(self, day_index: int) -> dict[str, np.ndarray]:
        if self._preloaded is not None:
            return self._preloaded[day_index]
        if day_index == self._cache_day_index and self._cache_payload is not None:
            return self._cache_payload
        payload = self._read_day(self.records[day_index])
        self._cache_day_index = day_index
        self._cache_payload = payload
        return payload

    def __getitem__(self, index: int) -> dict:
        day_index, hour = divmod(index, HOURS_PER_DAY)
        data = self._load_day(day_index)
        inputs = np.ascontiguousarray(data["inputs_norm"][hour])          # (H, W, 2)
        target = np.ascontiguousarray(data["target_norm"][hour, ..., 0])  # (H, W)
        invariants = np.ascontiguousarray(data["invariants_norm"])        # (H, W, 2)
        input_valid = np.ascontiguousarray(data["input_valid_mask"][hour])
        target_valid = np.ascontiguousarray(data["target_valid_mask"][hour])
        basin_mask = np.ascontiguousarray(data["basin_mask"])

        return {
            "inputs": torch.from_numpy(_nan_to_zero(inputs)).permute(2, 0, 1).float(),
            "invariants": torch.from_numpy(_nan_to_zero(invariants)).permute(2, 0, 1).float(),
            "target": torch.from_numpy(_nan_to_zero(target)).unsqueeze(0).float(),
            "input_valid_mask": torch.from_numpy(input_valid).bool(),
            "target_valid_mask": torch.from_numpy(target_valid).bool(),
            "basin_mask": torch.from_numpy(basin_mask).bool(),
            "region": self.region,
            "date": self.records[day_index]["date"],
        }


class DayGroupedSampler(Sampler[int]):
    """Randomizes day order (and, independently, hour order within each
    day) instead of shuffling every ``(day, hour)`` index globally --
    see module docstring for why this is what makes
    ``PreparedHourlyDataset``'s per-day cache effective during shuffled
    training. ``set_epoch`` must be called once per epoch by the training
    loop (the way ``DistributedSampler`` does) so the random order actually
    changes across epochs; without it every epoch would visit days in the
    exact same order.
    """

    def __init__(self, dataset: PreparedHourlyDataset, *, shuffle: bool = True, seed: int = 0):
        self.num_days = len(dataset.records)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.num_days * HOURS_PER_DAY

    def __iter__(self) -> Iterator[int]:
        rng = np.random.default_rng(self.seed + self.epoch)
        day_order = rng.permutation(self.num_days) if self.shuffle else np.arange(self.num_days)
        for day in day_order:
            hour_order = rng.permutation(HOURS_PER_DAY) if self.shuffle else np.arange(HOURS_PER_DAY)
            for hour in hour_order:
                yield int(day) * HOURS_PER_DAY + int(hour)
