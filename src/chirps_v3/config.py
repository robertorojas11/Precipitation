"""Experiment contract and chronological folds for CHIRPS v3."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json

from src.chirps_v3 import VERSION


@dataclass(frozen=True)
class Fold:
    name: str
    train_start: int
    train_end: int
    validation_start: int
    validation_end: int

    def contains(self, year: int, split: str) -> bool:
        bounds = ((self.train_start, self.train_end) if split == "train" else
                  (self.validation_start, self.validation_end))
        return bounds[0] <= year <= bounds[1]


FOLDS = (
    Fold("fold_2016_2017", 2004, 2015, 2016, 2017),
    Fold("fold_2018_2019", 2004, 2017, 2018, 2019),
    Fold("fold_2020_2021", 2004, 2019, 2020, 2021),
    Fold("fold_2022_2023", 2004, 2021, 2022, 2023),
    Fold("fold_2024_2025", 2004, 2023, 2024, 2025),
)

ROOT = Path("outputs") / VERSION / "chirps"
PREPARED_ROOT = Path("data") / "v2_clean" / "chirps"


def get_fold(name: str) -> Fold:
    try:
        return next(fold for fold in FOLDS if fold.name == name)
    except StopIteration as error:
        raise ValueError(f"Unknown fold {name!r}; choose {[f.name for f in FOLDS]}") from error


def write_contract(path: Path = ROOT / "experiment_contract.json") -> Path:
    payload = {
        "version": VERSION,
        "target": "chirps",
        "primary_metric": "pooled_masked_physical_mm_r2",
        "target_r2": 0.8,
        "data_policy": "CHIRPS is used only as the supervised target",
        "context_days": [1, 3, 5, 7],
        "folds": [asdict(fold) for fold in FOLDS],
        "final_holdout": {"start": "2026-01-01", "end": "2026-12-31", "minimum_days": 365},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path

