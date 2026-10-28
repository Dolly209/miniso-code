"""Content-addressed checkpoints for standardized M3/M4 target results."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import pandas as pd

from data.artifacts import StageArtifactStore


def placement_fingerprint(frames: Sequence[pd.DataFrame | None], metadata: dict[str, Any]) -> str:
    digest = hashlib.sha256(json.dumps(metadata, sort_keys=True, default=str).encode())
    for frame in frames:
        if frame is None:
            digest.update(b"none")
            continue
        digest.update(str((list(frame.columns), list(map(str, frame.dtypes)), len(frame))).encode())
        for start in range(0, len(frame), 100_000):
            digest.update(pd.util.hash_pandas_object(frame.iloc[start:start + 100_000], index=True).values.tobytes())
    return digest.hexdigest()


class PlacementCheckpointStore:
    names = ("store_plan", "summary", "demand_curve", "diagnostics", "compliance")

    def __init__(self, root: Path, fingerprint: str, resume: bool = True):
        self.root = root / fingerprint
        self.resume = resume

    def _directory(self, sku: str) -> Path:
        return self.root / hashlib.sha256(sku.encode()).hexdigest()

    def load(self, sku: str):
        from placement.engine import PlacementResult
        directory = self._directory(sku)
        marker = directory / "complete.json"
        if not self.resume or not marker.exists():
            return None
        manifest = StageArtifactStore.read_json(marker)
        if manifest["sku_code"] != sku:
            raise ValueError("placement checkpoint SKU mismatch")
        frames = []
        for name in self.names:
            path = directory / (name + ".parquet")
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["hashes"][name]:
                raise ValueError(f"Incomplete or changed placement checkpoint: {path}; rerun with --no-resume")
            frames.append(StageArtifactStore.read_frame(path))
        return PlacementResult(*frames, tuple(manifest["warnings"]))

    def save(self, sku: str, result) -> None:
        directory = self._directory(sku)
        hashes = {}
        for name in self.names:
            path = directory / (name + ".parquet")
            StageArtifactStore.write_frame(path, getattr(result, name))
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        StageArtifactStore.write_json(directory / "complete.json", {
            "sku_code": sku, "hashes": hashes, "warnings": list(result.warnings),
        })
