"""Compact canonical snapshots for full downstream replay; never raw table IO in algorithms."""
from __future__ import annotations

from collections import OrderedDict
from hashlib import sha256
from pathlib import Path
import json
import pandas as pd

from data.artifacts import StageArtifactStore


def write_recompute_frames(root: Path, frames: dict[str, pd.DataFrame | None]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        if frame is None or not len(frame.columns):
            (root / f"{name}.parquet").unlink(missing_ok=True)
            continue
        if frame is not None and len(frame.columns):
            StageArtifactStore.write_frame(root / f"{name}.parquet", frame)


def seal_recompute_bundle(root: Path, parameters: dict, cutoff: str) -> None:
    hashes = {p.name: sha256(p.read_bytes()).hexdigest() for p in sorted(root.glob("*.parquet"))}
    StageArtifactStore.write_json(root / "manifest.json", {
        "version": 1, "cutoff": cutoff, "parameters": parameters, "files": hashes,
        "quantity_scope": "pure W_cover demand; safety/display/order kept separate",
        "history_storage": "store week1 sufficient statistics plus full tier lifecycle; no daily grid",
    })


class ReplayBundleStore:
    """Resolve keys with an LRU bounded by decoded bytes, not raw table size."""
    def __init__(self, replay_root: Path, expected_parameters: dict | None = None,
                 max_cache_mb: float = 256):
        self.index: dict[str, tuple[Path, str]] = {}
        self.cache: OrderedDict = OrderedDict()
        self.expected_parameters = expected_parameters
        self.max_cache_bytes = max_cache_mb * 1024**2
        self.cache_bytes = 0
        self.sizes = {}
        group_root = replay_root / "checkpoint_groups"
        for schedule in sorted(group_root.glob("*/target_schedule.csv")):
            rows = StageArtifactStore.read_frame(schedule)
            for row in rows.itertuples(index=False):
                key = str(getattr(row, "prediction_checkpoint_key", row.sku_code))
                if key in self.index:
                    raise ValueError(f"duplicate recompute checkpoint: {key}")
                self.index[key] = (schedule.parent / "round6" / "_recompute_inputs", str(row.sku_code))
        if not self.index:
            raise ValueError(f"no checkpoint schedules under {replay_root}")

    def __call__(self, key: str) -> dict:
        if key not in self.index:
            raise ValueError(f"recompute bundle missing checkpoint {key}")
        root, sku = self.index[key]
        if root not in self.cache:
            self.cache[root] = load_recompute_bundle(root)
            size = sum(int(f.memory_usage(index=True, deep=True).sum()) for f in self.cache[root]["frames"].values())
            self.sizes[root] = size
            self.cache_bytes += size
            while len(self.cache) > 1 and self.cache_bytes > self.max_cache_bytes:
                old_root, _ = self.cache.popitem(last=False)
                self.cache_bytes -= self.sizes.pop(old_root)
        self.cache.move_to_end(root)
        if self.expected_parameters is not None and self.cache[root]["parameters"] != self.expected_parameters:
            raise ValueError(f"canonical rule/config version differs from frozen pipeline: {root}")
        return {**self.cache[root], "target_sku": sku, "bundle_identity": str(root)}


def load_recompute_bundle(root: Path) -> dict:
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Step5 recompute snapshot missing: {root}; replay features to a NEW directory with updated code")
    manifest = StageArtifactStore.read_json(manifest_path)
    required = {f"{name}.parquet" for name in ("features", "comparable_summary", "comparables", "stores",
                "restored_store_weekly", "restored_tier_weekly")}
    if not required.issubset(manifest["files"]):
        raise ValueError(f"incomplete Step5 canonical bundle: {root}; generate placement and factors in sequence")
    frames = {}
    for name, checksum in manifest["files"].items():
        path = root / name
        if sha256(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"recompute snapshot hash mismatch: {path}")
        frames[Path(name).stem] = StageArtifactStore.read_frame(path)
    return {"frames": frames, "parameters": manifest["parameters"],
            "cutoff": manifest["cutoff"], "bundle_identity": str(root)}
