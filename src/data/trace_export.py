"""Read-only, bounded exports from saved standardized artifacts.

Original business-table IO is intentionally absent. Raw product enrichment uses
DecisionDataAdapter separately. Missing files are reported, never invented.
"""
from __future__ import annotations

from contextlib import ExitStack
import gzip
from hashlib import sha256
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from data.artifacts import StageArtifactStore as Store


DEFAULT_TRACE_SKUS: tuple[str, ...] = ()


def subset_entities(frame: pd.DataFrame, *, target: str, related: set[str], name: str,
                    social_keys: set[str] | None = None) -> pd.DataFrame:
    if "target_sku_code" in frame:
        return frame.loc[frame.target_sku_code.astype(str).eq(target)].copy()
    if "sku_code" in frame:
        return frame.loc[frame.sku_code.astype(str).isin(related)].copy()
    if "external_ip_key" in frame:
        return frame.loc[frame.external_ip_key.astype(str).isin(social_keys or set())].copy()
    # Store universe and tier policies must include unselected stores as well:
    # restricting to selected stores would hide the exclusion evidence.
    return frame.copy()


class TraceWriter:
    def __init__(self, root: Path):
        if root.exists():
            raise ValueError(f"trace output already exists; use a new directory: {root}")
        self.root = root
        self.root.mkdir(parents=True)
        self.records: list[dict] = []

    def frame(self, relative: str, frame: pd.DataFrame, *, source: str, scope: str) -> None:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # Empty schema-less artifacts are explicitly represented in inventory;
        # a blank CSV would be unreadable and look like a failed export.
        if len(frame.columns) == 0:
            self.records.append(dict(file=relative, rows=0, status="not_available_no_columns", source=source, scope=scope))
            return
        if relative.endswith(".gz"):
            frame.to_csv(path, index=False, encoding="utf-8-sig", compression="gzip")
        else:
            Store.write_frame(path, frame)
        self.records.append(dict(file=relative, rows=len(frame), status="saved", source=source, scope=scope))

    def json(self, relative: str, payload: dict) -> None:
        Store.write_json(self.root / relative, payload)
        self.records.append(dict(file=relative, rows=1, status="saved", source="saved metadata or frozen reproduction audit", scope="JSON metadata; no refitting"))

    def finish(self, report: dict) -> Path:
        self.json("trace_report.json", report)
        self.frame("file_inventory.csv", pd.DataFrame(self.records), source="export manifest", scope="all exported files except inventory itself")
        archive = self.root.with_suffix(".zip")
        if archive.exists():
            raise ValueError(f"archive already exists: {archive}")
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(self.root.rglob("*")):
                if p.is_file():
                    z.write(p, str(p.relative_to(self.root.parent)),
                            compress_type=zipfile.ZIP_STORED if p.suffix == ".gz" else zipfile.ZIP_DEFLATED)
        return archive


def canonical_frame(root: Path, name: str) -> pd.DataFrame | None:
    """Read a single canonical artifact with its original integrity check."""
    manifest = Store.read_json(root / "manifest.json")
    filename = name + ".parquet"
    if filename not in manifest["files"]:
        return None
    path = root / filename
    if sha256(path.read_bytes()).hexdigest() != manifest["files"][filename]:
        raise ValueError(f"canonical hash mismatch: {path}")
    return Store.read_frame(path)


def export_label_weeks(writer: TraceWriter, *, weekly_path: Path | None, labels: pd.DataFrame,
                       skus: list[str], progress=None) -> dict:
    """Stream the original label source once; preserve store-week rows as gzip.

    No 0-filling, no re-restoration, no calendar reconstruction. Summaries use
    only columns actually present in the saved weekly source.
    """
    if weekly_path is None or not weekly_path.exists():
        return {"status": "not_available", "source": str(weekly_path),
                "reason": "original label store-week artifact is missing; labels alone cannot reconstruct it"}
    covers = labels.set_index("sku_code")["W_cover"].to_dict()
    sums: dict[str, dict] = {}
    counts = {sku: 0 for sku in skus}
    with ExitStack() as stack:
        handles = {}
        for index, part in enumerate(Store.iter_frame_parts(weekly_path, required=("sku_code", "lifecycle_week", "restored_natural_sales"),
                                                            sku_codes=skus, require_all_skus=False), 1):
            if progress:
                progress({"phase": "label_weekly_export", "part": index, "selected_rows": len(part)})
            for sku, group in part.groupby("sku_code", sort=False):
                sku = str(sku)
                frame = group.copy()
                frame["audit_in_W_cover"] = pd.to_numeric(frame.lifecycle_week, errors="coerce").between(1, float(covers[sku]))
                relative = f"label_source/{sku}/store_weekly_original.csv.gz"
                if sku not in handles:
                    path = writer.root / relative
                    path.parent.mkdir(parents=True, exist_ok=True)
                    handles[sku] = stack.enter_context(gzip.open(path, "wt", encoding="utf-8-sig", newline=""))
                frame.to_csv(handles[sku], index=False, header=counts[sku] == 0)
                counts[sku] += len(frame)
                numeric = [c for c in ("restored_natural_sales", "clean_sales_qty", "restored_lost_sales", "restored_exposure_days", "unknown_exposure_days", "source_day_rows") if c in frame]
                for c in numeric:
                    frame[c] = pd.to_numeric(frame[c], errors="raise")
                grouped = frame.groupby(["lifecycle_week", "audit_in_W_cover"], as_index=False)[numeric].sum(min_count=1)
                for row in grouped.to_dict(orient="records"):
                    key = (sku, row["lifecycle_week"])
                    if key not in sums:
                        sums[key] = {"sku_code": sku, "lifecycle_week": row["lifecycle_week"], "audit_in_W_cover": row["audit_in_W_cover"], **{c: 0. for c in numeric}}
                    for c in numeric:
                        sums[key][c] += row[c]
    weekly = pd.DataFrame(sums.values())
    if not weekly.empty:
        writer.frame("label_source/weekly_sums.csv", weekly.sort_values(["sku_code", "lifecycle_week"]), source=str(weekly_path), scope="original label store-week aggregation; outside-window weeks retained")
    reconciliations = []
    for sku in skus:
        relative = f"label_source/{sku}/store_weekly_original.csv.gz"
        writer.records.append(dict(file=relative, rows=counts[sku], status="saved" if counts[sku] else "no_source_rows", source=str(weekly_path), scope="original target store-week rows; no fabricated days"))
        values = weekly.loc[weekly.sku_code.eq(sku) & weekly.audit_in_W_cover.eq(True), "restored_natural_sales"] if not weekly.empty else pd.Series(dtype=float)
        total = values.sum(min_count=1)
        label = float(labels.set_index("sku_code").loc[sku, "D_first_ref"])
        reconciliations.append(dict(sku_code=sku, D_first_ref=label, reconstructed_from_original_weeks=total,
                                    difference=total-label, matches=bool(np.isclose(total,label,rtol=1e-8,atol=1e-6)) if pd.notna(total) else False))
    writer.frame("label_source/label_reconciliation.csv", pd.DataFrame(reconciliations), source=str(weekly_path), scope="labels versus original W_cover sums")
    return {"status": "exported", "source": str(weekly_path), "store_week_rows": counts,
            "all_label_sums_match": all(row["matches"] for row in reconciliations)}
