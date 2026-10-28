from __future__ import annotations

import pandas as pd

from data.artifacts import StageArtifactStore


def test_read_frame_projects_and_filters_csv_in_chunks(tmp_path) -> None:
    path = tmp_path / "placement.csv"
    pd.DataFrame(
        [
            {"sku_code": "001", "selected": 1, "value": 10, "unused": "a"},
            {"sku_code": "002", "selected": 0, "value": 20, "unused": "b"},
            {"sku_code": "003", "selected": 1, "value": 30, "unused": "c"},
            {"sku_code": "004", "selected": 1, "value": 40, "unused": "d"},
        ]
    ).to_csv(path, index=False)

    result = StageArtifactStore().read_frame(
        path,
        required=("sku_code", "selected"),
        columns=("sku_code", "value"),
        optional_columns=("unused", "not_present"),
        filters={"sku_code": ["001", "003"], "selected": 1},
        chunksize=2,
    )

    assert result["sku_code"].tolist() == ["001", "003"]
    assert result["value"].tolist() == [10, 30]
    assert result["unused"].tolist() == ["a", "c"]
    assert "not_present" not in result.columns


def test_read_frame_projects_and_filters_parquet(tmp_path) -> None:
    path = tmp_path / "placement.parquet"
    pd.DataFrame(
        [
            {"sku_code": "001", "selected": 1, "value": 10, "unused": "a"},
            {"sku_code": "002", "selected": 0, "value": 20, "unused": "b"},
            {"sku_code": "003", "selected": 1, "value": 30, "unused": "c"},
        ]
    ).to_parquet(path, index=False)

    result = StageArtifactStore().read_frame(
        path,
        required=("sku_code", "selected"),
        columns=("sku_code", "value"),
        optional_columns=("unused", "not_present"),
        filters={"sku_code": ["001", "003"], "selected": 1},
    )

    assert result["sku_code"].tolist() == ["001", "003"]
    assert result["value"].tolist() == [10, 30]
    assert result["unused"].tolist() == ["a", "c"]
    assert "not_present" not in result.columns


def test_iter_frame_parts_can_audit_missing_skus_without_aborting(
    tmp_path,
) -> None:
    parts = tmp_path / "parts"
    parts.mkdir()
    pd.DataFrame(
        {"sku_code": ["A"], "value": [1.0]}
    ).to_parquet(parts / "part-00001.parquet", index=False)

    frames = list(
        StageArtifactStore.iter_frame_parts(
            parts,
            required=("sku_code", "value"),
            sku_codes=["A", "MISSING"],
            require_all_skus=False,
        )
    )

    combined = pd.concat(frames, ignore_index=True)
    assert combined.to_dict(orient="records") == [
        {"sku_code": "A", "value": 1.0}
    ]
