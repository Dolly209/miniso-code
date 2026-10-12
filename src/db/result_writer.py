from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Mapping

import pandas as pd

from db.connection import Database


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, sort_keys=True)


def _sql_value(value: Any) -> Any:
    return None if value is None or (isinstance(value, float) and pd.isna(value)) else value


def make_run_id(stage: str, cutoff_time: datetime, sku_code: str, payload: Mapping[str, Any]) -> str:
    digest = hashlib.sha256(_json(payload).encode("utf-8")).hexdigest()[:12]
    return f"{stage}-{cutoff_time.strftime('%Y%m%dT%H%M%S')}-{sku_code}-{digest}"[:64]


class ResultWriter:
    """Append-only persistence for completed standardized stage artifacts."""

    def __init__(self, database: Database) -> None:
        self.database = database

    def write_comparables(
        self,
        *,
        cutoff_time: datetime,
        candidates: pd.DataFrame,
        metadata: Mapping[str, Any],
    ) -> list[str]:
        run_ids: list[str] = []
        with self.database.transaction() as cursor:
            for sku, group in candidates.groupby("target_sku_code", sort=True):
                payload = group.to_dict(orient="records")
                run_id = make_run_id("M1", cutoff_time, str(sku), {"rows": payload, **metadata})
                self._insert_run(cursor, run_id, str(sku), "COMPARABLE", cutoff_time, metadata)
                for row in payload:
                    cursor.execute(
                        """
                        INSERT INTO algo_candidate_detail
                        (run_id,candidate_sku_code,path_code,hard_filter_pass,
                         similarity_scores_json,quality_score,alpha_weight,
                         history_restore_metrics_json,exclusion_reason)
                        VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                        """,
                        (
                            run_id, str(row["candidate_sku_code"]), str(row["path_code"]),
                            int(row["hard_filter_pass"]),
                            _json({key: value for key, value in row.items() if key.startswith("similarity_")}),
                            _sql_value(row.get("quality_score")), _sql_value(row.get("alpha_weight")),
                            _json({"historical_velocity": row.get("historical_velocity")}),
                            row.get("exclusion_reason") or None,
                        ),
                    )
                run_ids.append(run_id)
        return run_ids

    def write_first_order(
        self,
        *,
        cutoff_time: datetime,
        decisions: pd.DataFrame,
        calculations: pd.DataFrame,
        metadata: Mapping[str, Any],
    ) -> list[str]:
        run_ids: list[str] = []
        with self.database.transaction() as cursor:
            for row in decisions.to_dict(orient="records"):
                sku = str(row["sku_code"])
                run_id = make_run_id("FIRST", cutoff_time, sku, {"decision": row, **metadata})
                self._insert_run(cursor, run_id, sku, "FIRST_ORDER", cutoff_time, metadata)
                self._insert_prediction(cursor, run_id, "FIRST_ORDER", row)
                for detail in calculations.loc[calculations["sku_code"].astype(str) == sku].to_dict(orient="records"):
                    cursor.execute(
                        """
                        INSERT INTO algo_calc_detail
                        (run_id,calc_item_code,calc_sequence,raw_value_json,detail_json)
                        VALUES (%s,%s,%s,%s,%s)
                        """,
                        (run_id, detail["calc_item_code"], int(detail["calc_sequence"]), _json(detail.get("calc_value")), _json(detail)),
                    )
                run_ids.append(run_id)
        return run_ids

    def write_reorder(
        self,
        *,
        cutoff_time: datetime,
        decisions: pd.DataFrame,
        attribution: pd.DataFrame,
        metadata: Mapping[str, Any],
    ) -> list[str]:
        run_ids: list[str] = []
        with self.database.transaction() as cursor:
            for row in decisions.to_dict(orient="records"):
                sku = str(row["sku_code"])
                run_id = make_run_id("REORDER", cutoff_time, sku, {"decision": row, **metadata})
                self._insert_run(cursor, run_id, sku, "REORDER", cutoff_time, metadata)
                self._insert_prediction(cursor, run_id, "REORDER", row)
                for sequence, detail in enumerate(
                    attribution.loc[attribution["sku_code"].astype(str) == sku].to_dict(orient="records"), start=1
                ):
                    cursor.execute(
                        """
                        INSERT INTO algo_calc_detail
                        (run_id,calc_item_code,calc_sequence,raw_value_json,detail_json)
                        VALUES (%s,%s,%s,%s,%s)
                        """,
                        (run_id, f"ATTRIBUTION:{detail['factor_code']}", sequence, _json(detail.get("contribution")), _json(detail)),
                    )
                run_ids.append(run_id)
        return run_ids

    @staticmethod
    def _insert_run(
        cursor: Any,
        run_id: str,
        sku_code: str,
        decision_type: str,
        cutoff_time: datetime,
        metadata: Mapping[str, Any],
    ) -> None:
        now = datetime.now(cutoff_time.tzinfo).replace(tzinfo=None)
        cursor.execute(
            """
            INSERT INTO algo_decision_run
            (run_id,run_state_seq,sku_code,decision_type,decision_cutoff_time,
             trigger_source,run_started_at,run_finished_at,run_status,model_version,
             parameter_version,rule_version,dataset_version,input_hash,output_hash,
             superseded_run_id,status_message)
            VALUES (%s,1,%s,%s,%s,%s,%s,%s,'COMPLETED',%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                run_id, sku_code, decision_type, cutoff_time.replace(tzinfo=None),
                metadata.get("trigger_source", "artifact_persist"), now, now,
                metadata.get("model_version"), metadata.get("parameter_version"),
                metadata.get("rule_version"), metadata.get("dataset_version"),
                metadata.get("input_hash"), metadata.get("output_hash"),
                metadata.get("superseded_run_id"), metadata.get("status_message"),
            ),
        )

    @staticmethod
    def _insert_prediction(cursor: Any, run_id: str, horizon: str, row: Mapping[str, Any]) -> None:
        cursor.execute(
            """
            INSERT INTO algo_prediction
            (run_id,prediction_horizon,demand_prediction,q_nat,q_plan,q_feas,q_opp,
             sabc_grade,reorder_action,confidence_score,error_interval_lower,
             error_interval_upper,downgrade_reason)
            VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            """,
            (
                run_id, horizon, _sql_value(row.get("D_first", row.get("D_window"))),
                _sql_value(row.get("Q_nat")), _sql_value(row.get("Q_plan")), _sql_value(row.get("Q_feas", row.get("QR_feas"))),
                _sql_value(row.get("Q_opp", row.get("QR_opp"))), _sql_value(row.get("grade")), _sql_value(row.get("action")),
                _sql_value(row.get("reliability_score")), _sql_value(row.get("prediction_p10")), _sql_value(row.get("prediction_p90")),
                None if row.get("reliability_status") != "uncalibrated" else "uncalibrated",
            ),
        )
