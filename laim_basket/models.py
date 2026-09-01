"""Небольшие явные контракты, которые проходят через продовый конвейер."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .publish import PublishedUmr
    from .reading.xlsx_reader import RawSheet
    from .transform.region import TableRegion


@dataclass(frozen=True)
class RunContext:
    basket_id: str
    file_hashes: dict[str, str]
    sheets: dict[str, RawSheet]
    documents: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class ResolvedLayout:
    basket_id: str
    sheet_name: str
    header_rows: tuple[int, ...]
    first_data_row: int
    last_data_row: int
    roles: dict[str, object]
    grouping: dict[str, object]
    dialogue_blob: dict[str, object] | None
    weight: dict[str, str] | None
    evidence: dict[str, str]
    column_names: dict[str, str]
    formula_rows: dict[str, tuple[int, ...]]
    region: TableRegion = field(repr=False, compare=False)

    def transform_config(self) -> dict[str, object]:
        return {
            "basket_id": self.basket_id,
            "roles": self.roles,
            "grouping": self.grouping,
            "dialogue_blob": self.dialogue_blob,
            "weight": self.weight,
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "layout_version": "laim-layout.v1",
            "basket_id": self.basket_id,
            "sheet_name": self.sheet_name,
            "header_rows": list(self.header_rows),
            "first_data_row": self.first_data_row,
            "last_data_row": self.last_data_row,
            "roles": self.roles,
            "grouping": self.grouping,
            "dialogue_blob": self.dialogue_blob,
            "weight": self.weight,
            "evidence": self.evidence,
            "column_names": self.column_names,
            "formula_rows": {key: list(value) for key, value in self.formula_rows.items()},
        }


@dataclass(frozen=True)
class MeasurementPlan:
    basket_id: str
    metric_name: str
    assessment_mode: str
    method: str
    sources: tuple[dict[str, object], ...]
    missing_policy: str
    majority_denominator: str | None
    reducer: str
    threshold: Decimal | None
    comparator: str | None
    scale: str
    precision: int
    reported_value: Decimal | None
    reported_raw: str | None
    evidence: dict[str, tuple[str, ...]]

    @property
    def evaluation_unit(self) -> str:
        return "dialogue" if self.assessment_mode == "dialogue" else "turn"

    def to_dict(self) -> dict[str, object]:
        reported = None
        if self.reported_value is not None:
            reported = {
                "value": str(self.reported_value),
                "raw": self.reported_raw,
            }
        return {
            "plan_version": "laim-measurement-plan.v2",
            "basket_id": self.basket_id,
            "metric_name": self.metric_name,
            "assessment_mode": self.assessment_mode,
            "score": {
                "method": self.method,
                "sources": list(self.sources),
                "missing_policy": self.missing_policy,
                "majority_denominator": self.majority_denominator,
            },
            "reducer": {"method": self.reducer},
            "release": {
                "threshold": str(self.threshold) if self.threshold is not None else None,
                "comparator": self.comparator,
                "scale": self.scale,
                "precision": self.precision,
            },
            "reported_value": reported,
            "evidence": {key: list(value) for key, value in self.evidence.items()},
        }


@dataclass(frozen=True)
class RunResult:
    status: str
    umr: PublishedUmr
    km: dict[str, object]
    excel_name: str
    measurement_plan: MeasurementPlan | None = None
    # Журнал прогона laim-run-report.v1 — содержимое порта km_result.
    report: dict[str, object] = field(default_factory=dict)
