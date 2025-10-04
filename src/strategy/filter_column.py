from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FilterColumn:
    """Metadata describing a filter value to record in trade reports."""

    column_key: str
    output: str
    output_form: str
    parameter: str
    label: str | None = None
