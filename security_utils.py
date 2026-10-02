from __future__ import annotations

import pandas as pd


MAX_UPLOAD_SIZE_BYTES = 25 * 1024 * 1024
_FORMULA_PREFIXES = ("=", "+", "-", "@")
_IGNORABLE_PREFIXES = " \t\r\n"


def neutralize_spreadsheet_formulas(value: object) -> object:
    """Prefix dangerous text cells so spreadsheet software treats them as text."""
    if isinstance(value, str) and value.lstrip(_IGNORABLE_PREFIXES).startswith(
        _FORMULA_PREFIXES
    ):
        return "'" + value
    return value


def prepare_csv_export(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy safe to open in spreadsheet software without formula execution."""
    safe_frame = frame.copy()
    for column in safe_frame.select_dtypes(include=["object", "string"]).columns:
        safe_frame[column] = safe_frame[column].map(neutralize_spreadsheet_formulas)
    return safe_frame
