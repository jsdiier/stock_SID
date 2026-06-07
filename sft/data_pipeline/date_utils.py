"""
ISO week ↔ date conversion utilities.

All week strings follow the format "YYYY-WNN" (zero-padded), which sorts
correctly as plain strings and matches the format used in train_meta.csv.
"""
from datetime import datetime, timedelta


def yyyymmdd_to_isoweek(date_str: str) -> str:
    """'20250101' → '2025-W01'"""
    d = datetime.strptime(date_str, '%Y%m%d')
    yr, wk, _ = d.isocalendar()
    return f"{yr}-W{wk:02d}"


def isoweek_monday(year_week: str) -> datetime:
    """'2025-W01' → datetime of that Monday (ISO 8601)."""
    yr, wk = year_week.split('-W')
    return datetime.strptime(f"{yr}-W{wk}-1", "%G-W%V-%u")


def weeks_in_range(start_yyyymmdd: str, end_yyyymmdd: str) -> list:
    """
    Return sorted list of ISO year-week strings whose Monday falls within
    the [start_yyyymmdd, end_yyyymmdd] date range (both inclusive).

    Example
    -------
    weeks_in_range('20250101', '20250131')
    → ['2025-W01', '2025-W02', '2025-W03', '2025-W04', '2025-W05']
    """
    start = datetime.strptime(start_yyyymmdd, '%Y%m%d')
    end   = datetime.strptime(end_yyyymmdd,   '%Y%m%d')

    # Back up to the Monday of the week containing start
    monday = start - timedelta(days=start.isoweekday() - 1)

    result = []
    current = monday
    while current <= end:
        yr, wk, _ = current.isocalendar()
        result.append(f"{yr}-W{wk:02d}")
        current += timedelta(weeks=1)

    return result
