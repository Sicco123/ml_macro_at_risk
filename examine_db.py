"""
Delete rows from completed_tasks for candidate missing forecasts.

For each (quantile, horizon, version, missing_date) tuple found in
missing_dates_report.csv, this script deletes rows from the
completed_tasks table where quantile=q AND horizon=h AND version=v AND window_end=d.

Safety: creates a timestamped backup of the database before deletion.

Usage: python3 export_completed_tasks_for_missing_candidates.py
"""

import csv
import os
import shutil
import sqlite3
from datetime import datetime
from typing import Iterable, Tuple, Set

DB_PATH = "/home/skooiker/ml_macro_at_risk/outputs/task_coordination.db"
MISSING_REPORT_CSV = "/home/skooiker/ml_macro_at_risk/missing_dates_report.csv"
OUTPUT_CSV = "/home/skooiker/ml_macro_at_risk/completed_tasks_for_missing_candidates.csv"  # unused now


def read_missing_tuples(path: str) -> Set[Tuple[float, int, str, str]]:
    """Read unique (quantile, horizon, version, missing_date) tuples from CSV."""
    tuples: Set[Tuple[float, int, str, str]] = set()
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        required = {"quantile", "horizon", "version", "missing_date"}
        if not required.issubset(reader.fieldnames or {}):
            raise ValueError(
                f"Input CSV must contain columns: {sorted(required)}; found {reader.fieldnames}"
            )
        for row in reader:
            try:
                q = float(row["quantile"]) if row["quantile"] != "" else None
                h = int(row["horizon"]) if row["horizon"] != "" else None
                v = row["version"]
                d = row["missing_date"]
                if q is None or h is None or not v or not d:
                    continue
                tuples.add((q, h, v, d))
            except Exception:
                # Skip malformed rows
                continue
    return tuples


def get_completed_tasks_columns(conn: sqlite3.Connection) -> Iterable[str]:
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(completed_tasks)")
    cols = [r[1] for r in cur.fetchall()]
    if not cols:
        # Fallback: try to infer via a dummy select
        cur.execute("SELECT * FROM completed_tasks LIMIT 0")
        cols = [d[0] for d in (cur.description or [])]
    return cols

def get_rows(sql, params=()) -> list:
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        rows = cur.fetchall()
        return rows
    finally:
        conn.close()

def main():
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"Database not found at {DB_PATH}")
    if not os.path.exists(MISSING_REPORT_CSV):
        raise FileNotFoundError(f"Missing report CSV not found at {MISSING_REPORT_CSV}")

    pairs = read_missing_tuples(MISSING_REPORT_CSV)
    if not pairs:
        print("No (q,h,v,d) tuples found in missing report. Nothing to delete.")
        return

    conn = sqlite3.connect(DB_PATH)
    try:
        # Ensure table exists (and fetch columns just to validate schema exists)
        _ = list(get_completed_tasks_columns(conn))

        
        select_sql = (
            "SELECT * FROM completed_tasks "
            "WHERE quantile = ? AND horizon = ? AND version = ?AND window_end = ?"
        )


        q = 0.95
        h = 1
        v = "T141"
        d = "2022-04-01"
        country = "NL"
        print(f"Processing tuple (q={q}, h={h}, v={v}, d={d})...")
        rows = get_rows(select_sql, (q, h, v, d))
        print(len(rows))
        
           
        select_sql = (
            "SELECT * FROM forecasts "
            "WHERE quantile = ? AND horizon = ? AND version = ?AND window_end = ? AND country = ?"
        )


        q = 0.95
        h = 1
        v = "T141"
        d = "2022-03-01"
        country = "BE"

        print(f"Processing tuple (q={q}, h={h}, v={v}, d={d}, country={country})...")
        rows = get_rows(select_sql, (q, h, v, d, country))
        print(len(rows))

        for row in rows:
            print(row)
        

    finally:
        conn.close()


if __name__ == "__main__":
    main()
