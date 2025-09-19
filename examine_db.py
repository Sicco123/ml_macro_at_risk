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
def get_claimed_tasks_columns(conn: sqlite3.Connection) -> Iterable[str]:
    cur = conn.cursor()
    cur.execute("PRAGMA table_info(claimed_tasks)")
    cols = [r[1] for r in cur.fetchall()]
    if not cols:
        # Fallback: try to infer via a dummy select
        cur.execute("SELECT * FROM claimed_tasks LIMIT 0")
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

def delete_rows(sql, params=()) -> int:
    """Execute a DELETE statement and return number of rows affected."""
    conn = sqlite3.connect(DB_PATH)
    try:
        cur = conn.cursor()
        cur.execute(sql, params)
        affected_rows = cur.rowcount
        conn.commit()  # Important: commit the transaction
        return affected_rows
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

    # Create backup before deletion
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{DB_PATH}.backup_{timestamp}"
    shutil.copy2(DB_PATH, backup_path)
    print(f"Created backup: {backup_path}")

    conn = sqlite3.connect(DB_PATH)
    try:
        # Fix SQL syntax (missing space before AND)
        delete_claimed_sql = (
            "DELETE FROM claimed_tasks "
            "WHERE version = ?"
        )

        # delete_forecasts_sql = (
        #     "DELETE FROM forecasts "
        #     "WHERE quantile = ? AND horizon = ? AND version = ? AND window_end = ? AND country = ?"
        # )

        # Delete from claimed_tasks
        q, h, v, d = 0.05, 1, "panel-ar", "2020-04-01"
        v = "T142"
        print(f"Deleting from claimed_tasks: (q={q}, h={h}, v={v}, d={d})...")
        affected = delete_rows(delete_claimed_sql, (v,))
        print(f"Deleted {affected} rows from claimed_tasks")

        # # Delete from forecasts
        # q, h, v, d, country = 0.95, 1, "T141", "2022-03-01", "BE"
        # print(f"Deleting from forecasts: (q={q}, h={h}, v={v}, d={d}, country={country})...")
        # affected = delete_rows(delete_forecasts_sql, (q, h, v, d, country))
        # print(f"Deleted {affected} rows from forecasts")

       
    finally:
        conn.close()

# def main():
#     if not os.path.exists(DB_PATH):
#         raise FileNotFoundError(f"Database not found at {DB_PATH}")
#     if not os.path.exists(MISSING_REPORT_CSV):
#         raise FileNotFoundError(f"Missing report CSV not found at {MISSING_REPORT_CSV}")

#     pairs = read_missing_tuples(MISSING_REPORT_CSV)
#     if not pairs:
#         print("No (q,h,v,d) tuples found in missing report. Nothing to delete.")
#         return

#     conn = sqlite3.connect(DB_PATH)
#     try:
#         # Ensure table exists (and fetch columns just to validate schema exists)
#         cols = list(get_claimed_tasks_columns(conn))
#         print(cols)

        
#         select_sql = (
#             "SELECT * FROM completed_tasks "
#             "WHERE version = ?"
#         )


#         v = "T142"
#         rows = get_rows(select_sql, (v,))
#         print(len(rows))
#         print(rows)

#         # get unique versions in claimed_tasks
#         select_versions_sql = (
#             "SELECT DISTINCT version FROM completed_tasks"
#         )
#         versions = get_rows(select_versions_sql)
#         print("Unique versions in claimed_tasks:")
#         for v in versions:
#             print(v)

#         # select_sql = (
#         #     "SELECT * FROM forecasts "
#         #     "WHERE quantile = ? AND horizon = ? AND version = ?AND window_end = ? AND country = ?"
#         # )


#         # q = 0.95
#         # h = 1
#         # v = "T141"
#         # d = "2022-03-01"
#         # country = "BE"

#         # print(f"Processing tuple (q={q}, h={h}, v={v}, d={d}, country={country})...")
#         # rows = get_rows(select_sql, (q, h, v, d, country))
#         # print(len(rows))

#         # for row in rows:
#         #     print(row)
        

#     finally:
#         conn.close()


if __name__ == "__main__":
    main()
