#!/usr/bin/env python3
"""
Migration utility to handle the transition from Parquet-based to SQLite-based task coordination.

This script:
1. Imports existing progress from Parquet files into SQLite
2. Optionally cleans up old Parquet files
3. Shows comparison between old and new systems
"""

import sqlite3
import pandas as pd
from pathlib import Path
import argparse
import sys
from datetime import datetime

def import_parquet_to_sqlite(progress_dir: Path, db_path: Path, dry_run: bool = False):
    """Import existing Parquet progress data into SQLite database"""
    
    print(f"Importing Parquet data from {progress_dir} to SQLite {db_path}")
    
    # Initialize SQLite database
    if not dry_run:
        with sqlite3.connect(str(db_path)) as conn:
            # Enable WAL mode
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            
            # Create tables if they don't exist
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS completed_tasks (
                    model TEXT NOT NULL, version TEXT, quantile REAL NOT NULL, horizon INTEGER NOT NULL,
                    country TEXT NOT NULL, window_start TEXT NOT NULL, window_end TEXT NOT NULL, is_global INTEGER NOT NULL DEFAULT 0,
                    completed_at TEXT NOT NULL, instance_id TEXT NOT NULL, model_path TEXT, forecast_rows INTEGER DEFAULT 0,
                    PRIMARY KEY (model, version, quantile, horizon, country, window_start, window_end, is_global)
                );
                
                CREATE TABLE IF NOT EXISTS claimed_tasks (
                    model TEXT NOT NULL, version TEXT, quantile REAL NOT NULL, horizon INTEGER NOT NULL,
                    country TEXT NOT NULL, window_start TEXT NOT NULL, window_end TEXT NOT NULL, is_global INTEGER NOT NULL DEFAULT 0,
                    claimed_at TEXT NOT NULL, instance_id TEXT NOT NULL, expires_at TEXT NOT NULL,
                    PRIMARY KEY (model, version, quantile, horizon, country, window_start, window_end, is_global)
                );
                
                CREATE TABLE IF NOT EXISTS failed_tasks (
                    model TEXT NOT NULL, version TEXT, quantile REAL NOT NULL, horizon INTEGER NOT NULL,
                    country TEXT NOT NULL, window_start TEXT NOT NULL, window_end TEXT NOT NULL, is_global INTEGER NOT NULL DEFAULT 0,
                    failed_at TEXT NOT NULL, instance_id TEXT NOT NULL, error_msg TEXT,
                    PRIMARY KEY (model, version, quantile, horizon, country, window_start, window_end, is_global)
                );
                
                CREATE INDEX IF NOT EXISTS idx_completed_country ON completed_tasks(country);
                CREATE INDEX IF NOT EXISTS idx_claimed_expires ON claimed_tasks(expires_at);
                CREATE INDEX IF NOT EXISTS idx_failed_country ON failed_tasks(country);
            """)
    
    # Import completed tasks
    completed_parquet = progress_dir / "completed_tasks.parquet"
    completed_count = 0
    if completed_parquet.exists():
        try:
            df = pd.read_parquet(completed_parquet)
            print(f"Found {len(df)} completed tasks in Parquet file")
            
            if not dry_run and not df.empty:
                with sqlite3.connect(str(db_path)) as conn:
                    for _, row in df.iterrows():
                        conn.execute("""
                            INSERT OR IGNORE INTO completed_tasks 
                            (model, version, quantile, horizon, country, window_start, window_end, is_global, completed_at, instance_id, model_path, forecast_rows)
                            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            row.get('MODEL', ''), row.get('VERSION', ''), float(row.get('QUANTILE', 0)),
                            int(row.get('HORIZON', 0)), row.get('COUNTRY', ''), row.get('WINDOW_START', ''),
                            row.get('WINDOW_END', ''), int(row.get('IS_GLOBAL', 0)), 
                            row.get('LAST_UPDATE', datetime.now().isoformat()),
                            row.get('INSTANCE_ID', 'migrated'), row.get('MODEL_PATH', None),
                            int(row.get('FORECAST_ROWS', 0))
                        ))
                    conn.commit()
                    completed_count = len(df)
        except Exception as e:
            print(f"Warning: Could not import completed tasks: {e}")
    
    # Import failed tasks from existing data (if available)
    failed_count = 0
    # Note: Original system didn't have a separate failed_tasks.parquet, 
    # failed tasks were in the main progress file with STATUS='failed'
    
    # Check if there's a main progress file
    main_progress = progress_dir / "progress.parquet"
    if main_progress.exists():
        try:
            df = pd.read_parquet(main_progress)
            failed_df = df[df.get('STATUS') == 'failed'] if 'STATUS' in df.columns else pd.DataFrame()
            
            if not failed_df.empty:
                print(f"Found {len(failed_df)} failed tasks in main progress file")
                
                if not dry_run:
                    with sqlite3.connect(str(db_path)) as conn:
                        for _, row in failed_df.iterrows():
                            conn.execute("""
                                INSERT OR IGNORE INTO failed_tasks 
                                (model, version, quantile, horizon, country, window_start, window_end, is_global, failed_at, instance_id, error_msg)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                row.get('MODEL', ''), row.get('VERSION', ''), float(row.get('QUANTILE', 0)),
                                int(row.get('HORIZON', 0)), row.get('COUNTRY', ''), row.get('WINDOW_START', ''),
                                row.get('WINDOW_END', ''), int(row.get('IS_GLOBAL', 0)),
                                row.get('LAST_UPDATE', datetime.now().isoformat()),
                                row.get('INSTANCE_ID', 'migrated'), row.get('ERROR_MSG', None)
                            ))
                        conn.commit()
                        failed_count = len(failed_df)
        except Exception as e:
            print(f"Warning: Could not import failed tasks: {e}")
    
    # Don't import claimed tasks from old system since they're likely stale
    
    print(f"Migration summary:")
    print(f"  Completed tasks imported: {completed_count}")
    print(f"  Failed tasks imported: {failed_count}")
    print(f"  Claimed tasks: Skipped (likely stale)")
    
    if dry_run:
        print("  (DRY RUN - no actual changes made)")

def compare_systems(progress_dir: Path, db_path: Path):
    """Compare task counts between Parquet and SQLite systems"""
    
    print("\n" + "="*60)
    print("COMPARISON: Parquet vs SQLite Task Tracking")
    print("="*60)
    
    # Parquet counts
    parquet_completed = 0
    parquet_claimed = 0
    parquet_failed = 0
    
    completed_parquet = progress_dir / "completed_tasks.parquet"
    if completed_parquet.exists():
        try:
            df = pd.read_parquet(completed_parquet)
            parquet_completed = len(df)
        except:
            pass
    
    claimed_parquet = progress_dir / "claimed_tasks.parquet"
    if claimed_parquet.exists():
        try:
            df = pd.read_parquet(claimed_parquet)
            parquet_claimed = len(df)
        except:
            pass
    
    main_progress = progress_dir / "progress.parquet"
    if main_progress.exists():
        try:
            df = pd.read_parquet(main_progress)
            if 'STATUS' in df.columns:
                parquet_failed = len(df[df['STATUS'] == 'failed'])
        except:
            pass
    
    # SQLite counts
    sqlite_completed = 0
    sqlite_claimed = 0
    sqlite_failed = 0
    
    if db_path.exists():
        try:
            with sqlite3.connect(str(db_path)) as conn:
                cursor = conn.execute("SELECT COUNT(*) FROM completed_tasks")
                sqlite_completed = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM claimed_tasks")
                sqlite_claimed = cursor.fetchone()[0]
                
                cursor = conn.execute("SELECT COUNT(*) FROM failed_tasks")
                sqlite_failed = cursor.fetchone()[0]
        except Exception as e:
            print(f"Warning: Could not read SQLite database: {e}")
    
    print(f"Parquet system:")
    print(f"  Completed: {parquet_completed}")
    print(f"  Claimed:   {parquet_claimed}")
    print(f"  Failed:    {parquet_failed}")
    print(f"  Total:     {parquet_completed + parquet_claimed + parquet_failed}")
    
    print(f"\nSQLite system:")
    print(f"  Completed: {sqlite_completed}")
    print(f"  Claimed:   {sqlite_claimed}")
    print(f"  Failed:    {sqlite_failed}")
    print(f"  Total:     {sqlite_completed + sqlite_claimed + sqlite_failed}")
    
    print("="*60)

def cleanup_parquet_files(progress_dir: Path, dry_run: bool = False):
    """Clean up old Parquet files (with backup)"""
    
    parquet_files = [
        "completed_tasks.parquet",
        "claimed_tasks.parquet", 
        "progress.parquet",
        "errors.parquet"
    ]
    
    backup_dir = progress_dir / "parquet_backup"
    
    print(f"\nCleaning up Parquet files...")
    
    if not dry_run:
        backup_dir.mkdir(exist_ok=True)
    
    cleaned_files = []
    for filename in parquet_files:
        file_path = progress_dir / filename
        if file_path.exists():
            if not dry_run:
                # Create backup
                backup_path = backup_dir / filename
                file_path.rename(backup_path)
                print(f"  Moved {filename} to backup: {backup_path}")
            else:
                print(f"  Would move {filename} to backup")
            cleaned_files.append(filename)
    
    # Also clean up lock files
    lock_files = list(progress_dir.glob("*.lock"))
    for lock_file in lock_files:
        if not dry_run:
            lock_file.unlink()
            print(f"  Removed lock file: {lock_file.name}")
        else:
            print(f"  Would remove lock file: {lock_file.name}")
        cleaned_files.append(lock_file.name)
    
    if not cleaned_files:
        print("  No Parquet files found to clean up")
    elif dry_run:
        print(f"  (DRY RUN - would clean up {len(cleaned_files)} files)")
    else:
        print(f"  Cleaned up {len(cleaned_files)} files (backed up to {backup_dir})")

def main():
    parser = argparse.ArgumentParser(description="Migrate from Parquet to SQLite task coordination")
    parser.add_argument("--progress-dir", type=Path, default="outputs/progress", 
                       help="Directory containing Parquet progress files")
    parser.add_argument("--db-path", type=Path, default="outputs/task_coordination.db",
                       help="SQLite database path")
    parser.add_argument("--import", action="store_true", dest="do_import",
                       help="Import Parquet data to SQLite")
    parser.add_argument("--compare", action="store_true",
                       help="Compare Parquet vs SQLite task counts")
    parser.add_argument("--cleanup", action="store_true",
                       help="Clean up old Parquet files (creates backup)")
    parser.add_argument("--dry-run", action="store_true",
                       help="Show what would be done without making changes")
    
    args = parser.parse_args()
    
    if not any([args.do_import, args.compare, args.cleanup]):
        parser.print_help()
        print("\nExample usage:")
        print("  python migrate_parquet_to_sqlite.py --compare")
        print("  python migrate_parquet_to_sqlite.py --import --dry-run")
        print("  python migrate_parquet_to_sqlite.py --import --cleanup")
        return
    
    # Ensure paths exist
    if not args.progress_dir.exists() and (args.do_import or args.compare):
        print(f"Error: Progress directory {args.progress_dir} does not exist")
        return 1
    
    if args.do_import:
        import_parquet_to_sqlite(args.progress_dir, args.db_path, args.dry_run)
    
    if args.compare:
        compare_systems(args.progress_dir, args.db_path)
    
    if args.cleanup:
        cleanup_parquet_files(args.progress_dir, args.dry_run)
    
    print("\nMigration completed!")
    print(f"SQLite database: {args.db_path}")
    print("The runner will now use SQLite for much faster task coordination.")

if __name__ == "__main__":
    sys.exit(main())
