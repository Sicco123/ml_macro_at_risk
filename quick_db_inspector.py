#!/usr/bin/env python3
"""
Quick Database Inspector for task_coordination.db

A simple script for quick inspection of the task coordination database.
"""

import sqlite3
import pandas as pd
from pathlib import Path


def quick_db_summary(db_path: str = "/home/skooiker/ml_macro_at_risk/outputs/task_coordination.db"):
    """Quick summary of the database structure and content."""
    
    if not Path(db_path).exists():
        print(f"Database not found: {db_path}")
        return
    
    conn = sqlite3.connect(db_path)
    
    print(f"Database: {db_path}")
    print(f"Size: {Path(db_path).stat().st_size / (1024*1024):.1f} MB")
    print()
    
    # Table overview
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [table[0] for table in cursor.fetchall()]
    
    print("Tables and row counts:")
    total_rows = 0
    for table in tables:
        cursor.execute(f'SELECT COUNT(*) FROM {table}')
        count = cursor.fetchone()[0]
        total_rows += count
        print(f"  {table}: {count:,} rows")
        
        # Get column information
        cursor.execute(f'PRAGMA table_info({table})')
        columns = cursor.fetchall()
        column_names = [col[1] for col in columns]
        print(f"    Columns: {', '.join(column_names)}")
    
    print(f"  Total: {total_rows:,} rows")
    print()
    
    # Key statistics
    if 'completed_tasks' in tables:
        print("Completed Tasks Summary:")
        
        # Models
        df = pd.read_sql_query("SELECT model, COUNT(*) as count FROM completed_tasks GROUP BY model", conn)
        print("  Models:")
        for _, row in df.iterrows():
            print(f"    {row['model']}: {row['count']:,}")
        
        # Global vs country tasks
        df = pd.read_sql_query("SELECT is_global, COUNT(*) as count FROM completed_tasks GROUP BY is_global", conn)
        global_count = df[df['is_global'] == 1]['count'].iloc[0] if len(df[df['is_global'] == 1]) > 0 else 0
        country_count = df[df['is_global'] == 0]['count'].iloc[0] if len(df[df['is_global'] == 0]) > 0 else 0
        print(f"  Global tasks: {global_count:,}")
        print(f"  Country-specific tasks: {country_count:,}")
        
        # Quantiles and horizons
        df = pd.read_sql_query("SELECT DISTINCT quantile FROM completed_tasks ORDER BY quantile", conn)
        quantiles = df['quantile'].tolist()
        print(f"  Quantiles: {quantiles}")
        
        df = pd.read_sql_query("SELECT DISTINCT horizon FROM completed_tasks ORDER BY horizon", conn)
        horizons = df['horizon'].tolist()
        print(f"  Horizons: {horizons}")
        print()
    
    if 'forecasts' in tables:
        print("Forecasts Summary:")
        
        # Date range
        df = pd.read_sql_query("SELECT MIN(time) as min_date, MAX(time) as max_date FROM forecasts", conn)
        print(f"  Date range: {df['min_date'].iloc[0]} to {df['max_date'].iloc[0]}")
        
        # Countries
        df = pd.read_sql_query("SELECT COUNT(DISTINCT country) as country_count FROM forecasts", conn)
        print(f"  Countries: {df['country_count'].iloc[0]}")
        
        # Models in forecasts
        df = pd.read_sql_query("SELECT DISTINCT model FROM forecasts ORDER BY model", conn)
        models = df['model'].tolist()
        print(f"  Models: {models}")
    
    conn.close()


if __name__ == "__main__":
    quick_db_summary()
