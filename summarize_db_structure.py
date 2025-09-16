#!/usr/bin/env python3
"""
Database Structure Summary Tool for task_coordination.db

This script analyzes and summarizes the structure of the task coordination database
used in the ML Macro at Risk project.

Author: Generated for ml_macro_at_risk project
Date: September 15, 2025
"""

import sqlite3
import pandas as pd
from typing import Dict, List, Any
import os
from datetime import datetime


class DatabaseStructureSummarizer:
    """Summarizes the structure and content of the task coordination database."""
    
    def __init__(self, db_path: str):
        """Initialize with database path."""
        self.db_path = db_path
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"Database not found: {db_path}")
    
    def get_table_info(self) -> Dict[str, List[Dict[str, Any]]]:
        """Get detailed information about all tables in the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get all table names
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [table[0] for table in cursor.fetchall()]
        
        table_info = {}
        
        for table in tables:
            # Get column information
            cursor.execute(f'PRAGMA table_info({table})')
            columns = cursor.fetchall()
            
            # Get row count
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            row_count = cursor.fetchone()[0]
            
            # Parse column information
            column_details = []
            primary_keys = []
            
            for col in columns:
                col_info = {
                    'name': col[1],
                    'type': col[2],
                    'not_null': col[3] == 1,
                    'default_value': col[4],
                    'is_primary_key': col[5] > 0
                }
                column_details.append(col_info)
                
                if col[5] > 0:
                    primary_keys.append(col[1])
            
            table_info[table] = {
                'columns': column_details,
                'primary_keys': primary_keys,
                'row_count': row_count
            }
        
        conn.close()
        return table_info
    
    def get_sample_data(self, table_name: str, limit: int = 5) -> pd.DataFrame:
        """Get sample data from a specific table."""
        conn = sqlite3.connect(self.db_path)
        try:
            df = pd.read_sql_query(f'SELECT * FROM {table_name} LIMIT {limit}', conn)
            return df
        finally:
            conn.close()
    
    def get_unique_values(self, table_name: str, column_name: str, limit: int = 20) -> List[Any]:
        """Get unique values from a specific column."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(f'SELECT DISTINCT {column_name} FROM {table_name} ORDER BY {column_name} LIMIT {limit}')
            values = [row[0] for row in cursor.fetchall()]
            return values
        finally:
            conn.close()
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get overall database statistics."""
        conn = sqlite3.connect(self.db_path)
        
        stats = {
            'file_size_mb': os.path.getsize(self.db_path) / (1024 * 1024),
            'total_tables': 0,
            'total_rows': 0,
            'last_modified': datetime.fromtimestamp(os.path.getmtime(self.db_path))
        }
        
        table_info = self.get_table_info()
        stats['total_tables'] = len(table_info)
        stats['total_rows'] = sum(info['row_count'] for info in table_info.values())
        
        conn.close()
        return stats
    
    def analyze_task_completion_stats(self) -> Dict[str, Any]:
        """Analyze task completion statistics."""
        conn = sqlite3.connect(self.db_path)
        
        stats = {}
        
        # Task counts by status
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM completed_tasks')
        stats['completed_tasks'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM claimed_tasks')
        stats['claimed_tasks'] = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM failed_tasks')
        stats['failed_tasks'] = cursor.fetchone()[0]
        
        # Task distribution by model
        cursor.execute('SELECT model, COUNT(*) FROM completed_tasks GROUP BY model ORDER BY COUNT(*) DESC')
        stats['tasks_by_model'] = dict(cursor.fetchall())
        
        # Task distribution by country
        cursor.execute('SELECT country, COUNT(*) FROM completed_tasks WHERE country != "__ALL__" GROUP BY country ORDER BY COUNT(*) DESC')
        stats['tasks_by_country'] = dict(cursor.fetchall())
        
        # Global vs country-specific tasks
        cursor.execute('SELECT is_global, COUNT(*) FROM completed_tasks GROUP BY is_global')
        global_stats = dict(cursor.fetchall())
        stats['global_tasks'] = global_stats.get(1, 0)
        stats['country_specific_tasks'] = global_stats.get(0, 0)
        
        conn.close()
        return stats
    
    def print_summary(self):
        """Print a comprehensive summary of the database structure."""
        print("=" * 80)
        print("TASK COORDINATION DATABASE STRUCTURE SUMMARY")
        print("=" * 80)
        print(f"Database: {self.db_path}")
        print()
        
        # Overall statistics
        stats = self.get_database_stats()
        print("OVERALL STATISTICS:")
        print(f"  File size: {stats['file_size_mb']:.2f} MB")
        print(f"  Total tables: {stats['total_tables']}")
        print(f"  Total rows: {stats['total_rows']:,}")
        print(f"  Last modified: {stats['last_modified']}")
        print()
        
        # Table information
        table_info = self.get_table_info()
        
        print("TABLES OVERVIEW:")
        for table_name, info in table_info.items():
            print(f"\n  {table_name.upper()} ({info['row_count']:,} rows)")
            print(f"    Primary Key: {', '.join(info['primary_keys']) if info['primary_keys'] else 'None'}")
            print("    Columns:")
            for col in info['columns']:
                pk_indicator = " (PK)" if col['is_primary_key'] else ""
                null_indicator = " NOT NULL" if col['not_null'] else ""
                default_info = f" DEFAULT {col['default_value']}" if col['default_value'] is not None else ""
                print(f"      - {col['name']} ({col['type']}){pk_indicator}{null_indicator}{default_info}")
        
        print("\n" + "=" * 80)
        print("DETAILED TABLE ANALYSIS")
        print("=" * 80)
        
        # Detailed analysis for each table
        for table_name in table_info.keys():
            print(f"\n{table_name.upper()} TABLE:")
            print("-" * 40)
            
            # Sample data
            sample_df = self.get_sample_data(table_name, 3)
            if not sample_df.empty:
                print("Sample data (first 3 rows):")
                print(sample_df.to_string(index=False, max_cols=10))
                print()
            
            # Key column analysis
            if table_name in ['completed_tasks', 'claimed_tasks', 'failed_tasks']:
                # Analyze task-related tables
                print("Key statistics:")
                
                models = self.get_unique_values(table_name, 'model')
                print(f"  Models: {models}")
                
                countries = self.get_unique_values(table_name, 'country', 10)
                print(f"  Countries (sample): {countries}")
                
                quantiles = self.get_unique_values(table_name, 'quantile')
                print(f"  Quantiles: {quantiles}")
                
                horizons = self.get_unique_values(table_name, 'horizon')
                print(f"  Horizons: {horizons}")
                
            elif table_name == 'forecasts':
                print("Key statistics:")
                
                models = self.get_unique_values(table_name, 'model')
                print(f"  Models: {models}")
                
                countries = self.get_unique_values(table_name, 'country', 10)
                print(f"  Countries (sample): {countries}")
                
                # Date range
                conn = sqlite3.connect(self.db_path)
                cursor = conn.cursor()
                cursor.execute('SELECT MIN(time), MAX(time) FROM forecasts')
                min_date, max_date = cursor.fetchone()
                print(f"  Date range: {min_date} to {max_date}")
                conn.close()
        
        # Task completion analysis
        print("\n" + "=" * 80)
        print("TASK COMPLETION ANALYSIS")
        print("=" * 80)
        
        completion_stats = self.analyze_task_completion_stats()
        
        print("Task Status Summary:")
        print(f"  Completed: {completion_stats['completed_tasks']:,}")
        print(f"  Currently claimed: {completion_stats['claimed_tasks']:,}")
        print(f"  Failed: {completion_stats['failed_tasks']:,}")
        print()
        
        print("Task Distribution:")
        print(f"  Global tasks: {completion_stats['global_tasks']:,}")
        print(f"  Country-specific tasks: {completion_stats['country_specific_tasks']:,}")
        print()
        
        print("Tasks by Model:")
        for model, count in completion_stats['tasks_by_model'].items():
            print(f"  {model}: {count:,}")
        print()
        
        if completion_stats['tasks_by_country']:
            print("Top Countries by Task Count:")
            for country, count in list(completion_stats['tasks_by_country'].items())[:10]:
                print(f"  {country}: {count:,}")
        
        print("\n" + "=" * 80)
        print("SUMMARY COMPLETE")
        print("=" * 80)


def main():
    """Main function to run the database structure summary."""
    db_path = "/home/skooiker/ml_macro_at_risk/outputs/task_coordination.db"
    
    try:
        summarizer = DatabaseStructureSummarizer(db_path)
        summarizer.print_summary()
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"An error occurred: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
