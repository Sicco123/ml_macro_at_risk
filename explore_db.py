"""
Simple script to explore the database structure
"""

import sqlite3

def explore_database(db_path):
    """Explore the database structure."""
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print("=== Available Tables ===")
        for table in tables:
            print(f"- {table[0]}")
        
        print("\n=== Table Details ===")
        for table in tables:
            table_name = table[0]
            print(f"\nTable: {table_name}")
            
            # Get schema
            cursor.execute(f"PRAGMA table_info({table_name})")
            columns = cursor.fetchall()
            print("Columns:")
            for col in columns:
                print(f"  {col[1]} ({col[2]})")
            
            # Get row count
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"Row count: {count}")
            
            # Show a few sample rows
            if count > 0:
                cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
                sample_rows = cursor.fetchall()
                print("Sample data:")
                for row in sample_rows:
                    print(f"  {row}")
        
        conn.close()
        
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    db_path = "/home/skooiker/ml_macro_at_risk/outputs/task_coordination.db"
    explore_database(db_path)
