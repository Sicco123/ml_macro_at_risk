"""
Script to analyze missing dates in the forecasts table of task_coordination.db

This script checks for missing window_end dates per quantile, horizon, and version
across all countries for the period 2016-02-01 to 2025-06-01.
"""

import sqlite3
from datetime import datetime, timedelta

def connect_to_db(db_path):
    """Connect to the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        return conn
    except sqlite3.Error as e:
        print(f"Error connecting to database: {e}")
        return None

def get_table_schema(conn, table_name):
    """Get the schema of a table."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = cursor.fetchall()
    return columns

def get_unique_values(conn, table_name, column_name):
    """Get unique values for a specific column."""
    cursor = conn.cursor()
    cursor.execute(f"SELECT DISTINCT {column_name} FROM {table_name} ORDER BY {column_name}")
    values = [row[0] for row in cursor.fetchall()]
    return values

def generate_expected_dates(start_date, end_date):
    """Generate expected month-start dates between start_date and end_date."""
    expected_dates = []
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_dt = datetime.strptime(end_date, '%Y-%m-%d')
    
    while current_date <= end_dt:
        # Ensure we're on the first day of the month
        month_start = current_date.replace(day=1)
        expected_dates.append(month_start.strftime('%Y-%m-%d'))
        
        # Move to next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    
    return expected_dates

def check_missing_dates(db_path):
    """Main function to check for missing dates in the forecasts table."""
    
    # Connect to database
    conn = connect_to_db(db_path)
    if not conn:
        return
    
    try:
        # First, let's examine the table structure
        print("=== forecasts Table Schema ===")
        schema = get_table_schema(conn, 'forecasts')
        for col in schema:
            print(f"{col[1]} ({col[2]})")
        print()
        
        # Get unique values for key columns
        print("=== Unique Values Analysis ===")
        quantiles = get_unique_values(conn, 'forecasts', 'quantile')
        horizons = get_unique_values(conn, 'forecasts', 'horizon')
        versions = get_unique_values(conn, 'forecasts', 'version')
        countries = get_unique_values(conn, 'forecasts', 'country')
        
        print(f"Quantiles: {quantiles}")
        print(f"Horizons: {horizons}")
        print(f"Versions: {versions}")
        print(f"Countries ({len(countries)}): {countries}")
        print()
        
        # Generate expected dates (month start dates from 2016-02-01 to 2025-05-01)
        expected_dates = generate_expected_dates('2016-02-01', '2025-05-01')
        print(f"Expected date range: {expected_dates[0]} to {expected_dates[-1]}")
        print(f"Total expected dates per combination: {len(expected_dates)}")
        print()
        
        # Check for missing dates per quantile, horizon, version combination
        print("=== Missing Dates Analysis ===")
        
        missing_report = []
        
        for quantile in quantiles:
            for horizon in horizons:
                for version in versions:
                    print(f"\nAnalyzing: Quantile={quantile}, Horizon={horizon}, Version={version}")
                    
                    # Get existing window_end dates for this combination across all countries
                    query = """
                    SELECT DISTINCT window_end, COUNT(DISTINCT country) as country_count
                    FROM forecasts 
                    WHERE quantile = ? AND horizon = ? AND version = ?
                    GROUP BY window_end
                    ORDER BY window_end
                    """
                    
                    cursor = conn.cursor()
                    cursor.execute(query, (quantile, horizon, version))
                    existing_dates = cursor.fetchall()
                    
                    # Convert to dictionary for easier lookup
                    existing_date_counts = {date: count for date, count in existing_dates}
                    
                    missing_dates = []
                    incomplete_dates = []
                    
                    for expected_date in expected_dates:
                        if expected_date not in existing_date_counts:
                            missing_dates.append(expected_date)
                        elif existing_date_counts[expected_date] < len(countries):
                            # Date exists but not for all countries
                            incomplete_dates.append((expected_date, existing_date_counts[expected_date]))
                    
                    # Report results
                    if missing_dates:
                        print(f"  Missing dates ({len(missing_dates)}): {missing_dates[:5]}{'...' if len(missing_dates) > 5 else ''}")
                        missing_report.append({
                            'quantile': quantile,
                            'horizon': horizon,
                            'version': version,
                            'missing_dates': missing_dates,
                            'missing_count': len(missing_dates)
                        })
                    
                    if incomplete_dates:
                        print(f"  Incomplete dates ({len(incomplete_dates)}): {incomplete_dates[:3]}{'...' if len(incomplete_dates) > 3 else ''}")
                        for date, count in incomplete_dates[:3]:
                            print(f"    {date}: {count}/{len(countries)} countries")
                    
                    if not missing_dates and not incomplete_dates:
                        print(f"  ✓ Complete: All {len(expected_dates)} dates present for all {len(countries)} countries")
        
        # Summary report
        print("\n=== SUMMARY REPORT ===")
        if missing_report:
            print("Combinations with missing dates:")
            for item in missing_report:
                print(f"  Quantile {item['quantile']}, Horizon {item['horizon']}, Version {item['version']}: {item['missing_count']} missing dates")
        else:
            print("✓ No missing dates found for any combination!")
        
        # Save detailed report to file
        if missing_report:
            with open('/home/skooiker/ml_macro_at_risk/missing_dates_detailed_report.txt', 'w') as f:
                f.write("DETAILED MISSING DATES REPORT\n")
                f.write("=" * 50 + "\n\n")
                
                for item in missing_report:
                    f.write(f"Quantile: {item['quantile']}, Horizon: {item['horizon']}, Version: {item['version']}\n")
                    f.write(f"Missing dates ({item['missing_count']}):\n")
                    for date in item['missing_dates']:
                        f.write(f"  {date}\n")
                    f.write("\n")
            
            print(f"\nDetailed report saved to: missing_dates_detailed_report.txt")
    
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    db_path = "/home/skooiker/ml_macro_at_risk/outputs/task_coordination.db"
    check_missing_dates(db_path)
