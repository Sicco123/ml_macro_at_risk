#!/usr/bin/env python3
"""
Quick progress monitoring script
"""
import pandas as pd
import sys
from pathlib import Path

def main():
    progress_path = Path("outputs/progress/progress.parquet")
    
    if not progress_path.exists():
        print("No progress file found")
        return
    
    try:
        df = pd.read_parquet(progress_path)
        
        print("=== PROGRESS SUMMARY ===")
        print(f"Total tasks tracked: {len(df)}")
        
        status_counts = df['STATUS'].value_counts()
        for status, count in status_counts.items():
            print(f"{status}: {count}")
        
        if 'done' in status_counts and 'claimed' in status_counts:
            total_tracked = len(df)
            completed = status_counts.get('done', 0)
            claimed = status_counts.get('claimed', 0)
            failed = status_counts.get('failed', 0)
            
            if total_tracked > 0:
                completed_pct = (completed / total_tracked) * 100
                print(f"\nCompletion: {completed_pct:.1f}% ({completed}/{total_tracked})")
        
        # Show active workers
        if 'INSTANCE_ID' in df.columns:
            active_workers = df[df['STATUS'] == 'claimed']['INSTANCE_ID'].nunique()
            print(f"Active workers: {active_workers}")
            
            if active_workers > 0:
                print("\nActive worker instances:")
                for worker in df[df['STATUS'] == 'claimed']['INSTANCE_ID'].unique():
                    worker_tasks = len(df[df['INSTANCE_ID'] == worker])
                    print(f"  {worker}: {worker_tasks} tasks")
        
        # Show recent activity
        print(f"\nMost recent updates:")
        recent = df.nlargest(5, 'LAST_UPDATE')[['STATUS', 'COUNTRY', 'WINDOW_START', 'LAST_UPDATE', 'INSTANCE_ID']]
        for _, row in recent.iterrows():
            print(f"  {row['STATUS']}: {row['COUNTRY']} {row['WINDOW_START']} by {row['INSTANCE_ID']} at {row['LAST_UPDATE']}")
            
    except Exception as e:
        print(f"Error reading progress: {e}")

if __name__ == "__main__":
    main()
