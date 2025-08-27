#!/usr/bin/env python3
"""
Progress monitoring script for parallel forecasting
"""

import pandas as pd
import time
from pathlib import Path
from datetime import datetime
import argparse

def format_status_emoji(status):
    """Convert status to emoji"""
    status_map = {
        'pending': '⏳',
        'training': '🔄', 
        'done': '✅',
        'failed': '❌'
    }
    return status_map.get(status, '❓')

def display_progress(progress_file):
    """Display current progress"""
    
    if not progress_file.exists():
        print("📊 Progress file not found. No tasks started yet.")
        return
    
    try:
        df = pd.read_parquet(progress_file)
        
        if len(df) == 0:
            print("📊 Progress file is empty. No tasks recorded yet.")
            return
        
        print("\n" + "="*80)
        print(f"📊 PARALLEL FORECASTING PROGRESS - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*80)
        
        # Overall summary
        status_counts = df['STATUS'].value_counts()
        total_tasks = len(df)
        
        print(f"\n📈 OVERALL SUMMARY ({total_tasks} total tasks)")
        print("-" * 50)
        for status, count in status_counts.items():
            percentage = (count / total_tasks) * 100
            emoji = format_status_emoji(status)
            print(f"{emoji} {status.title():<10}: {count:>5} ({percentage:>5.1f}%)")
        
        # Progress by model
        print(f"\n🤖 PROGRESS BY MODEL")
        print("-" * 50)
        model_summary = df.groupby(['MODEL', 'STATUS']).size().unstack(fill_value=0)
        
        for model in model_summary.index:
            print(f"\n{model.upper()}:")
            for status in ['pending', 'training', 'done', 'failed']:
                if status in model_summary.columns:
                    count = model_summary.loc[model, status]
                    if count > 0:
                        emoji = format_status_emoji(status)
                        print(f"  {emoji} {status}: {count}")
        
        # Progress by country
        print(f"\n🌍 PROGRESS BY COUNTRY")
        print("-" * 50)
        country_summary = df.groupby(['COUNTRY', 'STATUS']).size().unstack(fill_value=0)
        
        # Show top 10 countries by total tasks
        country_totals = country_summary.sum(axis=1).sort_values(ascending=False)
        
        for country in country_totals.head(10).index:
            done_count = country_summary.loc[country].get('done', 0)
            total_count = country_summary.loc[country].sum()
            percentage = (done_count / total_count) * 100 if total_count > 0 else 0
            
            print(f"{country:<4}: {done_count:>3}/{total_count:<3} ({percentage:>5.1f}%)")
        
        if len(country_totals) > 10:
            print(f"... and {len(country_totals) - 10} more countries")
        
        # Recent activity
        print(f"\n⏰ RECENT ACTIVITY (last 10 updates)")
        print("-" * 50)
        
        # Sort by last update and show recent ones
        df_sorted = df.sort_values('LAST_UPDATE', ascending=False)
        recent_df = df_sorted.head(10)
        
        for _, row in recent_df.iterrows():
            emoji = format_status_emoji(row['STATUS'])
            update_time = pd.to_datetime(row['LAST_UPDATE']).strftime('%H:%M:%S')
            print(f"{emoji} {update_time} | {row['MODEL']:<10} | {row['COUNTRY']:<4} | Q{row['QUANTILE']:<4} H{row['HORIZON']:<2} | {row['STATUS']}")
        
        # Failed tasks
        failed_df = df[df['STATUS'] == 'failed']
        if len(failed_df) > 0:
            print(f"\n❌ FAILED TASKS ({len(failed_df)} tasks)")
            print("-" * 50)
            
            for _, row in failed_df.head(5).iterrows():
                error_msg = row['ERROR_MSG']
                if pd.notna(error_msg):
                    # Show first line of error
                    first_line = error_msg.split('\n')[0][:60]
                    print(f"• {row['MODEL']} | {row['COUNTRY']} | Q{row['QUANTILE']} H{row['HORIZON']}: {first_line}")
            
            if len(failed_df) > 5:
                print(f"... and {len(failed_df) - 5} more failed tasks")
        
        # Estimated completion
        done_tasks = status_counts.get('done', 0)
        training_tasks = status_counts.get('training', 0)
        pending_tasks = status_counts.get('pending', 0)
        
        if done_tasks > 0 and (training_tasks + pending_tasks) > 0:
            print(f"\n⏱️  ESTIMATED COMPLETION")
            print("-" * 50)
            
            # Simple linear estimate based on completion rate
            remaining = training_tasks + pending_tasks
            completion_rate = done_tasks / total_tasks
            
            if completion_rate > 0:
                # Very rough estimate
                print(f"Remaining tasks: {remaining}")
                print(f"Completion rate: {completion_rate:.2%}")
                print("(Note: This is a rough estimate and may not be accurate)")
        
        print("\n" + "="*80)
        
    except Exception as e:
        print(f"❌ Error reading progress file: {e}")

def main():
    """Main monitoring function"""
    
    parser = argparse.ArgumentParser(description="Monitor parallel forecasting progress")
    parser.add_argument("--output-dir", type=str, default="parallel_outputs", 
                       help="Output directory containing progress files")
    parser.add_argument("--watch", action="store_true", 
                       help="Watch mode - refresh every 30 seconds")
    parser.add_argument("--refresh", type=int, default=30, 
                       help="Refresh interval in seconds for watch mode")
    
    args = parser.parse_args()
    
    progress_file = Path(args.output_dir) / "progress" / "progress.parquet"
    
    if args.watch:
        print("🔍 Starting watch mode. Press Ctrl+C to exit.")
        print(f"Refreshing every {args.refresh} seconds...")
        
        try:
            while True:
                display_progress(progress_file)
                time.sleep(args.refresh)
        except KeyboardInterrupt:
            print("\n\n👋 Monitoring stopped.")
    else:
        display_progress(progress_file)

if __name__ == "__main__":
    main()
