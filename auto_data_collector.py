"""
Automated Data Collection Script
Collects weather data every 2 hours
Run this and let it collect data automatically
"""

import time
from datetime import datetime
from data_collector import WeatherDataCollector
from config import Config

def auto_collect_data(interval_hours=2, total_collections=100):
    """
    Automatically collect weather data at regular intervals
    
    Args:
        interval_hours: Hours between each collection (default: 2)
        total_collections: Total number of times to collect data (default: 100)
    """
    Config.init_app()
    collector = WeatherDataCollector()
    
    print("=" * 60)
    print("🤖 AUTOMATED DATA COLLECTION STARTED")
    print("=" * 60)
    print(f"⏰ Collecting data every {interval_hours} hours")
    print(f"🎯 Target: {total_collections} collections")
    print(f"⏱️ Total time: ~{interval_hours * total_collections} hours")
    print("=" * 60)
    print("\n💡 Tip: Keep this running in background!")
    print("💡 Press Ctrl+C to stop anytime\n")
    
    interval_seconds = interval_hours * 3600  # Convert hours to seconds
    
    for i in range(total_collections):
        try:
            print(f"\n{'='*60}")
            print(f"📊 Collection {i+1}/{total_collections}")
            print(f"🕐 Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"{'='*60}\n")
            
            # Collect data
            df = collector.collect_and_save()
            
            if df is not None:
                print(f"\n✅ Collection {i+1} completed!")
                print(f"📈 Total records in dataset: {len(df)}")
            else:
                print(f"\n⚠️ Collection {i+1} failed - will retry next time")
            
            # Wait before next collection (unless it's the last one)
            if i < total_collections - 1:
                print(f"\n😴 Sleeping for {interval_hours} hours...")
                print(f"⏰ Next collection at: {datetime.fromtimestamp(time.time() + interval_seconds).strftime('%Y-%m-%d %H:%M:%S')}")
                time.sleep(interval_seconds)
            
        except KeyboardInterrupt:
            print("\n\n🛑 Data collection stopped by user")
            print(f"✅ Completed {i+1} collections")
            break
        except Exception as e:
            print(f"\n❌ Error in collection {i+1}: {e}")
            print("⏳ Will retry in next cycle...")
            if i < total_collections - 1:
                time.sleep(interval_seconds)
    
    print("\n" + "=" * 60)
    print("🎉 AUTOMATED COLLECTION COMPLETED!")
    print("=" * 60)
    print("\n📁 Data saved in: data/weather_data.csv")
    print("🚀 Now you can train your ML models!")

if __name__ == "__main__":
    # Collect data every 2 hours, 50 times (100 hours = ~4 days)
    auto_collect_data(interval_hours=2, total_collections=50)
    
    # For faster testing (every 10 minutes):
    # auto_collect_data(interval_hours=0.16, total_collections=20)  # 10 minutes
    
    # For overnight collection (every 30 minutes):
    # auto_collect_data(interval_hours=0.5, total_collections=40)  # 20 hours