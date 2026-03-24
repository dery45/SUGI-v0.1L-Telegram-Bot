import subprocess
import sys
import time
from pathlib import Path

# Script to launch all Sugi Background Services + Telegram Bot
# Run with: `python start_all.py`

def run_service(name: str, cmd: list[str]) -> subprocess.Popen:
    print(f"🚀 Starting {name}...")
    try:
        # Use Popen to run non-blocking
        return subprocess.Popen(cmd)
    except Exception as e:
        print(f"❌ Failed to start {name}: {e}")
        return None

def main():
    print("==================================================")
    print("          Sugi v0.1L Master Startup Script        ")
    print("==================================================")

    processes = []

    # 1. Start Vector CSV/XLSX Service
    processes.append(run_service("Vector CSV Watcher", [sys.executable, str(Path("services/vectorCSV.py"))]))
    
    # 2. Start Vector PDF Service
    processes.append(run_service("Vector PDF Watcher", [sys.executable, str(Path("services/vectorpdf.py"))]))
    
    # 3. Start Weather Service
    processes.append(run_service("Weather Insight Service", [sys.executable, str(Path("services/vectorWeather.py"))]))
    
    # 4. Start Daily Insight (MongoDB Async Pushing)
    processes.append(run_service("Daily Insight Cron", [sys.executable, str(Path("services/daily_insight.py"))]))

    # 5. Start Telegram Bot
    # Note: If TELEGRAM_BOT_TOKEN is blank, this script will terminate safely but log an error.
    processes.append(run_service("Telegram Bot", [sys.executable, str(Path("interfaces/telegram/telegram_bot.py"))]))

    print("\n✅ All background services initiated! (Press Ctrl+C to stop all)")
    print("   Note: Ensure 'chroma run --path data/db --port 8000' is running in a separate terminal!")
    print("==================================================\n")

    try:
        # Keep master script alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down all Sugi services...")
        for p in processes:
            if p:
                p.terminate()
        for p in processes:
            if p:
                p.wait()
        print("✅ Shutdown complete.")

if __name__ == "__main__":
    main()
