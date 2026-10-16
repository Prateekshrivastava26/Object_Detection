# core/logger.py

import datetime

LOG_FILE = "logs/exam_log.txt"

def log_event(message):
    with open(LOG_FILE, "a") as f:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")