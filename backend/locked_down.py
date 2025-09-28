# locked_down.py
# A desktop productivity monitor that uses the Gemini API to classify app/browser usage.

import arrow
import pywinctl as pwc
import pyautogui
import pyperclip
import time
from google import genai
from google.genai import types
import json
import os
import sys
from category_db import CategoryDB
from pathlib import Path
import subprocess
from ranking import Ranking
from bots import Bots

os.environ["GEMINI_API_KEY"] = "--"

# --- GLOBAL STATE ---
current_session_data = {
    "is_active": False,
    "last_data": "",
    "session_start_time": None,
    "total_entertainment_time": 0.0,
    "category": "NEUTRAL",
    "reason": "",
    "points": 100.0,  # Initial points
}

# Points per minute by category
POINT_RATES = {
    "PRODUCTIVITY": 5.0,
    "ENTERTAINMENT": -5.0,
    "NEUTRAL": 0.0,
}

# New: list to keep a cumulative history of events
session_history = []  # each entry: {'timestamp', 'category', 'reason', 'data', 'total_time_min'}

# --- 1. MONITORING AND DATA GATHERING (The Scout) ---
def _debug_log(msg: str):
    """Lightweight debug logger gated by env var LOCKED_DOWN_DEBUG."""
    try:
        if os.getenv("LOCKED_DOWN_DEBUG"):
            print(f"[locked_down][debug] {msg}")
    except Exception:
        pass


def _proc_name_from_pid(pid: int) -> str | None:
    try:
        # Prefer /proc/<pid>/comm
        comm_path = Path(f"/proc/{pid}/comm")
        if comm_path.exists():
            name = comm_path.read_text(errors="ignore").strip()
            if name:
                return name.lower()
        # Fallback to exe symlink
        exe_path = Path(f"/proc/{pid}/exe")
        if exe_path.exists():
            try:
                target = os.readlink(exe_path)
                if target:
                    return os.path.basename(target).lower()
            except OSError:
                pass
        # Fallback to first token of cmdline
        cmdline_path = Path(f"/proc/{pid}/cmdline")
        if cmdline_path.exists():
            raw = cmdline_path.read_bytes()
            if raw:
                first = raw.split(b"\x00", 1)[0].decode(errors="ignore")
                return os.path.basename(first).lower()
    except Exception:
        return None
    return None

def get_active_app_data():
    try:
        cmd = ['gdbus', 'call', '--session', '--dest', 'org.gnome.Shell',
               '--object-path', '/org/gnome/Shell/Extensions/SystemStatus',
               '--method', 'org.gnome.Shell.Extensions.SystemStatus.GetActiveApplication']
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        output = result.stdout.strip()
        
        title_start = output.find("'title': '") + len("'title': '")
        if title_start != -1:
            title_end = output.find("'", title_start)
            window_title = output[title_start:title_end]
            return "firefox", window_title
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass 

    try:
        active_window = pwc.getActiveWindow()
        if active_window:
            app_name = active_window.getAppName().lower()
            window_title = active_window.title or "Untitled"
            return app_name, window_title
    except Exception as e:
        _debug_log(f"PyWinCtl error: {e}")
        
    return "unknown", "Unknown Window"


def classify_usage(input_text, db: CategoryDB = None):

    if "Title: " in input_text and len(input_text.split("Title: ")[1].strip()) < 3:
        return {"category": "NEUTRAL", "reason": "Window title was empty or unreadable; defaulted to neutral."}
    
    category = db.get_category(input_text)
    if category is not None:
        return {"category": category[1], "reason": "Category retrieved from local database."}

    if os.getenv("GEMINI_API_KEY") is None:
        return {"category": "ERROR", "reason": "GEMINI_API_KEY environment variable not set."}
    

    response_schema = {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": ["ENTERTAINMENT", "PRODUCTIVITY", "NEUTRAL"]},
            "reason": {"type": "string", "description": "Brief justification based on app or title keywords."}
        },
        "required": ["category", "reason"]
    }

    # 3. Define the prompt (Optimized to be more strict and decisive)
    prompt = f"""
    CONTEXT: You are a strict productivity monitor. Analyze the window the user has open. 
    If the name of the window is strongly associated with streaming, gaming or shows (e.g., 'Netflix', 'YouTube Gameplay', 'Twitch', 'Steam', 'Gameplay'), classify it as ENTERTAINMENT.
    If the window is used for productivity, work or study (e.g. 'GitHub', 'VS Code', 'Google Docs', "YouTube Physics Review" or 'Documentation', classify it as PRODUCTIVITY.

    INPUT DATA: "{input_text}"
    """

    try:
        client = genai.Client()
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=response_schema
            )
        )
        
        if response.text:
            db.add_category(input_text, json.loads(response.text).get("category", "NEUTRAL"))
            return json.loads(response.text)
        else:
            return {"category": "NEUTRAL", "reason": "LLM returned an empty response."}
            
    except Exception:
        return {"category": "ERROR", "reason": "API/Network failure."}

# --- 3. APPLICATION LOGIC (The Session Manager) ---

def monitor_loop():
    global current_session_data, session_history
    CHECK_INTERVAL = 5

    db = CategoryDB()
    rank_db = Ranking()

    user_name = "Me"
    if rank_db.get_score(user_name) != None:
        current_session_data["points"] = rank_db.get_score(user_name)
    else:
        rank_db.add_player(user_name,current_session_data["points"])

    bots = Bots(rank_db,
                {"bot1":0,
                 "bot2":0.5,
                 "bot3":0.75})

    while True:
        app_name, data_string = get_active_app_data()
        # Use structured string for classification and deduping so app changes are captured
        structured_input = data_string
        # f"App: {app_name} | Title: {data_string}"
        if structured_input == current_session_data["last_data"]:
            time.sleep(CHECK_INTERVAL)
            continue
        
        classification_result = classify_usage(structured_input, db=db)
        new_category = classification_result.get("category", "NEUTRAL")


        elapsed_time = 0
        if current_session_data["last_data"]:
            # Calculate elapsed time since last check
            elapsed_time = (time.time() - (current_session_data.get("last_check_time") or time.time())) / 60
        
        # Update current session info
        if new_category == "ENTERTAINMENT" and not current_session_data["is_active"]:
            current_session_data["is_active"] = True
            current_session_data["session_start_time"] = time.time()
        elif new_category != "ENTERTAINMENT" and current_session_data["is_active"]:
            # End entertainment session
            start_time = current_session_data["session_start_time"] or time.time()
            duration = time.time() - start_time
            current_session_data["total_entertainment_time"] += duration
            current_session_data["is_active"] = False
            current_session_data["session_start_time"] = None

        # Always update last data and classification
        current_session_data["last_data"] = structured_input
        current_session_data["category"] = new_category
        current_session_data["reason"] = classification_result.get("reason", "")

        # Update points based on last category
        rate = POINT_RATES.get(current_session_data["category"], 0)
        point_change = elapsed_time * rate
        current_session_data["points"] += point_change
        current_session_data["last_check_time"] = time.time()

        # Calculate live total time
        current_duration = (time.time() - current_session_data["session_start_time"]) if current_session_data["is_active"] else 0
        current_total_time = current_session_data["total_entertainment_time"] + current_duration

        # --- Append to history only if there’s a meaningful change ---
        last_event = session_history[-1] if session_history else None
        if (not last_event
            or last_event["data"] != structured_input
            or last_event["category"] != new_category):
            event = {
                "timestamp": arrow.now().isoformat(),
                "category": new_category,
                "reason": classification_result.get("reason", ""),
                "data": structured_input,
                "total_time_min": current_total_time / 60,
                "points": current_session_data["points"],  # <-- NEW
            }
            session_history.append(event)
        bots.change_bots()
        rank_db.set_score(user_name,current_session_data["points"])
        time.sleep(CHECK_INTERVAL)

# --- EXECUTION ---
if __name__ == '__main__':
    monitor_loop()
