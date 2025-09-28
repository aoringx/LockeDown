# backend.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from locked_down import current_session_data, monitor_loop, session_history
import threading

app = FastAPI()

# Allow browser to fetch from frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/session")
def get_session():
    return {
        "current": current_session_data,
        "history": session_history,
    }

@app.get("/session/current")
def get_current_session():
    return current_session_data

@app.get("/session/history")
def get_session_history():
    return session_history

# Remove static file serving to make this a pure API backend
# app.mount("/", StaticFiles(directory="static", html=True), name="static")

thread = threading.Thread(target=monitor_loop, daemon=True)
thread.start()