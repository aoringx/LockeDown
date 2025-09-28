import uvicorn
from fastapi import FastAPI

app = FastAPI()

@app.get("/data")
def get_data():
    return "you are stupid"

uvicorn.run(app, host="0.0.0.0", port = 4999)