"""
Entry point wrapper for uvicorn execution.
Run server: uvicorn main:app --reload --port 8000
"""
from app.main import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
