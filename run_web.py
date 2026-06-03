"""Launch the slipguard web UI — the drag-and-drop receipt validity check.

PyCharm:  right-click this file -> Run 'run_web', then open http://127.0.0.1:8000 in your browser
          and drag a receipt (image or PDF) onto the page.
Terminal: python run_web.py        (equivalent to `slipguard serve`)

Requirements:
  * web deps:  pip install -e ".[web]"      (FastAPI + uvicorn)
  * a model behind it: an API key in .env (GROQ_API_KEY / GEMINI_API_KEY) OR a local LM Studio
    vision model. The verdict is produced by whichever provider is configured (default: auto).
"""

import pathlib
import sys

# Make `import slipguard` work whether or not the package is pip-installed in this interpreter
# (so this file runs from PyCharm regardless of how the interpreter is set up).
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "src"))

HOST, PORT = "127.0.0.1", 8000

if __name__ == "__main__":
    try:
        import uvicorn
    except ImportError:
        sys.exit('The web UI needs FastAPI + uvicorn. In PyCharm\'s terminal run:\n'
                 '    pip install -e ".[web]"')
    print(f"slipguard web UI  ->  http://{HOST}:{PORT}   (Ctrl+C / stop button to quit)")
    print("Open that URL, then drag a receipt (image or PDF) onto the page.")
    # Import string (not the app object) keeps the frontend/dist + .env resolution identical to
    # `slipguard serve`; reload is off so it runs in this one process.
    uvicorn.run("slipguard.web.api:app", host=HOST, port=PORT, reload=False)
