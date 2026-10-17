.PHONY: install train test demo dashboard lint clean

install:
	python -m venv .venv
	.venv\Scripts\Activate.ps1
	pip install -r requirements.txt
	pip install -e .

test:
	.\.venv\Scripts\python.exe -m pytest -q

demo:
	$$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m hybrid_ai.dashboard.demo --smoke

dashboard:
	.\.venv\Scripts\streamlit.exe run src\hybrid_ai\dashboard\app.py

web:
	$$env:PYTHONPATH="src"; .\.venv\Scripts\python.exe -m uvicorn hybrid_ai.web.api:app --port 8502 --app-dir src

clean:
	.\.venv\Scripts\python.exe -c "import pathlib,shutil; [shutil.rmtree(p,ignore_errors=True) for p in list(pathlib.Path('.').rglob('__pycache__'))]; shutil.rmtree('.pytest_cache',ignore_errors=True); shutil.rmtree('demo_outputs',ignore_errors=True)"
