.PHONY: install train test demo dashboard lint clean

install:
	python -m venv .venv
	.venv\Scripts\Activate.ps1
	pip install -r requirements.txt
	pip install -e .

test:
	pytest

demo:
	python -m hybrid_ai.dashboard.demo

dashboard:
	streamlit run src/hybrid_ai/dashboard/app.py

web:
	cd src && ..\.venv\Scripts\python.exe -m uvicorn hybrid_ai.web.api:app --port 8502

clean:
	for /d /r . %%d in (__pycache__) do @if exist "%%d" rd /s /q "%%d"
	if exist .pytest_cache rd /s /q .pytest_cache
	if exist data\pareto_front.json del /q data\pareto_front.json
	if exist demo_outputs rd /s /q demo_outputs
