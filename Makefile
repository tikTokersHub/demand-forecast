.PHONY: install ingest features validate train pipeline serve dashboard test lint format ci

install:
	uv pip install -e .

ingest:
	python src/scripts/run_ingest.py

features:
	python src/scripts/run_features.py

validate:
	python src/scripts/run_validate.py

train:
	python src/scripts/run_train.py

pipeline: ingest features validate train

serve:
	uvicorn src.demand_forecast.api:app --reload --port 8000

dashboard:
	streamlit run src/demand_forecast/ui.py --server.port 8501

test:
	pytest tests/ -v

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

ci: lint test