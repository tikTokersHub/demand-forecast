.PHONY: install ingest features validate train pipeline serve dashboard test lint format ci

install:
	uv pip install -e .

install-dl:
	uv pip install torch --index-url https://download.pytorch.org/whl/cu121
	uv pip install pytorch-forecasting lightning

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

train-tft:
	python src/demand_forecast/tft_model.py

tensorboard:
	tensorboard --logdir tb_logs --port 6006