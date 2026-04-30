.PHONY: test run report run-all

test:
	pytest tests/ -v

run:
	python -m scripts.run_pipeline

report:
	python -m scripts.06_generate_report

run-all:
	python scripts/01_download_data.py
	python scripts/02_build_dataset.py
	python scripts/03_train_model.py
	python scripts/04_generate_predictions.py
	python scripts/05_evaluate.py
