.PHONY: install run clean

install:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

run:
	. .venv/bin/activate && streamlit run main.py

format:
	ruff format .

fix:
	ruff check --fix .

lint:
	ruff check .

clean: format fix lint
	rm -rf .pytest_cache */__pycache__ */*/__pycache__ */*/*/__pycache__
	rm -rf .pytest_cache */*/*/*/__pycache__ */*/*/*/*/__pycache__
	ruff clean