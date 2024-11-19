.PHONY: install run clean

# Create virtual environment and install dependencies
install:
	python -m venv .venv
	. .venv/bin/activate && pip install -r requirements.txt

# Run the Streamlit application
run:
	. .venv/bin/activate && streamlit run main.py

# Clean up generated files and virtual environment
clean:
	rm -rf .venv
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete