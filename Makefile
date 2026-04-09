.ONESHELL:

PROJECT?=psd-paths
VERSION?=3.13
VENV=${PROJECT}-${VERSION}


sync:
	@echo "Syncing source code to uv workspace"
	uv sync --upgrade --all-extras


test:
	@echo "Running tests with uv"
	uv run pytest tests/

kernel:
	@echo "Installing Jupyter kernel"
	# Use the uv‑managed Python interpreter to register the kernel
	uv run python -m ipykernel install \
	    --user \
	    --name=${VENV} \
	    --display-name=${VENV}



dashboard:
	@echo "Generating HTML pipeline status dashboard"
	uv run python templates/generate_dashboard.py

status:
	@echo "Pipeline step status"
	uv run python -c "import sys; sys.path.insert(0,'templates'); from run_pipeline import show_status; show_status()"

context-py:
	files-to-prompt . -e py -e md  -e toml  --ignore  ./_archive/  ./.venv/ --cxml -o py-context.txt 