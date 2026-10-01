.ONESHELL:

PROJECT?=psd-paths
VERSION?=3.13
VENV=${PROJECT}-${VERSION}


sync:
	@echo "Installing the exact versions pinned in uv.lock"
	uv sync --locked --all-extras

# After editing pyproject.toml (e.g. a new eeg-spectral tag): re-lock only what
# changed, keep every other pin, then install.
update:
	uv lock
	uv sync --locked --all-extras

# Move every package to its newest allowed version. Results can change: rerun
# the pipeline afterwards.
upgrade:
	uv lock --upgrade
	uv sync --locked --all-extras


test:
	@echo "Running tests with uv"
	uv run --locked pytest tests/

kernel:
	@echo "Installing Jupyter kernel"
	# Use the uv‑managed Python interpreter to register the kernel
	uv run --locked python -m ipykernel install \
	    --user \
	    --name=${VENV} \
	    --display-name=${VENV}



dashboard:
	@echo "Generating HTML pipeline status dashboard"
	uv run --locked python templates/generate_dashboard.py

status:
	@echo "Pipeline step status"
	uv run --locked python -c "import sys; sys.path.insert(0,'templates'); from run_pipeline import cfg, show_status; show_status(cfg)"

context-py:
	files-to-prompt . -e py -e md  -e toml  --ignore  ./_archive/  ./.venv/ --cxml -o py-context.txt 