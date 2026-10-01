.ONESHELL:
.DEFAULT_GOAL := help

PROJECT?=psd-paths
VERSION?=3.13
VENV=${PROJECT}-${VERSION}

# `make` on its own lists the commands below (the text after ##).
help: ## List the commands
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  make %-10s %s\n", $$1, $$2}'

sync: ## Install the exact package versions in uv.lock (first setup, after git pull)
	uv sync --locked --all-extras

update: ## Re-lock after editing pyproject.toml (e.g. a new eeg-spectral tag), then install
	uv lock
	uv sync --locked --all-extras

upgrade: ## Move all packages to their newest allowed versions (results can change)
	uv lock --upgrade
	uv sync --locked --all-extras

kernel: ## Register the Jupyter kernel the pipeline uses (once per machine)
	uv run --locked python -m ipykernel install \
	    --user \
	    --name=${VENV} \
	    --display-name=${VENV}

run: ## Run all subjects (steps 01-05), then the group step, into the run folder in settings.toml
	uv run --locked python templates/run_pipeline.py

group: ## Rebuild the group tables of the run in settings.toml (after a partial rerun)
	uv run --locked python -c "import sys; sys.path.insert(0,'templates'); from run_pipeline import cfg, run_group; print(run_group(cfg))"

status: ## Show which steps succeeded or failed per subject
	uv run --locked python -c "import sys; sys.path.insert(0,'templates'); from run_pipeline import cfg, show_status; show_status(cfg)"

dashboard: ## Write the HTML status dashboard of the run
	uv run --locked python templates/generate_dashboard.py

context-py:
	files-to-prompt . -e py -e md  -e toml  --ignore  ./_archive/  ./.venv/ --cxml -o py-context.txt
