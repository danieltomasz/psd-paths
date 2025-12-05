# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an EEG analysis pipeline for processing resting-state EEG data and extracting Power Spectral Density (PSD) features. The project uses high-density EEG recordings (256-channel GSN-HydroCel montage) and processes them through filtering, artifact removal, ICA, and spectral parameterization using the SpecParam library.

## Environment Setup

**Python Version**: 3.13

**Package Manager**: This project uses `uv` for dependency management with a workspace structure.

**Installation**:

```bash
# Install dependencies using uv
uv sync

# Activate virtual environment
source .venv/bin/activate
```

**Conda Alternative** (if preferred):

```bash
# For Apple Silicon
conda env create -n conda-paths-3.12 -f local.yml

# For Intel processors
conda env create -n conda-paths-3.12 -f local_noARM.yml
```

## Key Commands

### Development

```bash
# Format code
black .
ruff format

# Lint code
ruff check

# Run tests
pytest src/spectral/tests/

# Run specific test
pytest src/spectral/tests/test_specparam.py -v
```
