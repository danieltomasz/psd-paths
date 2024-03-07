.ONESHELL:
.PHONY: install

PROJECT?=paths
VERSION?=3.11.8

VENV=${PROJECT}-${VERSION}
VENV_DIR=$(shell pyenv root)/versions/${VENV}
PYTHON=${VENV_DIR}/bin/python
JUPYTER_ENV_NAME=${VENV}
VARS = PYDEVD_DISABLE_FILE_VALIDATION=1

CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate

pip-install:
	@echo "Installing $(VENV)"
	env PYTHON_CONFIGURE_OPTS=--enable-shared pyenv virtualenv ${VERSION} ${VENV}
	pyenv local ${VENV}
	$(PYTHON) -m pip  install -U pip
	$(PYTHON) -m pip install  -r requirements.txt
	PYDEVD_DISABLE_FILE_VALIDATION=1  $(PYTHON) -m ipykernel install --user --name ${VENV}

conda-install:
	$(eval PYTHON_DIST := miniforge3-latest)
	$(eval VENV := conda-paths-3.12)
	$(eval CONDA_BIN := ~/.pyenv/versions/${PYTHON_DIST}/bin/conda)
	$(eval PYTHON := ~/.pyenv/versions/${VENV}/bin/python)
	@echo "Installing $(VENV) with $(PYTHON_DIST)"
	env PYTHON_CONFIGURE_OPTS=--enable-shared pyenv install --skip-existing ${PYTHON_DIST}
	${CONDA_BIN} update -n base -c conda-forge conda
	pyenv uninstall ${VENV} || true
	pyenv virtualenv  ${PYTHON_DIST} ${VENV} || true;
	${CONDA_BIN}  env update --name ${VENV}  --file local.yml --prune -v
	@eval "$$(pyenv init -)" && \
	pyenv activate ${VENV}; \
	pyenv local ${VENV}; \
	PYDEVD_DISABLE_FILE_VALIDATION=1 ${PYTHON} -m ipykernel install --user --name ${VENV}

update:
	$(PYTHON) -m pip install --upgrade -r requirements.txt --upgrade-strategy=eager

download:
	$(PYTHON) -m pip download -r requirements.txt -d downloads

local:
	cd src/mnelab && $(PYTHON) -m pip install -e . & cd ../..