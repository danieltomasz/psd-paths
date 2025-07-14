.ONESHELL:

PROJECT?=psd-paths
VERSION?=3.13
VENV=${PROJECT}-${VERSION}
VENV_DIR=$(shell pyenv root)/versions/${VENV}
PYTHON=${VENV_DIR}/bin/python
VARS = PYDEVD_DISABLE_FILE_VALIDATION=1
SHELL = /bin/bash
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate


install:
	$(eval PYTHON := ~/.pyenv/versions/${VENV}/bin/python)
	$(eval PYTHON_DIST := miniforge3-25.1.1-2)
	$(eval CONDA_BIN := ~/.pyenv/versions/${PYTHON_DIST}/bin/conda)
	$(eval VENV := ${VENV})
	@echo "Installing $(VENV) with $(PYTHON_DIST)"
	env PYTHON_CONFIGURE_OPTS=--enable-shared pyenv install --skip-existing ${PYTHON_DIST}
	${CONDA_BIN} update -n base -c conda-forge conda
	pyenv uninstall ${VENV} || true
	${CONDA_BIN}  config --set verbosity 2 
	${CONDA_BIN} env update -n ${VENV} -f local.yml 
	$(eval CONDA_ENV_PATH := $(shell ${CONDA_BIN} env list | grep '${VENV}' | awk '{print $$2}'))
	@echo "Linking Conda environment to pyenv"
	$(eval PYENV_ROOT := $(shell pyenv root))
	ln -sfn  ~/.pyenv/versions/${PYTHON_DIST}/envs/${VENV} ${PYENV_ROOT}/versions/${VENV}
#	${PYTHON}   -m pip install  -U  -r requirements.txt
	@eval "$$(pyenv init -)" && \
	pyenv activate ${VENV}; \
	pyenv local ${VENV}; \
	PYDEVD_DISABLE_FILE_VALIDATION=1 ${PYTHON} -m ipykernel install --user --name ${VENV}

python-info:
	echo ${VIRTUAL_ENV}


clean-conda:
	$(eval PYTHON_DIST :=  miniforge3-25.1.1-0)
	$(eval CONDA_BIN := ~/.pyenv/versions/${PYTHON_DIST}/bin/conda)
	${CONDA_BIN} clean --all



cc:
	@echo "Cleaning cache"
	$(eval PYTHON_DIST := miniforge3-latest)
	$(eval CONDA_BIN := ~/.pyenv/versions/${PYTHON_DIST}/bin/conda)
	${CONDA_BIN}  clean -i

update:
	$(PYTHON) -m pip install --upgrade -r requirements.txt --upgrade-strategy=eager

download:
	$(PYTHON) -m pip download -r requirements.txt -d downloads

