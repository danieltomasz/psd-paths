.ONESHELL:
.PHONY: install

PROJECT?=paths
VERSION?=3.11.8

VENV=${PROJECT}-${VERSION}
VENV_DIR=$(shell pyenv root)/versions/${VENV}
PYTHON=${VENV_DIR}/bin/python
JUPYTER_ENV_NAME=${VENV}

install:
	@echo "Installing $(VENV)"
	env PYTHON_CONFIGURE_OPTS=--enable-shared pyenv virtualenv ${VERSION} ${VENV}
	pyenv local ${VENV}
	$(PYTHON) -m pip  install -U pip
	$(PYTHON) -m pip install  -r requirements.txt
	PYDEVD_DISABLE_FILE_VALIDATION=1  $(PYTHON) -m ipykernel install --user --name ${VENV}

update:
	$(PYTHON) -m pip install --upgrade -r requirements.txt --upgrade-strategy=eager