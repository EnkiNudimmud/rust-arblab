# Makefile for common tasks (use `make <target>`)

VENV = .venv
PY = $(VENV)/bin/python
PIP = $(VENV)/bin/pip
MATURIN = $(VENV)/bin/maturin

.PHONY: help venv create-venv install deps build-rust run-streamlit docker-build docker-up clean

help:
	@echo "Targets:"
	@echo "  make create-venv    -> create a .venv using system python (or edit PY if needed)"
	@echo "  make install        -> install python deps into the .venv"
	@echo "  make build-rust     -> build & install rust extension via maturin"
	@echo "  make run-streamlit  -> run streamlit app"
	@echo "  make docker-build   -> docker-compose build"
	@echo "  make docker-up      -> docker-compose up"

create-venv:
	# create a venv using the conda python you prefer
	/Users/melvinalvarez/miniconda3/envs/rhftlab/bin/python3.11 -m venv $(VENV)
	@echo "Created venv at $(VENV). Activate it with: source $(VENV)/bin/activate"

install: create-venv
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install --no-cache-dir -r docker/requirements.txt

build-rust: install
	$(PIP) install --upgrade maturin
	$(MATURIN) develop --manifest-path rust_connector/Cargo.toml --release

run-streamlit: install
	$(PY) -m streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0

docker-build:
	docker-compose build --no-cache --pull

docker-up:
	docker-compose up --build

clean:
	rm -rf $(VENV) build dist target *.egg-info