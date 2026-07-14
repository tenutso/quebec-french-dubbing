VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

.PHONY: help venv install install-audio install-gpu test fixture sample deploy-gpu clean

help:
	@echo "Targets:"
	@echo "  install        Core deps (models/subtitles) into $(VENV)"
	@echo "  install-audio  Add audio mastering deps (numpy/soundfile/pyloudnorm)"
	@echo "  install-gpu    Add OSS GPU stack (demucs/pyannote/whisperx) — needs a GPU"
	@echo "  test           Run the test suite"
	@echo "  fixture        Generate tests/fixtures/sample_2spk.mp4"
	@echo "  sample         Run the pipeline on config/job.yaml"
	@echo "  deploy-gpu     Deploy the Modal GPU app"

venv:
	python3 -m venv $(VENV)

install: venv
	$(PIP) install -q -U pip
	$(PIP) install -q -e ".[dev,translate,tts]"

install-audio:
	$(PIP) install -q -e ".[audio]"

install-gpu:
	$(PIP) install -q -e ".[gpu,orchestration]"

test:
	$(VENV)/bin/pytest -q

fixture:
	$(PY) scripts/make_fixture.py

sample: fixture
	$(PY) -m dubbing.cli run config/job.yaml

deploy-gpu:
	$(VENV)/bin/modal deploy src/dubbing/modal_app.py

clean:
	rm -rf .work tests/fixtures/sample_2spk.mp4
