VENV ?= .venv
PY   := $(VENV)/bin/python
PIP  := $(VENV)/bin/pip

# Chatterbox over-pins its deps (notably torch==2.6.0), so it is installed with
# --no-deps against the host CUDA torch trio. Bump this to move the version.
CHATTERBOX_VERSION ?= 0.1.7
# WhisperX 3.8.6 caps huggingface_hub <1.0 in its metadata, which conflicts with
# Chatterbox's transformers==5.2.0 (needs hf_hub 1.x). The cap is stale — 3.8.6
# runs fine on hf_hub 1.x — so whisperx is installed --no-deps; its real deps are
# in pyproject core. Bump this to move the version.
WHISPERX_VERSION ?= 3.8.6
# PyTorch wheels matched to the CUDA build pinned in constraints-gpu.txt.
TORCH_INDEX ?= https://download.pytorch.org/whl/cu128

.PHONY: help venv install install-premium test fixture sample deploy-gpu clean

help:
	@echo "Targets:"
	@echo "  install         Create the GPU venv and install the full local pipeline"
	@echo "  install-premium Add the optional premium cloud provider SDKs"
	@echo "  test            Run the test suite"
	@echo "  fixture         Generate tests/fixtures/sample_2spk.mp4"
	@echo "  sample          Run the pipeline on config/job.yaml"
	@echo "  deploy-gpu      Deploy the Modal GPU app"

# --system-site-packages reuses the host's CUDA-matched PyTorch.
venv:
	python3 -m venv --system-site-packages $(VENV)

# One environment, one command. Installs the core + local-TTS deps against the
# host torch trio, then adds WhisperX and Chatterbox with --no-deps (each carries
# a stale/over-strict pin — hf_hub<1.0 and torch==2.6.0 respectively — that a
# clean resolve cannot satisfy; their real deps are already in the resolve above).
install: venv
	$(PIP) install -q -U pip
	$(PIP) install -q --constraint constraints-gpu.txt --extra-index-url $(TORCH_INDEX) \
	  -e ".[dev,tts-local]"
	$(PIP) install -q --no-deps "whisperx==$(WHISPERX_VERSION)" "chatterbox-tts==$(CHATTERBOX_VERSION)"

install-premium:
	$(PIP) install -q --constraint constraints-gpu.txt --extra-index-url $(TORCH_INDEX) \
	  -e ".[premium]"

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
