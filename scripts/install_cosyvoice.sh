#!/usr/bin/env bash
#
# Install CosyVoice (Fun-CosyVoice 3.0) into this repo's .venv for the `cosyvoice` TTS
# provider. CosyVoice is a from-source package with its own requirements, so this is a
# SEPARATE, optional step (not part of `make install`).
#
# WARNING: CosyVoice's requirements can conflict with the pinned pipeline stack (torch 2.8
# / transformers 5.2). We install them under constraints-gpu.txt to protect the CUDA torch
# trio, but re-run `make test` afterward and, if anything broke, prefer an isolated venv.
#
#   ./scripts/install_cosyvoice.sh                 # clone + deps + model download
#   COSYVOICE_DIR=/opt/CosyVoice ./scripts/install_cosyvoice.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-$REPO_ROOT/.venv}"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
COSYVOICE_DIR="${COSYVOICE_DIR:-$REPO_ROOT/third_party/CosyVoice}"
MODEL_ID="${COSYVOICE_MODEL_ID:-FunAudioLLM/Fun-CosyVoice3-0.5B-2512}"
MODEL_DIR="${COSYVOICE_MODEL_DIR:-$COSYVOICE_DIR/pretrained_models/Fun-CosyVoice3-0.5B}"

log() { printf '\033[1;36m==>\033[0m %s\n' "$*"; }

[ -x "$PY" ] || { echo "error: $PY not found — run 'make install' first."; exit 1; }

# 1. Clone (recursive: pulls the third_party/Matcha-TTS submodule the package imports).
if [ -d "$COSYVOICE_DIR/.git" ]; then
  log "Updating CosyVoice checkout at $COSYVOICE_DIR"
  git -C "$COSYVOICE_DIR" pull --ff-only
  git -C "$COSYVOICE_DIR" submodule update --init --recursive
else
  log "Cloning CosyVoice -> $COSYVOICE_DIR"
  git clone --recursive https://github.com/FunAudioLLM/CosyVoice.git "$COSYVOICE_DIR"
fi

# 2. Install CosyVoice's deps WITHOUT disturbing the pinned stack.
# CosyVoice's requirements.txt hard-pins torch==2.3.1 / transformers==4.51.3 /
# numpy==1.26.4 (older than ours), so `-r requirements.txt` is unsatisfiable against
# the pipeline env. Instead install only the deps it needs that we don't already
# provide, under a constraint that forbids downgrading the pinned stack. CosyVoice
# imports and runs fine against the newer torch/transformers/numpy.
log "Installing CosyVoice's additive deps (pinned stack protected from downgrade)"
CONSTRAINT="$(mktemp)"
cat "$REPO_ROOT/constraints-gpu.txt" > "$CONSTRAINT"
cat >> "$CONSTRAINT" <<'EOF'
transformers==5.2.0
numpy>=2.1
tokenizers>=0.22
librosa>=0.11
lightning>=2.6
pydantic>=2.9
protobuf>=7
networkx>=3.3
gradio>=6
EOF
"$PIP" install --constraint "$CONSTRAINT" \
  hyperpyyaml gdown hydra-core wget inflect wetext modelscope \
  openai-whisper pyworld x-transformers onnxruntime pyarrow

# 3. Download the Fun-CosyVoice 3.0 model (once).
log "Downloading model $MODEL_ID -> $MODEL_DIR"
"$PY" - "$MODEL_ID" "$MODEL_DIR" <<'PY'
import sys
from huggingface_hub import snapshot_download
snapshot_download(sys.argv[1], local_dir=sys.argv[2])
print("model at", sys.argv[2])
PY

cat <<EOF

$(printf '\033[1;32m✔ CosyVoice installed.\033[0m')  Now point the provider at it and select it:

  export COSYVOICE_ROOT="$COSYVOICE_DIR"
  export COSYVOICE_MODEL_DIR="$MODEL_DIR"
  .venv/bin/dubbing run config/job.yaml --tts-provider cosyvoice   # or pick "cosyvoice" in the web UI

Re-run 'make test' to confirm the install did not disturb the pinned stack.
EOF
