#!/usr/bin/env bash
#
# All-in-one installer for the Quebec French dubbing pipeline + web UI.
#
# Run in one line on a fresh CUDA GPU box (Debian/Ubuntu or macOS):
#
#   curl -fsSL https://raw.githubusercontent.com/tenutso/quebec-french-dubbing/main/install.sh | bash
#
# or, from a clone:  ./install.sh [options]
#
# It installs system deps (ffmpeg, rubberband-cli), creates the GPU venv and
# installs the pipeline + Gradio UI (the same steps as `make install`), then sets
# up Ollama + a French model for local translation. Add --launch to open the web
# UI when it finishes.
#
# Options:
#   --dir PATH      where to clone/use the repo         (default ./quebec-french-dubbing)
#   --model NAME    Ollama translation model to pull    (default mistral-small)
#   --no-ollama     skip installing/pulling Ollama
#   --no-system     skip apt/brew system packages
#   --launch        launch the web UI after install
#   --share         with --launch, expose a public Gradio link
#   -h, --help      show this help
#
set -euo pipefail

REPO_URL="${DUBBING_REPO_URL:-https://github.com/tenutso/quebec-french-dubbing.git}"
TARGET_DIR="${DUBBING_DIR:-quebec-french-dubbing}"
OLLAMA_MODEL="${DUBBING_MODEL:-mistral-small}"
DO_OLLAMA=1
DO_SYSTEM=1
DO_LAUNCH=0
SHARE_FLAG=""

log()  { printf '\033[1;36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m warn:\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31merror:\033[0m %s\n' "$*" >&2; exit 1; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dir)       TARGET_DIR="$2"; shift 2 ;;
    --model)     OLLAMA_MODEL="$2"; shift 2 ;;
    --no-ollama) DO_OLLAMA=0; shift ;;
    --no-system) DO_SYSTEM=0; shift ;;
    --launch)    DO_LAUNCH=1; shift ;;
    --share)     SHARE_FLAG="--share"; shift ;;
    -h|--help)   sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *)           die "unknown option: $1" ;;
  esac
done

# --- sudo + package-manager helpers -----------------------------------------
SUDO=""
if [ "$(id -u)" -ne 0 ]; then command -v sudo >/dev/null 2>&1 && SUDO="sudo"; fi

install_system_deps() {
  local pkgs="ffmpeg rubberband-cli"
  if command -v apt-get >/dev/null 2>&1; then
    log "Installing system packages (apt): $pkgs"
    $SUDO apt-get update -qq
    $SUDO apt-get install -y -qq $pkgs
  elif command -v brew >/dev/null 2>&1; then
    log "Installing system packages (brew): ffmpeg rubberband"
    brew install ffmpeg rubberband
  else
    warn "No apt-get/brew found. Install ffmpeg and rubberband-cli manually."
  fi
}

# --- prerequisites ----------------------------------------------------------
command -v python3 >/dev/null 2>&1 || die "python3 is required"
command -v git     >/dev/null 2>&1 || die "git is required"
if ! command -v nvidia-smi >/dev/null 2>&1; then
  warn "nvidia-smi not found — the GPU pipeline needs an NVIDIA GPU + CUDA driver."
fi

# --- get the code -----------------------------------------------------------
# If we're already inside the repo, use it in place; otherwise clone.
if [ -f "pyproject.toml" ] && grep -q 'name = "dubbing"' pyproject.toml 2>/dev/null; then
  REPO_DIR="$(pwd)"
  log "Using existing checkout at $REPO_DIR"
else
  if [ -d "$TARGET_DIR/.git" ]; then
    log "Updating existing clone at $TARGET_DIR"
    git -C "$TARGET_DIR" pull --ff-only
  else
    log "Cloning $REPO_URL -> $TARGET_DIR"
    git clone --depth 1 "$REPO_URL" "$TARGET_DIR"
  fi
  REPO_DIR="$(cd "$TARGET_DIR" && pwd)"
fi
cd "$REPO_DIR"

# --- system + python env ----------------------------------------------------
[ "$DO_SYSTEM" -eq 1 ] && install_system_deps || warn "Skipping system packages (--no-system)"

log "Installing the pipeline + web UI into .venv (this pulls the GPU model stack)"
make install

# --- Ollama (local translation LLM) -----------------------------------------
if [ "$DO_OLLAMA" -eq 1 ]; then
  if ! command -v ollama >/dev/null 2>&1; then
    if command -v curl >/dev/null 2>&1; then
      log "Installing Ollama"
      curl -fsSL https://ollama.com/install.sh | sh
    else
      warn "curl not found; skipping Ollama install."
    fi
  fi
  if command -v ollama >/dev/null 2>&1; then
    pgrep -f "ollama serve" >/dev/null 2>&1 || { log "Starting ollama serve"; (ollama serve >/tmp/ollama.log 2>&1 &) ; sleep 4; }
    log "Pulling translation model: $OLLAMA_MODEL"
    ollama pull "$OLLAMA_MODEL"
  fi
else
  warn "Skipping Ollama (--no-ollama). Use a premium translation provider instead."
fi

# --- done -------------------------------------------------------------------
cat <<EOF

$(printf '\033[1;32m✔ Install complete.\033[0m')

Before running, export your Hugging Face token for the gated diarization model:
  export HF_TOKEN=hf_xxx          # accept terms once at the model's HF page
  export DUBBING_GPU_BACKEND=local

Launch the web UI (0.0.0.0:7860):
  cd "$REPO_DIR" && make web           # add SHARE=--share for a public link
Or run a job from the CLI:
  cd "$REPO_DIR" && .venv/bin/dubbing run config/job.yaml
EOF

if [ "$DO_LAUNCH" -eq 1 ]; then
  log "Launching web UI"
  exec make web SHARE="$SHARE_FLAG"
fi
