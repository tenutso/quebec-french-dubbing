# Quebec French Dubbing Pipeline

Turns a multi-speaker webinar/course video into a **high-quality Québec French (fr-CA) dub +
industry-standard subtitles**.

**Cost strategy:** the default path is **fully local and free** — open-source models on your
GPU for every stage. Premium cloud vendors (Claude, ElevenLabs, Azure) remain pluggable for
where they move quality most, but nothing is required beyond GPU time.

| Stage | Default engine | Cost |
|---|---|---|
| Source separation (vocals vs music/SFX) | Demucs | free (GPU) |
| Diarization (who spoke when) | pyannote.audio | free (GPU) |
| ASR + word timestamps | WhisperX (large-v3) | free (GPU) |
| **Translation** EN→fr-CA | **Ollama** (`mistral-small`) | free (GPU) — or `claude` (premium) |
| Subtitles (SRT/VTT, Netflix fr-CA style) | pysubs2 + rules | free |
| **TTS** (per-speaker voice cloning) | **Chatterbox** (multilingual) | free (GPU) — or `elevenlabs`/`azure` |
| Time-fit / mix (EBU R128) / mux | ffmpeg + numpy | free |

---

## Run a dub on your own file

> Assumes a Linux box with an NVIDIA GPU (≈16GB+), `ffmpeg`, and the project installed (see
> [Setup](#setup)). Everything below runs locally — no API keys except a free Hugging Face
> token for the (gated) diarization model.

**1. Start the local LLM** (translation) and make sure the model is present:

```bash
pgrep -f "ollama serve" >/dev/null || (ollama serve >/tmp/ollama.log 2>&1 &)
ollama list | grep -q mistral-small || ollama pull mistral-small
```

**2. Provide the Hugging Face token** for pyannote and select the local GPU backend:

```bash
export HF_TOKEN=hf_xxx              # accept the model terms once (see Notes)
export DUBBING_GPU_BACKEND=local
```

**3. Point a job config at your video.** Copy the template and edit `input_path`/`work_dir`
(use absolute paths):

```bash
cp config/job.example.yaml config/job.yaml
# edit config/job.yaml:
#   input_path: /abs/path/to/your_video.mp4
#   work_dir:   /abs/path/to/output_dir
```

**4. Run it:**

```bash
.venv-gpu/bin/dubbing run config/job.yaml
```

**5. Collect the outputs** from your `work_dir`:

| File | What it is |
|---|---|
| `<name>.fr-CA.mp4` | video with the **French dub as the default audio track** (original kept as a secondary track) |
| `<name>.fr-CA.srt` / `.vtt` | Québec-French subtitles |
| `dub_track.wav` | the mastered French audio on its own |

The first run downloads the WhisperX and Chatterbox models once, then caches them. Runtime
scales with video length (roughly real-time analysis + a few seconds of GPU per spoken cue).

### Common variations

```bash
# Subtitles only — no dubbing, much faster:
#   set `dub_style: subtitles_only` in config/job.yaml, then run.

# Voice-over instead of full replacement (original ducked under the French):
#   set `dub_style: voice_over`.

# A/B a premium vendor without editing the file (needs that vendor's API key):
.venv-gpu/bin/dubbing run config/job.yaml --tts-provider azure          # native fr-CA neural voices
.venv-gpu/bin/dubbing run config/job.yaml --translation-provider claude # premium translation
```

### Tuning the local voice (Chatterbox)

Chatterbox clones each speaker zero-shot. Cloning an English speaker into French bleeds some
English prosody in (inherent to cross-lingual cloning). Dials, via env vars:

```bash
export CHATTERBOX_CFG_WEIGHT=0.5     # default; lower (0.2–0.3) = less English-reference prosody
export CHATTERBOX_EXAGGERATION=0.5   # lower = calmer delivery
export CHATTERBOX_FR_REF=/path/fr.wav  # use a neutral fr-CA clip for prosody (loses speaker identity)
```

For the strongest **native Québec prosody**, use `--tts-provider azure` (preset fr-CA neural
voices; no cloning).

---

## Setup

Two virtualenvs: a light one for tests, and a GPU one (created with `--system-site-packages`
so it reuses the host's CUDA-matched PyTorch).

```bash
# Light env for the unit tests (no GPU/model deps):
make install        # core + provider SDKs into .venv
make test           # 39 tests

# GPU env for real runs (needs a CUDA PyTorch already present on the host):
python3 -m venv --system-site-packages .venv-gpu
.venv-gpu/bin/pip install -U pip
.venv-gpu/bin/pip install --constraint constraints-gpu.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -e ".[dev,audio,translate,tts,gpu]" chatterbox-tts

# Ollama (local translation LLM):
#   install Ollama, then: ollama serve &  &&  ollama pull mistral-small
```

`constraints-gpu.txt` pins the `torch/torchaudio/torchvision` trio so the model libraries
(pyannote, WhisperX, Chatterbox — which over-pin their deps) can't downgrade your CUDA build.

The GPU stages can also run on **Modal** instead of locally: `make deploy-gpu`, then
`export DUBBING_GPU_BACKEND=modal`.

---

## Pluggable providers

TTS and translation are Strategy interfaces resolved by name from a registry
(`src/dubbing/providers/`); a provider that can't produce `fr-CA` is rejected at job start.

- **Translation:** `ollama` (local, default), `claude` (premium).
- **TTS:** `chatterbox` (local cloning, default), `elevenlabs` (cloning), `azure` (fr-CA neural).

Select per job in `config/job.yaml` under `providers:`, or override on the CLI with
`--tts-provider` / `--translation-provider`.

---

## Notes

- **Diarization is gated.** `pyannote/speaker-diarization-community-1` requires a (free) HF
  account to accept its conditions once on the model's Hugging Face page; `HF_TOKEN` then
  authenticates it. Override the pipeline with `PYANNOTE_PIPELINE` if needed.
- **VRAM:** the Ollama model is unloaded automatically after translation so the local TTS
  isn't starved of GPU memory.
- **Secrets** live in your environment (or a `.env` you source) and are never committed.

## Layout

```
src/dubbing/
  pipeline.py            stage runner
  cli.py                 `dubbing run <config>` entry point
  models.py              pydantic artifacts + job config
  subtitle_rules.py      Netflix fr-CA subtitle standards engine
  ffmpeg_utils.py        probe / extract / loudness helpers
  gpu_runners.py         Demucs / pyannote / WhisperX (OSS GPU)
  modal_app.py           on-demand GPU functions (Modal)
  stages/                one module per pipeline stage
  providers/             translation (ollama/claude) + TTS (chatterbox/elevenlabs/azure)
```
