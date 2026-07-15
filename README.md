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

## Quick start

On a fresh CUDA GPU box (Debian/Ubuntu or macOS), one command installs everything — system
deps, the Python env, the web UI, and Ollama + a French model — then you can launch the
browser app:

```bash
curl -fsSL https://raw.githubusercontent.com/tenutso/quebec-french-dubbing/main/install.sh | bash
# then, from the cloned repo:
export HF_TOKEN=hf_xxx          # free, for the gated diarization model (accept its terms once)
export DUBBING_GPU_BACKEND=local
make web                        # http://localhost:7860  (add SHARE=--share for a public link)
```

`install.sh --launch` installs and opens the UI in one step. Prefer the CLI or a hand install?
See [Run a dub](#run-a-dub-on-your-own-file) and [Setup](#setup).

---

## Web UI

`make web` (or `dubbing-web`) serves a Gradio app on **`0.0.0.0:7860`**: drop in an MP4, pick
the dub style / providers / loudness, watch per-stage progress, and preview + download the
dubbed video, subtitles, and mastered dub track. It calls the same pipeline as the CLI, so the
same prerequisites apply (GPU, `ffmpeg` + `rubberband-cli`, `HF_TOKEN`, and Ollama for local
translation). A **Chatterbox voice tuning** panel exposes the cross-lingual knobs (CFG weight,
exaggeration, temperature, chunk size, and an optional native fr-CA reference clip) so you can
A/B accent/prosody without setting `CHATTERBOX_*` env vars by hand.

```bash
make web                       # localhost + LAN on :7860
make web SHARE=--share         # also expose a temporary public Gradio link
make web PORT=8000             # or dubbing-web --host 0.0.0.0 --port 8000 --share
```

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
.venv/bin/dubbing run config/job.yaml
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
.venv/bin/dubbing run config/job.yaml --tts-provider azure          # native fr-CA neural voices
.venv/bin/dubbing run config/job.yaml --translation-provider claude # premium translation
```

### Tuning the local voice (Chatterbox)

Chatterbox clones each speaker zero-shot. Cloning an English speaker into French bleeds some
English prosody in (inherent to cross-lingual cloning). Dials, via env vars:

```bash
export CHATTERBOX_CFG_WEIGHT=0.5     # default; lower (0.2–0.3) = less English-reference prosody
export CHATTERBOX_EXAGGERATION=0.5   # lower = calmer delivery
export CHATTERBOX_TEMPERATURE=0.8    # sampling variance
export CHATTERBOX_MAX_CHARS=180      # synth chunk size (smaller = more stable, fewer cutoffs)
export CHATTERBOX_FR_REF=/path/fr.wav  # use a neutral fr-CA clip for prosody (loses speaker identity)
export CHATTERBOX_MODEL_DIR=/path/ckpt # load a fine-tuned checkpoint (see below); blank = base model
```

For the strongest **native Québec prosody**, use `--tts-provider azure` (preset fr-CA neural
voices; no cloning). The same dials are exposed in the [Web UI](#web-ui)'s *Chatterbox voice
tuning* panel.

### Fine-tuning Chatterbox on Québec French

You can train your own Québec-French checkpoint and point the pipeline at it with
`CHATTERBOX_MODEL_DIR` (or the web UI's *Fine-tuned checkpoint dir* field), which loads it via
Chatterbox's `from_local(ckpt_dir, device)`.

> **What this fixes — and what it doesn't.** A fine-tune improves the model's *base* Québec
> pronunciation, vocabulary and default prosody. But the dub clones the **English** speaker
> zero-shot, and that reference still pulls prosody toward English — so a fine-tune *reduces*
> the accent, it doesn't eliminate it. Best results come from a fine-tuned checkpoint **plus**
> a low `CHATTERBOX_CFG_WEIGHT` (~0.3) and/or a neutral `CHATTERBOX_FR_REF` clip. If you need a
> fully native accent and don't need the original timbre, a preset fr-CA voice (`--tts-provider
> azure`) will always win.

The standard recipe is a **LoRA fine-tune of Chatterbox's T3 backbone** (~7.8M trainable
params of the 0.5B Llama model); `ve`/`s3gen`/tokenizer stay frozen. The real training run is a
separate offline effort (a QC dataset + GPU-hours), not part of `make install`:

1. **Data.** Collect Québec-French audio + accurate transcripts. A good start is Mozilla
   [Common Voice](https://commonvoice.mozilla.org) `fr` filtered to `accent = Canada/Québec`;
   supplement with other QC sources. Clean and VAD-trim to 24 kHz mono. LoRA is viable from a
   few hours of speech; ~5–20 h is comfortable.
2. **Preprocess.** Encode speech tokens with `s3tokenizer` and text with Chatterbox's
   `MTLTokenizer` (`grapheme_mtl_merged_expanded_v1.json`), using `language_id="fr"`.
3. **Train.** LoRA (rank-32 on the T3 attention/MLP projections), freezing `ve`/`s3gen`. Use a
   community trainer as a base — e.g.
   [chatterbox-finetuning-multilingual](https://github.com/Ahmed-Ezzat20/chatterbox-finetuning-multilingual)
   (`lora.py`) or [AliAbdallah21/…](https://github.com/AliAbdallah21/Chatterbox-Multilingual-TTS-Fine-Tuning);
   good quality is reported around ~2000 steps.
4. **Merge + package.** Merge the LoRA adapter into T3 and save `t3_mtl23ls_v2.safetensors`.
   Grab the base auxiliary files (`huggingface-cli download ResembleAI/chatterbox`) and place
   `ve.pt`, `s3gen.pt`, `grapheme_mtl_merged_expanded_v1.json` (and optional `conds.pt`)
   alongside your merged T3 in one `ckpt_dir`.
5. **Use.** `export CHATTERBOX_MODEL_DIR=/path/ckpt_dir` and run as usual, combined with the
   `CHATTERBOX_CFG_WEIGHT` / `CHATTERBOX_FR_REF` dials above.

---

## Setup

One environment. The venv is created with `--system-site-packages` so it reuses the host's
CUDA-matched PyTorch, and everything installs with a single command (needs a CUDA PyTorch
already present on the host):

```bash
make install        # creates .venv, installs the full local GPU pipeline
make test           # 39 tests
```

`make install` does exactly what you'd run by hand:

```bash
python3 -m venv --system-site-packages .venv
.venv/bin/pip install -U pip
# Core + local-TTS deps, against the host torch trio:
.venv/bin/pip install --constraint constraints-gpu.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  -e ".[dev,tts-local]"
# WhisperX and Chatterbox, each WITHOUT its deps — both carry a pin that a clean
# resolve can't satisfy (whisperx caps huggingface_hub<1.0, which is stale;
# chatterbox hard-pins torch==2.6.0). Their real deps are already installed above.
.venv/bin/pip install --no-deps whisperx==3.8.6 chatterbox-tts==0.1.7

# Optional premium cloud voices/translation (each needs that vendor's key):
make install-premium

# Ollama (local translation LLM):
#   install Ollama, then: ollama serve &  &&  ollama pull mistral-small
```

`constraints-gpu.txt` pins the `torch/torchaudio/torchvision` trio (plus a CUDA-matched
`torchcodec`) so the model libraries can't downgrade your CUDA build. Two libraries can't be
resolved cleanly and are installed with `--no-deps`: **Chatterbox** (hard-pins `torch==2.6.0`;
its other deps come from the `tts-local` extra) and **WhisperX** (its metadata caps
`huggingface_hub<1.0`, which is stale — it runs fine on the 1.x that Chatterbox's
`transformers==5.2.0` requires; its real deps live in the core dependency list).

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
  webapp.py              Gradio web UI (`dubbing-web` / `make web`)
  models.py              pydantic artifacts + job config
  subtitle_rules.py      Netflix fr-CA subtitle standards engine
  ffmpeg_utils.py        probe / extract / loudness helpers
  gpu_runners.py         Demucs / pyannote / WhisperX (OSS GPU)
  modal_app.py           on-demand GPU functions (Modal)
  stages/                one module per pipeline stage
  providers/             translation (ollama/claude) + TTS (chatterbox/elevenlabs/azure)
```
