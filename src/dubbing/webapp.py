"""Gradio web UI for the Quebec-French dubbing pipeline.

Upload an MP4, pick the dub options, and the same :func:`dubbing.pipeline.run`
that powers the CLI produces a French dub + subtitles you can preview and
download. Launch with the ``dubbing-web`` entry point (or ``make web``); it binds
to ``0.0.0.0:7860`` by default so it's reachable on your LAN, with ``--share`` for
a temporary public Gradio link.

Runtime prerequisites are the same as a CLI run: a CUDA GPU, ``ffmpeg`` +
``rubberband-cli``, ``HF_TOKEN`` for the gated diarization model, and — for the
default local providers — Ollama serving a French model (``mistral-small``).
"""

from __future__ import annotations

import argparse
import os
import tempfile
import traceback
from datetime import datetime
from pathlib import Path

import gradio as gr

from dubbing.models import DubStyle, Job, LoudnessTarget, ProviderSelection, VoiceStrategy
from dubbing.pipeline import run

# The tracked glossary ships in the repo; fall back to none if running from a
# layout where it isn't present (e.g. a packaged wheel).
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_GLOSSARY = _REPO_ROOT / "config" / "glossary.fr-CA.yaml"


def _workroot() -> Path:
    """Base directory for per-run work dirs (override with DUBBING_WEB_WORKROOT)."""
    root = Path(os.environ.get("DUBBING_WEB_WORKROOT", _REPO_ROOT / ".work" / "web"))
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_job(
    video_path: str | Path,
    work_dir: str | Path,
    *,
    dub_style: str,
    voice_strategy: str,
    loudness: str,
    burn_in_subtitles: bool,
    translation: str,
    tts: str,
) -> Job:
    """Assemble a validated :class:`Job` from the UI selections.

    Kept separate from the Gradio handler so it is unit-testable without a browser.
    """
    return Job(
        input_path=Path(video_path),
        work_dir=Path(work_dir),
        dub_style=DubStyle(dub_style),
        voice_strategy=VoiceStrategy(voice_strategy),
        loudness=LoudnessTarget(loudness),
        providers=ProviderSelection(translation=translation, tts=tts),
        burn_in_subtitles=bool(burn_in_subtitles),
        glossary_path=_DEFAULT_GLOSSARY if _DEFAULT_GLOSSARY.exists() else None,
    )


def apply_chatterbox_env(
    cfg_weight, exaggeration, temperature, fr_ref, max_chars, model_dir=None
) -> None:
    """Set the Chatterbox generation knobs the provider reads from the environment.

    The provider is re-instantiated each run (reading these env vars in __init__), and the
    app serializes runs via ``queue()``, so setting process env per run is safe. Every run
    sets all knobs — including clearing the fr-CA reference and fine-tuned checkpoint dir —
    so nothing leaks across runs.
    """
    os.environ["CHATTERBOX_CFG_WEIGHT"] = str(float(cfg_weight))
    os.environ["CHATTERBOX_EXAGGERATION"] = str(float(exaggeration))
    os.environ["CHATTERBOX_TEMPERATURE"] = str(float(temperature))
    os.environ["CHATTERBOX_MAX_CHARS"] = str(int(max_chars))
    ref = getattr(fr_ref, "name", fr_ref) if fr_ref else None
    if ref:
        os.environ["CHATTERBOX_FR_REF"] = str(ref)
    else:
        os.environ.pop("CHATTERBOX_FR_REF", None)
    if model_dir and str(model_dir).strip():
        os.environ["CHATTERBOX_MODEL_DIR"] = str(model_dir).strip()
    else:
        os.environ.pop("CHATTERBOX_MODEL_DIR", None)


def process(
    video_file,
    dub_style: str,
    voice_strategy: str,
    loudness: str,
    burn_in_subtitles: bool,
    translation: str,
    tts: str,
    cfg_weight: float,
    exaggeration: float,
    temperature: float,
    fr_ref,
    max_chars: int,
    model_dir: str = "",
    progress=gr.Progress(),
):
    """Run the full pipeline for an uploaded file and return UI outputs.

    Returns ``(video_path_or_None, list_of_download_files, status_markdown)``.
    """
    if not video_file:
        return None, [], "⚠️ Please upload an MP4 first."

    src = Path(getattr(video_file, "name", video_file))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    work_dir = Path(tempfile.mkdtemp(prefix=f"{src.stem}-{stamp}-", dir=_workroot()))

    apply_chatterbox_env(cfg_weight, exaggeration, temperature, fr_ref, max_chars, model_dir)
    job = build_job(
        src, work_dir,
        dub_style=dub_style, voice_strategy=voice_strategy, loudness=loudness,
        burn_in_subtitles=burn_in_subtitles, translation=translation, tts=tts,
    )

    def _on_stage(label: str, done: int, total: int) -> None:
        progress(done / total, desc=label)

    try:
        ctx = run(job, progress=_on_stage)
    except Exception:  # surface the failure in the UI instead of a blank error
        tb = traceback.format_exc()
        return None, [], f"❌ **Dub failed.**\n\n```\n{tb.strip()[-2000:]}\n```"

    # Collect deliverables: the muxed video, the sidecar subtitles, the dub track.
    downloads: list[str] = []
    out_video = str(ctx.output_path) if ctx.output_path else None
    if out_video:
        downloads.append(out_video)
    for path in ctx.subtitle_paths.values():
        downloads.append(str(path))
    if ctx.dub_track and Path(ctx.dub_track.path).exists():
        downloads.append(str(ctx.dub_track.path))

    style = job.dub_style.value
    status = (
        f"✅ **Done** — `{style}` dub via `{translation}` + `{tts}`.\n\n"
        f"Outputs in `{work_dir}`."
    )
    # subtitles_only jobs have no video to preview.
    preview = out_video if out_video and out_video.lower().endswith(".mp4") else None
    return preview, downloads, status


def build_demo():
    """Construct the Gradio Blocks app (no server started)."""
    with gr.Blocks(title="Quebec French Dubbing", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 🎬 Quebec French (fr-CA) Dubbing\n"
            "Upload an MP4 to produce a **Québec-French dub + subtitles**. "
            "The default path runs fully local on your GPU (Ollama translation, "
            "Chatterbox voice cloning). Premium providers need the matching API key "
            "and `make install-premium`."
        )
        with gr.Row():
            with gr.Column(scale=1):
                video_in = gr.File(
                    label="Source video (.mp4)", file_types=[".mp4", ".mov", ".mkv"]
                )
                dub_style = gr.Radio(
                    [s.value for s in DubStyle], value=DubStyle.FULL_REPLACEMENT.value,
                    label="Dub style",
                    info="full_replacement swaps speech (keeps music); "
                         "voice_over ducks the original; subtitles_only skips the dub.",
                )
                with gr.Row():
                    voice_strategy = gr.Radio(
                        [s.value for s in VoiceStrategy], value=VoiceStrategy.CLONE.value,
                        label="Voice strategy",
                    )
                    loudness = gr.Radio(
                        [s.value for s in LoudnessTarget], value=LoudnessTarget.WEB.value,
                        label="Loudness", info="web -16 LUFS · broadcast -23 LUFS",
                    )
                with gr.Row():
                    translation = gr.Dropdown(
                        ["ollama", "claude"], value="ollama", label="Translation",
                    )
                    tts = gr.Dropdown(
                        ["chatterbox", "cosyvoice", "elevenlabs", "azure"],
                        value="chatterbox", label="TTS voice",
                    )
                burn_in = gr.Checkbox(
                    label="Burn subtitles into the video", value=False
                )
                with gr.Accordion("Chatterbox voice tuning", open=False):
                    gr.Markdown(
                        "Cross-lingual cloning bleeds the English speaker's prosody into "
                        "the French. Lower **CFG weight** to reduce that; supply a neutral "
                        "**fr-CA reference** for native prosody (trades away the cloned "
                        "voice). Applies only when TTS = `chatterbox`."
                    )
                    cfg_weight = gr.Slider(
                        0.0, 1.0, value=0.5, step=0.05, label="CFG weight",
                        info="lower (0.2–0.3) = less English-reference prosody",
                    )
                    with gr.Row():
                        exaggeration = gr.Slider(
                            0.0, 1.0, value=0.5, step=0.05, label="Exaggeration",
                            info="lower = calmer delivery",
                        )
                        temperature = gr.Slider(
                            0.1, 1.5, value=0.8, step=0.05, label="Temperature",
                        )
                    max_chars = gr.Slider(
                        80, 400, value=180, step=10, label="Chunk size (chars)",
                        info="smaller = more stable, fewer runaways/early cutoffs",
                    )
                    fr_ref = gr.File(
                        label="Native fr-CA reference clip (optional, overrides clone)",
                        file_types=[".wav", ".mp3", ".flac"],
                    )
                    model_dir = gr.Textbox(
                        label="Fine-tuned checkpoint dir (optional)",
                        placeholder="/path/to/qc-fr checkpoint dir (from_local); blank = base model",
                        info="a Québec-French T3 checkpoint — see the README fine-tuning recipe",
                    )
                run_btn = gr.Button("Run dub", variant="primary")
            with gr.Column(scale=1):
                video_out = gr.Video(label="Dubbed preview")
                files_out = gr.Files(label="Download (video · SRT · VTT · dub track)")
                status_out = gr.Markdown()

        run_btn.click(
            fn=process,
            inputs=[
                video_in, dub_style, voice_strategy, loudness, burn_in, translation, tts,
                cfg_weight, exaggeration, temperature, fr_ref, max_chars, model_dir,
            ],
            outputs=[video_out, files_out, status_out],
        )
    return demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dubbing-web", description="Dubbing web UI")
    parser.add_argument("--host", default="0.0.0.0", help="bind address (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=7860, help="port (default 7860)")
    parser.add_argument("--share", action="store_true", help="expose a public Gradio link")
    args = parser.parse_args(argv)

    demo = build_demo()
    demo.queue()  # serialize runs; the GPU pipeline is one-at-a-time
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
