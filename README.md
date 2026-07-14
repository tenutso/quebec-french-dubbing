# Quebec French Dubbing Pipeline

Turns a multi-speaker webinar video into a **high-quality Québec French (fr-CA) dub +
industry-standard subtitles** for course content, webinars, and instructional videos.

**Cost strategy:** premium dollars go only where they move quality most — **translation
register** (Claude) and **TTS prosody/voice** (pluggable). Everything else runs on
**open-source models** (ffmpeg, Demucs, pyannote, WhisperX) on **on-demand cloud GPU**, so
you pay GPU-seconds instead of per-minute SaaS fees.

## Pipeline

```
1. Ingest & probe      video -> ASR wav (16k mono) + full wav (48k) + metadata   [ffmpeg]
2. Source separation   vocals stem + background (music/SFX) stem                 [Demucs, GPU]
3. Diarization         who-spoke-when speaker segments                           [pyannote, GPU]
4. ASR + alignment     word-level timestamps + speaker labels                    [WhisperX, GPU]
5. Cue construction    words -> standards-compliant caption/dub cues             [subtitle_rules]
6. Translation         EN -> fr-CA, subtitle + dub variants                      [Claude, premium]
7. Subtitle authoring  SRT + WebVTT (Netflix fr-CA style)                        [pysubs2]
8. TTS synthesis       per-speaker cloned/preset fr-CA voice, one clip per cue   [pluggable, premium]
9. Time-fit            fit each clip to its cue slot (±8%)                        [ffmpeg/rubberband]
10. Mix & master       dub bus + background stem, EBU R128 normalize             [numpy/pyloudnorm]
11. Mux & package      French audio track + subtitle sidecars -> MP4             [ffmpeg]
```

## Pluggable providers

TTS and translation are Strategy interfaces resolved by name from a registry
(`src/dubbing/providers/`), so any vendor with fr-CA support drops in:

- **TTS:** `elevenlabs` (cloning, default), `azure` (fr-CA neural, strongest native prosody),
  plus `google`/`polly` slots. A provider that can't produce `fr-CA` is rejected at job start.
- **Translation:** `claude` (Québec register + glossary), pluggable.

Select per job in `config/job.yaml`, or A/B on the CLI: `dubbing run job.yaml --tts-provider azure`.

## Quickstart

```bash
make install          # core + translation + tts SDKs
make install-audio    # mixing/mastering deps
make test             # 34 tests (subtitle rules, cues, providers, time-fit/mix, e2e)

# Real GPU stages (on a GPU box or via Modal):
make install-gpu
make deploy-gpu       # deploy the Modal app; set DUBBING_GPU_BACKEND=modal to dispatch
```

## Configuration

Copy `config/job.example.yaml` to `config/job.yaml` and edit. Secrets via env:
`ANTHROPIC_API_KEY`, `ELEVENLABS_API_KEY`, `AZURE_SPEECH_KEY`/`AZURE_SPEECH_REGION`,
`HF_TOKEN` (pyannote). GPU backend: `DUBBING_GPU_BACKEND=local|modal`.

## Layout

```
src/dubbing/
  pipeline.py            stage runner
  models.py              pydantic artifacts
  subtitle_rules.py      Netflix fr-CA standards engine
  ffmpeg_utils.py        probe / extract / loudness helpers
  gpu_runners.py         Demucs / pyannote / WhisperX (OSS)
  modal_app.py           on-demand GPU functions
  stages/                one module per pipeline stage
  providers/             TTS + translation Strategy interfaces + implementations
```
