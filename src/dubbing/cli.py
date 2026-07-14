"""Command-line entry point: ``dubbing run config/job.yaml [--tts-provider azure]``."""

from __future__ import annotations

import argparse
import sys

from dubbing.config import load_job
from dubbing.pipeline import run


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dubbing")
    sub = parser.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="Run the full pipeline for a job config.")
    run_p.add_argument("config", help="Path to job YAML.")
    run_p.add_argument(
        "--tts-provider",
        help="Override the TTS provider from config (e.g. azure) to A/B vendors.",
    )
    run_p.add_argument(
        "--translation-provider",
        help="Override the translation provider from config.",
    )

    args = parser.parse_args(argv)

    if args.command == "run":
        job = load_job(args.config)
        if args.tts_provider:
            job.providers.tts = args.tts_provider
        if args.translation_provider:
            job.providers.translation = args.translation_provider
        ctx = run(job)
        print(f"Done. Output: {ctx.output_path}")
        for kind, path in ctx.subtitle_paths.items():
            print(f"  {kind}: {path}")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
