"""Load a :class:`Job` from a YAML config file, with light path handling."""

from __future__ import annotations

from pathlib import Path

import yaml

from dubbing.models import Job


def load_job(config_path: str | Path) -> Job:
    """Parse a job YAML into a validated :class:`Job`.

    Relative ``input_path``/``work_dir``/``glossary_path`` are resolved against the
    config file's directory so a job file is portable.
    """
    config_path = Path(config_path).resolve()
    data = yaml.safe_load(config_path.read_text()) or {}

    base = config_path.parent
    for key in ("input_path", "work_dir", "glossary_path"):
        if data.get(key):
            data[key] = str((base / data[key]).resolve())

    return Job.model_validate(data)


def load_glossary(path: Path | None) -> dict[str, str]:
    """Load a term -> preferred fr-CA rendering map. Missing/empty -> ``{}``."""
    if path is None or not Path(path).exists():
        return {}
    raw = yaml.safe_load(Path(path).read_text()) or {}
    terms = raw.get("terms", raw)  # allow either {terms: {...}} or a flat map
    return {str(k): str(v) for k, v in terms.items()}
