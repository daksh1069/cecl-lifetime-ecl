"""Project configuration: config/default.yaml, optionally overridden by a second file."""
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / "config" / "default.yaml"


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, val in over.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(override: str | Path | None = None) -> dict:
    cfg = yaml.safe_load(DEFAULT.read_text())
    if override:
        path = Path(override)
        if not path.is_absolute():
            path = ROOT / path
        cfg = _merge(cfg, yaml.safe_load(path.read_text()) or {})
    return cfg


def path(cfg: dict, name: str) -> Path:
    """Absolute path for an entry of cfg['paths']; the directory is created."""
    p = ROOT / cfg["paths"][name]
    p.mkdir(parents=True, exist_ok=True)
    return p
