from __future__ import annotations
"""Configuration loading and path resolution helpers for CAMDAR."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional
import os

import yaml


@dataclass(frozen=True)
class AppPaths:
    """Canonical filesystem locations used by the runtime."""
    repo_root: Path
    configs_dir: Path
    assets_dir: Path
    logs_dir: Path


def find_repo_root(start: Optional[Path] = None) -> Path:
    """
    Find repo root by walking upward until we find configs/ and src/.
    """
    here = (start or Path.cwd()).resolve()
    for p in [here, *here.parents]:
        if (p / "src").exists() and (p / "configs").exists():
            return p
    # fallback: current dir
    return here


def load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file and enforce a dictionary top-level shape."""
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a dict at top level: {path}")
    return data


def deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge dict b into dict a (recursive) and return new dict.
    """
    out = dict(a)
    for k, v in b.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def resolve_paths(repo_root: Optional[Path] = None) -> AppPaths:
    """Resolve known project paths from repo root."""
    root = repo_root or find_repo_root()
    return AppPaths(
        repo_root=root,
        configs_dir=root / "configs",
        assets_dir=root / "assets",
        logs_dir=root / "logs",
    )


def apply_defaults(cfg: Dict[str, Any], paths: AppPaths) -> Dict[str, Any]:
    """
    Apply safe defaults and inject resolved paths.
    """
    defaults: Dict[str, Any] = {
        "app": {
            "camera_id": "cam_01",
            "display": False,
            "write_annotated": False,
            "annotated_out": str(paths.logs_dir / "annotated" / "out.mp4"),
            "log_events": True,
            "events_path": str(paths.logs_dir / "events.jsonl"),
        },
        "runtime": {
            "device": os.environ.get("CAMDAR_DEVICE", "cpu"),
        },
        "paths": {
            "repo_root": str(paths.repo_root),
            "configs_dir": str(paths.configs_dir),
            "assets_dir": str(paths.assets_dir),
            "logs_dir": str(paths.logs_dir),
        },
    }
    return deep_merge(defaults, cfg)


def apply_realtime_profile(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Optionally apply realtime tuning presets to pose stage parameters.

    Supported values under app.realtime_profile:
      - "balanced"   -> moderate speedup
      - "aggressive" -> maximum speed focus
      - "quality"    -> quality-biased while still explicit
      - true         -> alias for "balanced"
    """
    app_cfg = cfg.get("app", {})
    profile = app_cfg.get("realtime_profile", None)
    if profile in (None, False, "", "off", "none"):
        return cfg

    if profile is True:
        profile = "balanced"
    profile = str(profile).lower()

    presets: Dict[str, Dict[str, Any]] = {
        "balanced": {"conf": 0.35, "max_det": 15, "inference_stride": 2},
        "aggressive": {"conf": 0.45, "max_det": 10, "inference_stride": 3},
        "quality": {"conf": 0.30, "max_det": 25, "inference_stride": 1},
    }
    if profile not in presets:
        raise ValueError(
            f"Unknown app.realtime_profile '{profile}'. "
            f"Expected one of: {', '.join(sorted(presets))}"
        )

    stage_cfgs = cfg.get("pipeline", {}).get("stages", [])
    if not isinstance(stage_cfgs, list):
        return cfg

    for stage in stage_cfgs:
        if not isinstance(stage, dict):
            continue
        spec = str(stage.get("type", ""))
        name = str(stage.get("name", ""))
        if spec == "pose.yolo" or name == "pose":
            params = stage.setdefault("params", {})
            if isinstance(params, dict):
                params.update(presets[profile])

    app_cfg["realtime_profile_applied"] = profile
    cfg["app"] = app_cfg
    return cfg


def load_config(config_path: str, repo_root: Optional[str] = None) -> Dict[str, Any]:
    """Load user config and merge it with runtime defaults."""
    root = Path(repo_root).resolve() if repo_root else None
    paths = resolve_paths(root)
    cfg = load_yaml(
        Path(config_path).resolve()
        if Path(config_path).is_absolute()
        else (paths.repo_root / config_path)
    )
    cfg = apply_defaults(cfg, paths)
    cfg = apply_realtime_profile(cfg)
    return cfg
