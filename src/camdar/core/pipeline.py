from __future__ import annotations
"""Pipeline registry, dynamic stage loading, and sequential stage execution."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type
import importlib

from .types import FrameState


class Stage:
    """
    Base class for all pipeline stages.

    Contract:
      - process() takes a FrameState and returns a FrameState (may mutate or copy).
      - stages should attach outputs to FrameState.debug and/or PersonState fields.
    """

    name: str = "stage"

    def __init__(self, **kwargs: Any) -> None:
        self.cfg = kwargs

    def setup(self) -> None:
        """Optional hook: load models, warm up, allocate buffers."""
        return

    def process(self, state: FrameState) -> FrameState:
        """Override in subclasses."""
        raise NotImplementedError

    def teardown(self) -> None:
        """Optional hook: release resources."""
        return


# -----------------------------
# Registry
# -----------------------------
_STAGE_REGISTRY: Dict[str, Type[Stage]] = {}

_STAGES_IMPORTED = False


def ensure_stages_imported() -> None:
    """
    Lazily import camdar.stages to populate the stage registry.
    This avoids circular imports during module initialization.
    """
    global _STAGES_IMPORTED
    if _STAGES_IMPORTED:
        return

    import importlib

    importlib.import_module("camdar.stages.pose")
    importlib.import_module("camdar.viz.overlay")
    importlib.import_module("camdar.stages.tracking")
    importlib.import_module("camdar.stages.sequence")
    importlib.import_module("camdar.stages.posture")
    importlib.import_module("camdar.stages.status")

    _STAGES_IMPORTED = True


def register_stage(key: str):
    """
    Decorator to register a stage under a string key.

    Usage:
      @register_stage("pose.yolo")
      class YoloPoseStage(Stage): ...
    """

    def _wrap(cls: Type[Stage]) -> Type[Stage]:
        if key in _STAGE_REGISTRY:
            raise ValueError(f"Stage key already registered: {key}")
        _STAGE_REGISTRY[key] = cls
        return cls

    return _wrap


def get_registered_stage(key: str) -> Optional[Type[Stage]]:
    return _STAGE_REGISTRY.get(key)


# -----------------------------
# Dynamic import resolver
# -----------------------------
def resolve_stage_class(spec: str) -> Type[Stage]:
    """
    Resolve a stage from:
      1) registry key: "pose.yolo"
      2) python path: "camdar.stages.pose.yolo_pose:YoloPoseStage"

    Returns the Stage class.
    """
    ensure_stages_imported()

    # 1) registry lookup
    reg = get_registered_stage(spec)
    if reg is not None:
        return reg

    # 2) dynamic import: module_path:ClassName
    if ":" not in spec:
        raise ValueError(
            f"Unknown stage spec '{spec}'. "
            f"Use a registered key or 'module.path:ClassName'."
        )

    module_path, class_name = spec.split(":", 1)
    mod = importlib.import_module(module_path)
    cls = getattr(mod, class_name, None)
    if cls is None:
        raise ImportError(
            f"Could not find class '{class_name}' in module '{module_path}'"
        )

    if not issubclass(cls, Stage):
        raise TypeError(f"Resolved class '{class_name}' is not a Stage")
    return cls


# -----------------------------
# Pipeline runner
# -----------------------------
@dataclass
class Pipeline:
    stages: List[Stage]

    def setup(self) -> None:
        for s in self.stages:
            s.setup()

    def process(self, state: FrameState) -> FrameState:
        for s in self.stages:
            state = s.process(state)
        return state

    def teardown(self) -> None:
        for s in self.stages:
            s.teardown()


def build_pipeline_from_config(cfg: Dict[str, Any]) -> Pipeline:
    """
    Build a pipeline from a config dict.

    Expected config format (YAML -> dict):
      pipeline:
        stages:
          - name: "pose"
            type: "camdar.stages.pose.yolo_pose:YoloPoseStage"
            enabled: true
            params: { model_path: "...", conf: 0.25 }
          - name: "track"
            type: "tracking.bytetrack"
            enabled: true
            params: { ... }

    Notes:
      - "type" can be a registry key OR module:Class path.
      - "params" are passed to the stage constructor.
      - if "enabled" is missing, assumed True.
    """
    pipe_cfg = cfg.get("pipeline", cfg)
    stage_cfgs = pipe_cfg.get("stages", [])
    if not isinstance(stage_cfgs, list) or not stage_cfgs:
        raise ValueError("Config missing pipeline.stages list")

    stages: List[Stage] = []
    for i, sc in enumerate(stage_cfgs):
        if not isinstance(sc, dict):
            raise ValueError(f"Stage entry at index {i} must be a dict")

        enabled = sc.get("enabled", True)
        if not enabled:
            continue

        spec = sc.get("type")
        if not spec:
            raise ValueError(f"Stage at index {i} missing 'type'")

        params = sc.get("params", {}) or {}
        if not isinstance(params, dict):
            raise ValueError(f"Stage params for '{spec}' must be a dict")

        StageCls = resolve_stage_class(spec)
        stage = StageCls(**params)

        # allow overriding the stage instance name
        if "name" in sc and isinstance(sc["name"], str):
            stage.name = sc["name"]

        stages.append(stage)
        print(
            f"[Pipeline] Added stage '{stage.name}' of type '{spec}' with params {params}"
        )

    return Pipeline(stages=stages)
