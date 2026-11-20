from __future__ import annotations
"""Shared dataclasses that define CAMDAR's frame and person state contracts."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import numpy as np


# -----------------------------
# Basic geometry / primitives
# -----------------------------
BBoxXYXY = Tuple[float, float, float, float]  # (x1, y1, x2, y2)


@dataclass(frozen=True)
class FramePacket:
    """
    A single video frame + minimal metadata.
    image is expected to be a BGR numpy array (OpenCV).
    """

    frame_id: int
    ts: float  # seconds
    camera_id: str
    image: np.ndarray  # HxWx3 uint8 BGR


@dataclass(frozen=True)
class Detection:
    """
    Generic detection output (bbox + confidence).
    """

    bbox_xyxy: BBoxXYXY
    conf: float
    cls: Optional[int] = None  # optional class id
    cls_name: Optional[str] = None


@dataclass(frozen=True)
class Pose:
    """
    Pose keypoints associated with a single person.
    kpts arrays use shape (K, 3): [x, y, conf] per joint.
    """

    kpts_raw: np.ndarray  # (K,3) pixel coords
    kpts_norm: Optional[np.ndarray] = None  # (K,3) normalized coords
    avg_kpt_conf: float = 0.0
    # Optional: store keypoint format metadata
    kpt_format: str = "yolo"  # e.g., "coco17", "yolo", etc.


@dataclass
class Track:
    """
    A tracked target in the scene.
    """

    track_id: int
    bbox_xyxy: BBoxXYXY
    conf: float = 1.0
    # Optional kinematics (future)
    vx: float = 0.0
    vy: float = 0.0
    age: int = 0  # how many frames tracked
    last_seen_frame: int = -1


@dataclass
class Classification:
    """
    Generic classification output (label + probability).
    """

    label: str
    p: float
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PersonState:
    """
    All per-person state for the current frame.
    Stages progressively fill this in.
    """

    track: Track
    pose: Optional[Pose] = None

    # Iteration 1 outputs
    posture: Optional[Classification] = None  # standing/sitting/lying/unknown...
    status: Optional[Classification] = None  # normal/uncertain/emergency

    # Future extensions (leave None until implemented)
    face_bbox_xyxy: Optional[BBoxXYXY] = None
    facial_expression: Optional[Classification] = None

    depth_m: Optional[float] = None  # estimated distance (meters)
    world_xyz: Optional[Tuple[float, float, float]] = None

    action: Optional[Classification] = None  # e.g. fall event, fight, wave, etc.
    radar_assoc: Optional[Dict[str, Any]] = None  # radar fusion metadata

    # Debug / quality signals
    quality: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FrameState:
    """
    All outputs for a given frame.
    This is what your pipeline produces each step.
    """

    frame: FramePacket
    people: List[PersonState] = field(default_factory=list)

    # frame-level metadata (future)
    scene_events: List[Dict[str, Any]] = field(default_factory=list)
    debug: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AlertEvent:
    """
    Higher-level alert emitted by the pipeline (future-ready).
    """

    ts: float
    camera_id: str
    track_id: int
    severity: str  # e.g. "normal", "uncertain", "emergency"
    reason: str  # human-readable
    payload: Dict[str, Any] = field(default_factory=dict)
