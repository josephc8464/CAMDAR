from __future__ import annotations
"""YOLO-based human pose stage that populates PersonState pose/track fields."""

import cv2
import copy

# Some headless OpenCV builds omit HighGUI symbols expected by upstream libs.
if not hasattr(cv2, "imshow"):
    cv2.imshow = lambda *args, **kwargs: None
if not hasattr(cv2, "waitKey"):
    cv2.waitKey = lambda *args, **kwargs: -1
if not hasattr(cv2, "destroyAllWindows"):
    cv2.destroyAllWindows = lambda *args, **kwargs: None

from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from pathlib import Path
from ultralytics import YOLO

from camdar.core.pipeline import Stage, register_stage
from camdar.core.types import (
    BBoxXYXY,
    Classification,
    Detection,
    FrameState,
    PersonState,
    Pose,
    Track,
)
from .normalize import normalize_kpts_center_scale


def _to_bbox_xyxy(xyxy: np.ndarray) -> BBoxXYXY:
    """Convert raw array bbox output into tuple form used by core types."""
    x1, y1, x2, y2 = xyxy.tolist()
    return float(x1), float(y1), float(x2), float(y2)


@register_stage("pose.yolo")
class YoloPoseStage(Stage):
    """
    YOLOv11 pose inference stage.

    Params:
      model_path: path to YOLO pose weights (e.g., assets/models/yolo11x-pose.pt)
      conf: detection confidence threshold
      iou: NMS IoU threshold
      max_det: maximum detections per frame
      kpt_conf_thresh: keypoint conf threshold for normalization visibility
      device: "cpu", "cuda:0", etc (optional)
      classes: optional list of class ids to keep (usually person class only)
    """

    name = "pose.yolo"

    def __init__(
        self,
        model_path: str,
        repo_root: str = ".",
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 50,
        kpt_conf_thresh: float = 0.2,
        inference_stride: int = 1,
        track_with_bytetrack: bool = False,
        tracker_cfg: str = "bytetrack.yaml",
        persist_tracks: bool = True,
        device: Optional[str] = None,
        classes: Optional[List[int]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_path=model_path,
            conf=conf,
            iou=iou,
            max_det=max_det,
            kpt_conf_thresh=kpt_conf_thresh,
            inference_stride=inference_stride,
            track_with_bytetrack=track_with_bytetrack,
            tracker_cfg=tracker_cfg,
            persist_tracks=persist_tracks,
            device=device,
            classes=classes,
            **kwargs,
        )
        self.model_path = model_path
        self.repo_root = repo_root
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self.kpt_conf_thresh = kpt_conf_thresh
        self.inference_stride = max(1, int(inference_stride))
        self.track_with_bytetrack = bool(track_with_bytetrack)
        self.tracker_cfg = tracker_cfg
        self.persist_tracks = bool(persist_tracks)
        self.device = device
        self.classes = classes

        self._model: Optional[YOLO] = None
        self._last_people: List[PersonState] = []

    def setup(self) -> None:
        """Load YOLO weights and validate model file integrity."""
        p = Path(self.model_path)
        if not p.is_absolute():
            p = Path(self.repo_root).resolve() / p
        # Guard against corrupted weights
        if p.exists() and p.suffix == ".pt" and p.stat().st_size < 1_000_000:
            raise RuntimeError(
                f"Model file looks corrupted: {p} ({p.stat().st_size} bytes)"
            )
        self._model = YOLO(str(p))

    def process(self, state: FrameState) -> FrameState:
        """Run pose detection and populate per-person state for downstream stages."""
        assert self._model is not None, "YoloPoseStage not setup()"

        frame = state.frame.image

        # Skip expensive model calls on intermediate frames and reuse latest result.
        if (
            self.inference_stride > 1
            and (state.frame.frame_id % self.inference_stride) != 0
            and self._last_people
        ):
            state.people = copy.deepcopy(self._last_people)
            state.debug["pose"] = {"num": len(state.people), "skipped": True}
            return state

        # Ultralytics returns a list of Results; single image -> len=1.
        # When enabled, run ByteTrack here so we avoid a second model pass.
        if self.track_with_bytetrack:
            results = self._model.track(
                frame,
                persist=self.persist_tracks,
                tracker=self.tracker_cfg,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                device=self.device,
                classes=self.classes,
                verbose=False,
            )
        else:
            results = self._model.predict(
                frame,
                conf=self.conf,
                iou=self.iou,
                max_det=self.max_det,
                device=self.device,
                classes=self.classes,
                verbose=False,
            )
        r = results[0]

        if r.boxes is None or len(r.boxes) == 0:
            state.people = []
            state.debug["pose"] = {"num": 0}
            return state

        boxes_xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        track_ids = None
        if getattr(r.boxes, "id", None) is not None:
            track_ids = r.boxes.id.cpu().numpy()

        # keypoints: shape (N, K, 3) in pixel coords
        kpts_xyc = None
        if r.keypoints is not None and r.keypoints.data is not None:
            kpts_xyc = r.keypoints.data.cpu().numpy()

        people: List[PersonState] = []

        for i in range(len(boxes_xyxy)):
            bbox = _to_bbox_xyxy(boxes_xyxy[i])
            det_conf = float(confs[i])

            raw_kpts = None
            if kpts_xyc is not None and i < len(kpts_xyc):
                raw_kpts = kpts_xyc[i].astype(np.float32)  # (K,3)

            norm_kpts = None
            avg_kpt_conf = 0.0
            if raw_kpts is not None:
                norm_kpts, avg_kpt_conf = normalize_kpts_center_scale(
                    raw_kpts, conf_thresh=self.kpt_conf_thresh
                )

            pose = Pose(
                kpts_raw=(
                    raw_kpts
                    if raw_kpts is not None
                    else np.zeros((0, 3), dtype=np.float32)
                ),
                kpts_norm=norm_kpts,
                avg_kpt_conf=float(avg_kpt_conf),
                kpt_format="yolo",
            )

            tid = int(track_ids[i]) if track_ids is not None and i < len(track_ids) else -1
            track = Track(track_id=tid, bbox_xyxy=bbox, conf=det_conf)

            ps = PersonState(
                track=track,
                pose=pose,
                quality={
                    "det_conf": det_conf,
                    "avg_kpt_conf": float(avg_kpt_conf),
                },
            )
            people.append(ps)

        state.people = people
        self._last_people = copy.deepcopy(people)
        state.debug["pose"] = {"num": len(people), "skipped": False}
        return state
