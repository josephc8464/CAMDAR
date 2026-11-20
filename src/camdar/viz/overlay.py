from __future__ import annotations
"""Overlay stage that draws track, posture, and status annotations on frames."""

from typing import Any, Tuple
import cv2
import numpy as np

from camdar.core.pipeline import Stage, register_stage
from camdar.core.types import FrameState


def status_color(status: str) -> Tuple[int, int, int]:
    """
    Return a BGR color tuple for a given status string.
    """
    if status == "emergency":
        return (0, 0, 255)  # red
    elif status == "uncertain":
        return (0, 255, 255)  # yellow
    elif status == "normal":
        return (0, 255, 0)  # green
    else:
        return (255, 255, 255)  # white for unknown or other statuses


@register_stage("viz.overlay")
class OverlayStage(Stage):
    """Render person bounding boxes, labels, and optional keypoints."""
    name = "viz.overlay"

    def __init__(
        self, draw_kpts: bool = True, kpt_thresh: float = 0.3, **kwargs: Any
    ) -> None:
        super().__init__(draw_kpts=draw_kpts, kpt_thresh=kpt_thresh, **kwargs)
        self.draw_kpts = draw_kpts
        self.kpt_thresh = float(kpt_thresh)

    def process(self, state: FrameState) -> FrameState:
        """Mutate the frame image in-place with visualization overlays."""
        img = state.frame.image

        for person in state.people:
            tid = person.track.track_id
            det_conf = person.track.conf
            posture = person.posture.label if person.posture else "?"
            status = person.status.label if person.status else "?"
            x1, y1, x2, y2 = map(int, person.track.bbox_xyxy)
            thickness = 5 if (status or "") == "emergency" else 2

            color = status_color(status)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

            label = f"ID {tid} | {det_conf:.2f} | {posture} | {status}"
            tx, ty = x1, y2 + 20
            (label_w, label_h), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
            )
            cv2.rectangle(
                img, (tx, ty - label_h - 6), (tx + label_w + 6, ty + 4), color, -1
            )
            cv2.putText(
                img,
                label,
                (tx + 3, ty - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 0),
                2,
            )

            if self.draw_kpts and person.pose is not None:
                kpts = person.pose.kpts_raw
                if kpts is not None and kpts.size:
                    for x, y, c in kpts:
                        if c >= self.kpt_thresh:
                            cv2.circle(img, (int(x), int(y)), 3, (255, 255, 255), -1)

        state.debug["viz"] = {"drawn": len(state.people)}
        return state
