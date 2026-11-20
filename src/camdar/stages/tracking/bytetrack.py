from __future__ import annotations
"""ByteTrack-based stage that attaches persistent IDs to detections."""

import cv2

# Some headless OpenCV builds omit HighGUI symbols expected by upstream libs.
if not hasattr(cv2, "imshow"):
    cv2.imshow = lambda *args, **kwargs: None
if not hasattr(cv2, "waitKey"):
    cv2.waitKey = lambda *args, **kwargs: -1
if not hasattr(cv2, "destroyAllWindows"):
    cv2.destroyAllWindows = lambda *args, **kwargs: None

from typing import Any
from ultralytics import YOLO

from camdar.core.pipeline import Stage, register_stage
from camdar.core.types import FrameState, Track


@register_stage("tracking.bytetrack")
class ByteTrackStage(Stage):
    name = "tracking.bytetrack"

    def __init__(self, model_path: str, conf: float = 0.25, **kwargs: Any) -> None:
        super().__init__(model_path=model_path, conf=conf, **kwargs)
        self.model_path = model_path
        self.conf = conf
        self.model = None

    def setup(self):
        """Initialize tracking backend (Ultralytics + ByteTrack config)."""
        # Reuse YOLO model for tracking
        self.model = YOLO(self.model_path)

    def process(self, state: FrameState) -> FrameState:
        """Update current frame people with tracked IDs and refined bboxes."""
        if state.frame.image is None:
            return state

        results = self.model.track(
            state.frame.image,
            persist=True,
            tracker="bytetrack.yaml",
            conf=self.conf,
            verbose=False,
        )

        r = results[0]

        if r.boxes is None or r.boxes.id is None:
            state.people = []
            return state

        boxes = r.boxes.xyxy.cpu().numpy()
        ids = r.boxes.id.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()

        people = []

        for bbox, track_id, det_conf in zip(boxes, ids, confs):
            x1, y1, x2, y2 = bbox.tolist()

            people.append(
                Track(
                    track_id=int(track_id),
                    bbox_xyxy=(x1, y1, x2, y2),
                    conf=float(det_conf),
                )
            )

        # Update existing PersonState objects if pose already exists
        if state.people:
            for i, person in enumerate(state.people):
                if i < len(people):
                    person.track = people[i]
        else:
            # If pose stage not used, create minimal states
            from camdar.core.types import PersonState

            state.people = [PersonState(track=t) for t in people]

        return state
