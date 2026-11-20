from __future__ import annotations
"""Build fixed-length per-track pose sequences for temporal reasoning stages."""

from collections import deque
from typing import Any, Deque, Dict, List, Optional

import numpy as np

from camdar.core.pipeline import Stage, register_stage
from camdar.core.types import FrameState


@register_stage("sequence.builder")
class SequenceBuilderStage(Stage):
    """Maintain rolling pose vectors keyed by track ID."""
    name = "sequence.builder"

    def __init__(
        self, window: int = 16, max_gap: int = 3, use_norm: bool = True, **kwargs: Any
    ) -> None:
        super().__init__(window=window, max_gap=max_gap, use_norm=use_norm, **kwargs)
        self.window = int(window)
        self.max_gap = int(max_gap)
        self.use_norm = bool(use_norm)

        self.buffers: Dict[int, Deque[np.ndarray]] = {}
        self.last_seen_frame: Dict[int, int] = {}

    def _get_pose_vec(self, person) -> Optional[np.ndarray]:
        """Extract a flattened pose vector for sequence buffering."""
        if person.pose is None:
            return None

        k = (
            person.pose.kpts_norm
            if (self.use_norm and person.pose.kpts_norm is not None)
            else person.pose.kpts_raw
        )
        if k is None or k.size == 0:
            return None

        # flatten (K,3) -> (K*3,)
        return k.astype(np.float32).reshape(-1)

    def process(self, state: FrameState) -> FrameState:
        """Update per-track sequence buffers and annotate quality metadata."""
        fid = state.frame.frame_id

        for person in state.people:
            tid = person.track.track_id
            if tid is None or tid < 0:
                continue

            # reset buffer if gap too large
            if (
                tid in self.last_seen_frame
                and (fid - self.last_seen_frame[tid]) > self.max_gap
            ):
                self.buffers.pop(tid, None)

            if tid not in self.buffers:
                self.buffers[tid] = deque(maxlen=self.window)

            vec = self._get_pose_vec(person)
            if vec is not None:
                self.buffers[tid].append(vec)

            self.last_seen_frame[tid] = fid

            seq_len = len(self.buffers[tid])
            person.quality["seq_len"] = seq_len
            person.quality["seq_ready"] = seq_len >= self.window

            # Optional: attach sequence for downstream stages
            # (Keeps contract simple; posture stage can read it from quality)
            if person.quality["seq_ready"]:
                # store as (window, D)
                person.quality["pose_seq"] = np.stack(list(self.buffers[tid]), axis=0)

        return state
