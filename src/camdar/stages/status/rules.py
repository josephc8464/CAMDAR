from __future__ import annotations
"""Rule-based status inference using posture and pose confidence signals."""




from typing import Any, Dict
import numpy as np




from camdar.core.pipeline import Stage, register_stage
from camdar.core.types import FrameState, Classification








@register_stage("status.rules")
class StatusRulesStage(Stage):
    """
    Rule-based status classifier.




    Emergency triggers, checked in this priority order:
      1. lying/fallen stable for a minimum duration (existing behavior)
      2. hands_on_stomach / hands_on_neck gesture held for a few consecutive
         frames (debounced, not duration-based -- these are meant to fire
         fast)
      3. torso tilt (from posture.heuristic's person.action) sustained for
         tilt_emergency_seconds while the person's posture is "standing"
         or "sitting" (posture.heuristic uses separate calibrated
         thresholds for each, since natural resting torso angle differs)




    Params:
      emergency_seconds: duration threshold for lying/fallen emergency
      fps_assumed: used to convert seconds->frames (good enough for v1)
      min_pose_conf: if avg keypoint confidence below this -> uncertain
      min_posture_p: if posture probability below this -> uncertain
      stable_fraction: fraction of recent frames required to be present
      tilt_emergency_seconds: how long a sustained standing tilt must
        persist (real elapsed time, via FramePacket.ts) before it's an
        emergency
      gesture_consecutive_frames: how many consecutive frames
        hands_on_stomach/hands_on_neck must hold before it's an emergency
        (debounce against a single bad-frame detection). Set to 1 for an
        effectively immediate trigger.
    """




    name = "status.rules"




    def __init__(
        self,
        emergency_seconds: float = 2.0,
        fps_assumed: float = 30.0,
        min_pose_conf: float = 0.35,
        min_posture_p: float = 0.45,
        stable_fraction: float = 0.8,
        tilt_emergency_seconds: float = 5.0,
        gesture_consecutive_frames: int = 3,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            emergency_seconds=emergency_seconds,
            fps_assumed=fps_assumed,
            min_pose_conf=min_pose_conf,
            min_posture_p=min_posture_p,
            stable_fraction=stable_fraction,
            tilt_emergency_seconds=tilt_emergency_seconds,
            gesture_consecutive_frames=gesture_consecutive_frames,
            **kwargs,
        )
        self.emergency_frames = int(float(emergency_seconds) * float(fps_assumed))
        self.min_pose_conf = float(min_pose_conf)
        self.min_posture_p = float(min_posture_p)
        self.stable_fraction = float(stable_fraction)
        self.tilt_emergency_seconds = float(tilt_emergency_seconds)
        self.gesture_consecutive_frames = int(gesture_consecutive_frames)




        # Per-track state, persists across frames on the stage instance.
        # track_id -> frame timestamp (state.frame.ts) when continuous
        # standing-tilt began. Popped whenever tilt stops.
        self._tilt_start_ts: Dict[int, float] = {}
        # track_id -> consecutive-frame count of a hand gesture being held.
        # Reset to 0 whenever the gesture isn't seen.
        self._gesture_streak: Dict[int, int] = {}




    def process(self, state: FrameState) -> FrameState:
        """Assign `normal`/`uncertain`/`emergency` status per person."""
        for person in state.people:
            posture = person.posture
            pose = person.pose
            action = person.action




            avg_kpt_conf = 0.0 if pose is None else float(pose.avg_kpt_conf)
            posture_p = 0.0 if posture is None else float(posture.p)
            posture_label = "unknown" if posture is None else posture.label




            track_id = person.track.track_id if person.track is not None else None




            # Quality gate
            if avg_kpt_conf < self.min_pose_conf or posture_p < self.min_posture_p:
                person.status = Classification(label="uncertain", p=0.6)
                # Don't let stale timers persist through low-quality frames --
                # a gap in good data shouldn't silently count toward duration.
                if track_id is not None:
                    self._tilt_start_ts.pop(track_id, None)
                    self._gesture_streak[track_id] = 0
                continue




            # ----------------------------------------------------------
            # 1. Lying/fallen -- existing duration-based logic
            # ----------------------------------------------------------
            if posture_label == "lying/fallen":
                seq = person.quality.get("pose_seq", None)
                if isinstance(seq, np.ndarray) and seq.ndim == 2:
                    window = seq.shape[0]
                    need = min(self.emergency_frames, window)
                    if window >= need:
                        person.status = Classification(label="emergency", p=0.9)
                    else:
                        person.status = Classification(label="uncertain", p=0.7)
                else:
                    person.status = Classification(label="uncertain", p=0.7)
                continue




            # ----------------------------------------------------------
            # 2. Hand gestures (hands on chest / hands on neck) -- debounced
            # ----------------------------------------------------------
            action_label = action.label if action is not None else "none"
            action_extras = action.extras if action is not None else {}


            if action_label in ("hands_on_chest", "hands_on_neck"):
                streak = self._gesture_streak.get(track_id, 0) + 1 if track_id is not None else self.gesture_consecutive_frames
                if track_id is not None:
                    self._gesture_streak[track_id] = streak


                if streak >= self.gesture_consecutive_frames:
                    person.status = Classification(
                        label="emergency", p=0.9,
                        extras={"reason": action_label, "streak": streak}
                    )
                else:
                    person.status = Classification(
                        label="uncertain", p=0.6,
                        extras={"reason": action_label, "streak": streak}
                    )
                continue
            else:
                if track_id is not None:
                    self._gesture_streak[track_id] = 0




            # ----------------------------------------------------------
            # 3. Sustained tilt (standing or sitting) -- duration-based
            #    via frame.ts
            # ----------------------------------------------------------
            is_tilted = bool(action_extras.get("is_tilted", False))




            if posture_label in ("standing", "sitting") and is_tilted:
                ts = state.frame.ts
                start = self._tilt_start_ts.get(track_id) if track_id is not None else None




                if start is None:
                    if track_id is not None:
                        self._tilt_start_ts[track_id] = ts
                    person.status = Classification(
                        label="uncertain", p=0.6,
                        extras={"reason": f"tilt_building_{posture_label}", "elapsed_s": 0.0}
                    )
                    continue




                elapsed = ts - start
                if elapsed >= self.tilt_emergency_seconds:
                    person.status = Classification(
                        label="emergency", p=0.85,
                        extras={"reason": f"sustained_tilt_{posture_label}", "elapsed_s": elapsed}
                    )
                else:
                    person.status = Classification(
                        label="uncertain", p=0.6,
                        extras={"reason": f"tilt_building_{posture_label}", "elapsed_s": elapsed}
                    )
                continue
            else:
                if track_id is not None:
                    self._tilt_start_ts.pop(track_id, None)




            # Default normal
            person.status = Classification(label="normal", p=0.9)




        return state













