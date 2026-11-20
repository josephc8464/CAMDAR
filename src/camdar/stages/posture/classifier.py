

from __future__ import annotations
"""Heuristic posture classification from per-person keypoint geometry.


Uses ONLY perspective-independent signals for posture classification that
compare leg segments to each other. Works regardless of person height,
distance from camera, or camera angle distortion.


Also populates person.action with tilt / hand-gesture signals used by
status.rules to detect emergencies while nominally standing:
  - is_tilted: torso (shoulder-mid -> hip-mid) deviates from vertical by
    more than tilt_angle_thresh degrees.
  - hands_chest / hands_neck: both wrists close to the chest or
    neck region, gated by both distance AND vertical position.
"""


from typing import Any, Optional, Tuple
import time
import numpy as np


from camdar.core.pipeline import Stage, register_stage
from camdar.core.types import FrameState, Classification


# ============================================================
# DEBUG CONFIG - Set to False to disable keypoint dumps
# ============================================================
DEBUG_DUMP_ENABLED = False
DEBUG_DUMP_INTERVAL = 2.0  # seconds between dumps (avoid console flood)
_last_dump_time = 0.0


# ============================================================
# SIDE-VIEW GUARD
# When shoulder_width is below this threshold, the person is
# viewed from the side (or partially rotated) and some signals
# become unreliable.
# ============================================================
SIDE_VIEW_SHOULDER_THRESH = 100.0




def _angle(a, b, c):
    """Return angle ABC in degrees."""
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    cosang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    return float(np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))))




def _dist(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.linalg.norm(a - b))




def _midpoint(a, b):
    a, b = np.array(a), np.array(b)
    return (a + b) / 2.0




def _conf(kpts, idx, conf_thresh=0.2):
    """Check if a keypoint has sufficient confidence."""
    if kpts is None or idx >= kpts.shape[0]:
        return False
    return float(kpts[idx, 2]) >= conf_thresh




def _span_xy(
    kpts_xyc: np.ndarray, conf_thresh: float = 0.2
) -> Optional[Tuple[float, float]]:
    """Return (x_span, y_span) of visible keypoints."""
    if kpts_xyc is None or kpts_xyc.size == 0:
        return None
    conf = kpts_xyc[:, 2]
    vis = conf >= conf_thresh
    if int(vis.sum()) < 4:
        return None
    pts = kpts_xyc[vis, :2]
    x_span = float(pts[:, 0].max() - pts[:, 0].min())
    y_span = float(pts[:, 1].max() - pts[:, 1].min())
    return x_span, y_span




def _dump_keypoints(k, person_id="?", action_extras=None):
    """Print all keypoint positions for debugging."""
    global _last_dump_time
    now = time.time()
    if now - _last_dump_time < DEBUG_DUMP_INTERVAL:
        return
    _last_dump_time = now


    labels = [
        "nose", "L_eye", "R_eye", "L_ear", "R_ear",
        "L_shoulder", "R_shoulder", "L_elbow", "R_elbow",
        "L_wrist", "R_wrist", "L_hip", "R_hip",
        "L_knee", "R_knee", "L_ankle", "R_ankle"
    ]


    print(f"\n{'='*60}")
    print(f"  KEYPOINT DUMP - Person {person_id}")
    print(f"  Time: {time.strftime('%H:%M:%S')}")
    print(f"{'='*60}")
    print(f"  {'Name':<12s}  {'X':>8s}  {'Y':>8s}  {'Conf':>6s}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*6}")


    for i, name in enumerate(labels):
        if i < k.shape[0]:
            x, y, c = float(k[i][0]), float(k[i][1]), float(k[i][2])
            marker = " *" if c < 0.2 else ""
            print(f"  {name:<12s}  {x:8.1f}  {y:8.1f}  {c:6.2f}{marker}")
        else:
            print(f"  {name:<12s}  {'N/A':>8s}  {'N/A':>8s}  {'N/A':>6s}")


    # Print derived metrics
    print(f"\n  --- Derived Metrics ---")
    if k.shape[0] > 0 and float(k[0][2]) >= 0.2:
        print(f"  Nose Y (absolute):     {float(k[0][1]):8.1f}")
    if k.shape[0] > 6 and float(k[5][2]) >= 0.2 and float(k[6][2]) >= 0.2:
        sh_mid_y = (float(k[5][1]) + float(k[6][1])) / 2.0
        sh_width = _dist(k[5][:2], k[6][:2])
        print(f"  Shoulder mid Y:        {sh_mid_y:8.1f}")
        print(f"  Shoulder width:        {sh_width:8.1f}")
    if k.shape[0] > 12 and float(k[11][2]) >= 0.2 and float(k[12][2]) >= 0.2:
        hip_mid_y = (float(k[11][1]) + float(k[12][1])) / 2.0
        hip_width = _dist(k[11][:2], k[12][:2])
        print(f"  Hip mid Y:             {hip_mid_y:8.1f}")
        print(f"  Hip width:             {hip_width:8.1f}")
    if k.shape[0] > 14 and float(k[13][2]) >= 0.2 and float(k[14][2]) >= 0.2:
        knee_mid_y = (float(k[13][1]) + float(k[14][1])) / 2.0
        print(f"  Knee mid Y:            {knee_mid_y:8.1f}")
    if k.shape[0] > 16 and float(k[15][2]) >= 0.2 and float(k[16][2]) >= 0.2:
        ankle_mid_y = (float(k[15][1]) + float(k[16][1])) / 2.0
        print(f"  Ankle mid Y:           {ankle_mid_y:8.1f}")
    confs = k[:, 2]
    vis = confs >= 0.2
    if vis.sum() > 0:
        pts = k[vis, :2]
        y_min, y_max = float(pts[:, 1].min()), float(pts[:, 1].max())
        x_min, x_max = float(pts[:, 0].min()), float(pts[:, 0].max())
        print(f"  Bbox Y range:          {y_min:8.1f} to {y_max:.1f} (span: {y_max-y_min:.1f})")
        print(f"  Bbox X range:          {x_min:8.1f} to {x_max:.1f} (span: {x_max-x_min:.1f})")


    if action_extras:
        print(f"\n  --- Action / Gesture Signals ---")
        for key in ("tilt_angle", "is_tilted", "chest_dist_ratio", "chest_band_ok",
                    "neck_dist_ratio", "neck_band_ok", "hands_chest", "hands_neck"):
            if key in action_extras:
                print(f"  {key:<20s} {action_extras[key]}")
    print(f"{'='*60}\n")




@register_stage("posture.heuristic")
class PostureClassifierStage(Stage):
    """
    Perspective-independent posture classifier with emergency action detection.


    POSTURE DETECTION:
    Uses signals that remain stable regardless of person height or camera
    angle distortion. Handles both "feet out" and "feet tucked" sitting.


    Signal architecture:
      - shin_to_thigh + knee_position: detect "feet out" sitting
      - knee_angle + body_aspect: detect ALL sitting (including feet tucked)
      - torso orientation + body spans: detect lying/fallen


    ACTION/EMERGENCY DETECTION:
    Populates person.action with single-frame geometric signals:
      - is_tilted: torso deviates from vertical
      - hands_chest: both wrists near chest region
      - hands_neck: both wrists near neck region (NOT face)
    status.rules handles duration/debounce to turn these into emergencies.


    Params:
      conf_thresh: keypoint visibility threshold
      lying_ratio: if x_span / y_span exceeds this -> lying/fallen
      sitting_score_thresh: minimum combined score to classify as sitting
      lying_score_thresh: minimum combined score to classify as lying
      tilt_angle_thresh: degrees of torso deviation from vertical to flag
        is_tilted=True while standing
      tilt_angle_thresh_sitting: same, but used while sitting
      hand_chest_ratio: wrist-to-chest distance / shoulder_width threshold
      hand_neck_ratio: wrist-to-neck distance / shoulder_width threshold
    """


    name = "posture.heuristic"


    def __init__(
        self,
        conf_thresh: float = 0.2,
        lying_ratio: float = 2.5,
        sitting_score_thresh: float = 0.50,
        lying_score_thresh: float = 0.45,
        tilt_angle_thresh: float = 25.0,
        tilt_angle_thresh_sitting: float = 25.0,
        hand_chest_ratio: float = 0.65,
        hand_neck_ratio: float = 0.50,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            conf_thresh=conf_thresh,
            lying_ratio=lying_ratio,
            sitting_score_thresh=sitting_score_thresh,
            lying_score_thresh=lying_score_thresh,
            tilt_angle_thresh=tilt_angle_thresh,
            tilt_angle_thresh_sitting=tilt_angle_thresh_sitting,
            hand_chest_ratio=hand_chest_ratio,
            hand_neck_ratio=hand_neck_ratio,
            **kwargs,
        )
        self.conf_thresh = float(conf_thresh)
        self.lying_ratio = float(lying_ratio)
        self.sitting_score_thresh = float(sitting_score_thresh)
        self.lying_score_thresh = float(lying_score_thresh)
        self.tilt_angle_thresh = float(tilt_angle_thresh)
        self.tilt_angle_thresh_sitting = float(tilt_angle_thresh_sitting)
        self.hand_chest_ratio = float(hand_chest_ratio)
        self.hand_neck_ratio = float(hand_neck_ratio)


    # ================================================================
    # ACTION / EMERGENCY SIGNAL DETECTION
    # ================================================================
    def _detect_action(self, k: np.ndarray, ct: float, posture_label: str = "standing") -> Classification:
        """
        Single-frame tilt / hand-gesture signals.


        Deliberately does NOT decide emergency here -- it only reports raw
        geometric signals (is_tilted, hands_chest, hands_neck) plus a best
        single-label guess for convenience/logging. status.rules is where
        duration/debounce turns these into an actual emergency status,
        since that needs cross-frame state this per-frame stage doesn't have.


        posture_label selects which tilt threshold to use -- sitting and
        standing can have a different natural resting torso angle from an
        elevated, angled-down camera, so they're calibrated independently.
        """
        NOSE = 0
        L_SHOULDER, R_SHOULDER = 5, 6
        L_WRIST, R_WRIST = 9, 10
        L_HIP, R_HIP = 11, 12


        tilt_thresh = (
            self.tilt_angle_thresh_sitting if posture_label == "sitting"
            else self.tilt_angle_thresh
        )


        extras: dict = {"is_tilted": False, "hands_chest": False, "hands_neck": False}


        have_shoulders = _conf(k, L_SHOULDER, ct) and _conf(k, R_SHOULDER, ct)
        have_hips = _conf(k, L_HIP, ct) and _conf(k, R_HIP, ct)
        have_wrists = _conf(k, L_WRIST, ct) and _conf(k, R_WRIST, ct)
        have_nose = _conf(k, NOSE, ct)


        shoulder_mid = _midpoint(k[L_SHOULDER][:2], k[R_SHOULDER][:2]) if have_shoulders else None
        hip_mid = _midpoint(k[L_HIP][:2], k[R_HIP][:2]) if have_hips else None
        shoulder_width = _dist(k[L_SHOULDER][:2], k[R_SHOULDER][:2]) if have_shoulders else None


        # ---- Tilt: angle of shoulder-mid -> hip-mid vector from vertical ----
        if shoulder_mid is not None and hip_mid is not None:
            dx = shoulder_mid[0] - hip_mid[0]
            dy = shoulder_mid[1] - hip_mid[1]
            tilt_angle = float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))
            extras["tilt_angle"] = tilt_angle
            extras["tilt_thresh_used"] = tilt_thresh
            extras["is_tilted"] = tilt_angle > tilt_thresh


        # ---- Hand gestures: wrist proximity to neck / chest points ----
        if (have_wrists and shoulder_mid is not None and hip_mid is not None
                and shoulder_width and shoulder_width > 1e-6):
            l_wrist = k[L_WRIST][:2]
            r_wrist = k[R_WRIST][:2]
            l_wrist_y, r_wrist_y = float(l_wrist[1]), float(r_wrist[1])


            torso_height = abs(hip_mid[1] - shoulder_mid[1])


            # ---- Neck zone ----
            # Anchor point at the shoulder line (base of neck).
            # The vertical band requires wrists to be BELOW the nose
            # and ABOVE the upper shoulder area. This ensures that
            # face touches (hands at/above nose level) do NOT trigger,
            # only actual neck-level hand placement does.
            neck_pt = (
                shoulder_mid[0],
                shoulder_mid[1] - 0.1 * torso_height,
            )
            l_neck_d = _dist(l_wrist, neck_pt) / shoulder_width
            r_neck_d = _dist(r_wrist, neck_pt) / shoulder_width
            extras["neck_dist_ratio"] = (round(l_neck_d, 3), round(r_neck_d, 3))


            # Vertical band: wrist must be BELOW the nose (not on face)
            # and ABOVE mid-torso (not on chest).
            # nose_y is the UPPER boundary — hands must be at or below it.
            # shoulder + 25% torso is the LOWER boundary — below that is chest.
            nose_y_val = float(k[NOSE][1]) if have_nose else shoulder_mid[1] - 0.5 * torso_height
            neck_y_hi = nose_y_val                              # must be at or below nose
            neck_y_lo = shoulder_mid[1] + 0.25 * torso_height   # must be above mid-torso
            l_neck_band = neck_y_hi <= l_wrist_y <= neck_y_lo
            r_neck_band = neck_y_hi <= r_wrist_y <= neck_y_lo
            extras["neck_band_ok"] = (l_neck_band, r_neck_band)


            extras["hands_neck"] = (
                l_neck_d < self.hand_neck_ratio and r_neck_d < self.hand_neck_ratio
                and l_neck_band and r_neck_band
            )


            # ---- Chest zone ----
            # Anchor point at mid-torso (halfway between shoulders and hips)
            chest_pt = (
                shoulder_mid[0] + 0.5 * (hip_mid[0] - shoulder_mid[0]),
                shoulder_mid[1] + 0.5 * (hip_mid[1] - shoulder_mid[1]),
            )
            l_chest_d = _dist(l_wrist, chest_pt) / shoulder_width
            r_chest_d = _dist(r_wrist, chest_pt) / shoulder_width
            extras["chest_dist_ratio"] = (round(l_chest_d, 3), round(r_chest_d, 3))


            # Vertical band: wrist must be below upper shoulder area
            # and above the hip line (chest region, not face/thighs)
            chest_y_hi = shoulder_mid[1] + 0.15 * torso_height
            chest_y_lo = shoulder_mid[1] + 0.65 * torso_height
            l_chest_band = chest_y_hi <= l_wrist_y <= chest_y_lo
            r_chest_band = chest_y_hi <= r_wrist_y <= chest_y_lo
            extras["chest_band_ok"] = (l_chest_band, r_chest_band)


            extras["hands_chest"] = (
                l_chest_d < self.hand_chest_ratio and r_chest_d < self.hand_chest_ratio
                and l_chest_band and r_chest_band
            )


        if extras["hands_neck"]:
            label, p = "hands_on_neck", 0.85
        elif extras["hands_chest"]:
            label, p = "hands_on_chest", 0.80
        elif extras["is_tilted"]:
            label, p = "tilt", 0.70
        else:
            label, p = "none", 0.60


        return Classification(label=label, p=p, extras=extras)


    # ================================================================
    # POSTURE CLASSIFICATION (perspective-independent)
    # ================================================================
    def _classify_posture(self, pose, person_id="?") -> Classification:
        """
        Perspective-independent posture classification.


        Two detection paths for sitting:
        Path A (feet out): shin_to_thigh > 1.1 AND/OR knee_position < 0.48
        Path B (feet tucked): knee_angle < 145° AND body_aspect < 2.8


        Both paths contribute to the same score. The threshold is 0.50,
        requiring at least 2 signals to agree.
        """
        # Prefer normalized keypoints for span ratio check
        k_norm = pose.kpts_norm if pose.kpts_norm is not None else pose.kpts_raw
        spans = _span_xy(k_norm, conf_thresh=self.conf_thresh)


        # Use raw keypoints for geometric analysis
        k = pose.kpts_raw
        if k is None or k.shape[0] < 17:
            if spans is None:
                return Classification(label="unknown", p=0.3)
            x_span, y_span = spans
            if y_span < 1e-6:
                return Classification(label="unknown", p=0.3)
            ratio = x_span / y_span
            if ratio >= self.lying_ratio:
                p = float(min(0.95, 0.65 + 0.1 * (ratio - self.lying_ratio)))
                return Classification(
                    label="lying/fallen", p=p,
                    extras={"method": "xy_ratio", "ratio": ratio}
                )
            return Classification(label="standing", p=0.6, extras={"ratio": ratio})


        # --- COCO keypoint indices ---
        NOSE = 0
        L_SHOULDER, R_SHOULDER = 5, 6
        L_HIP, R_HIP = 11, 12
        L_KNEE, R_KNEE = 13, 14
        L_ANKLE, R_ANKLE = 15, 16


        ct = self.conf_thresh


        # --- Compute body spans ---
        ratio = None
        body_y_span = 0.0
        body_x_span = 0.0
        if spans is not None:
            x_span, y_span = spans
            if y_span > 1e-6:
                ratio = x_span / y_span
                if ratio >= self.lying_ratio:
                    p = float(min(0.95, 0.65 + 0.1 * (ratio - self.lying_ratio)))
                    return Classification(
                        label="lying/fallen", p=p,
                        extras={"method": "xy_ratio", "ratio": ratio}
                    )


        confs_all = k[:, 2]
        vis_all = confs_all >= ct
        if vis_all.sum() > 0:
            pts_all = k[vis_all, :2]
            body_y_span = float(pts_all[:, 1].max() - pts_all[:, 1].min())
            body_x_span = float(pts_all[:, 0].max() - pts_all[:, 0].min())


        # --- Shoulder width (for debug/extras) ---
        shoulder_width = 0.0
        if _conf(k, L_SHOULDER, ct) and _conf(k, R_SHOULDER, ct):
            shoulder_width = _dist(k[L_SHOULDER][:2], k[R_SHOULDER][:2])


        # ============================================================
        # LYING DETECTION
        # ============================================================
        lying_score = 0.0
        lying_extras = {}


        # Signal 1: Torso more horizontal than vertical
        if (_conf(k, L_SHOULDER, ct) and _conf(k, R_SHOULDER, ct) and
                _conf(k, L_HIP, ct) and _conf(k, R_HIP, ct)):
            shoulder_mid = _midpoint(k[L_SHOULDER][:2], k[R_SHOULDER][:2])
            hip_mid = _midpoint(k[L_HIP][:2], k[R_HIP][:2])
            torso_vertical = abs(shoulder_mid[1] - hip_mid[1])
            torso_horizontal = abs(shoulder_mid[0] - hip_mid[0])


            if (torso_vertical + torso_horizontal) > 1e-6:
                torso_h_ratio = torso_horizontal / (torso_vertical + torso_horizontal)
                lying_extras["torso_h_ratio"] = torso_h_ratio
                if torso_h_ratio > 0.6:
                    lying_score += 0.35
                elif torso_h_ratio > 0.4:
                    lying_score += 0.2


        # Signal 2: Head and hips at similar Y level
        if (_conf(k, NOSE, ct) and _conf(k, L_HIP, ct) and _conf(k, R_HIP, ct)
                and body_y_span > 1e-6):
            head_y = float(k[NOSE][1])
            hip_mid_y = (float(k[L_HIP][1]) + float(k[R_HIP][1])) / 2.0
            head_hip_vert_ratio = abs(head_y - hip_mid_y) / body_y_span
            lying_extras["head_hip_vert_ratio"] = head_hip_vert_ratio
            if head_hip_vert_ratio < 0.2:
                lying_score += 0.3
            elif head_hip_vert_ratio < 0.3:
                lying_score += 0.15


        # Signal 3: Body X span dominates Y span
        if body_x_span > 0 and body_y_span > 0:
            xy_ratio = body_x_span / body_y_span
            if xy_ratio > 1.0:
                lying_score += 0.2
                lying_extras["xy_ratio"] = xy_ratio
            elif xy_ratio > 0.8:
                lying_score += 0.1


        if lying_score >= self.lying_score_thresh:
            p = float(min(0.95, 0.55 + lying_score * 0.4))
            lying_extras["method"] = "multi_signal"
            lying_extras["lying_score"] = lying_score
            return Classification(label="lying/fallen", p=p, extras=lying_extras)


        # ============================================================
        # SITTING DETECTION (perspective-independent)
        #
        # Two detection paths:
        # PATH A — "Feet out" (legs extended forward):
        #   shin_to_thigh explodes (0.71 → 1.86)
        #   knee_position drops (0.57 → 0.35)
        #
        # PATH B — "Feet tucked" (legs under chair):
        #   knee_angle is still bent (~126°)
        #   body_aspect is very compact (~1.79)
        # ============================================================
        sitting_score = 0.0
        sitting_extras = {"shoulder_width": shoulder_width}


        # --- Compute key Y positions ---
        hip_mid_y = None
        knee_mid_y = None
        ankle_mid_y = None


        if _conf(k, L_HIP, ct) and _conf(k, R_HIP, ct):
            hip_mid_y = (float(k[L_HIP][1]) + float(k[R_HIP][1])) / 2.0


        knee_ys = []
        ankle_ys = []
        if _conf(k, L_KNEE, ct):
            knee_ys.append(float(k[L_KNEE][1]))
        if _conf(k, R_KNEE, ct):
            knee_ys.append(float(k[R_KNEE][1]))
        if _conf(k, L_ANKLE, ct):
            ankle_ys.append(float(k[L_ANKLE][1]))
        if _conf(k, R_ANKLE, ct):
            ankle_ys.append(float(k[R_ANKLE][1]))


        if knee_ys:
            knee_mid_y = float(np.mean(knee_ys))
        if ankle_ys:
            ankle_mid_y = float(np.mean(ankle_ys))


        # --- Signal 1: shin_to_thigh (PATH A — feet out) ---
        # Standing (all heights): 0.60-0.75
        # Sitting (feet out): ~1.86
        shin_to_thigh = None
        if hip_mid_y is not None and knee_mid_y is not None and ankle_mid_y is not None:
            hip_to_knee = abs(knee_mid_y - hip_mid_y)
            knee_to_ankle = abs(ankle_mid_y - knee_mid_y)


            if hip_to_knee > 1e-6:
                shin_to_thigh = knee_to_ankle / hip_to_knee
                sitting_extras["shin_to_thigh"] = shin_to_thigh


                if shin_to_thigh > 1.4:
                    sitting_score += 0.55
                elif shin_to_thigh > 1.1:
                    sitting_score += 0.40


        # --- Signal 2: knee_position (PATH A — feet out) ---
        # Standing (all heights): 0.57-0.62
        # Sitting (feet out): ~0.35
        knee_position = None
        if hip_mid_y is not None and knee_mid_y is not None and ankle_mid_y is not None:
            hip_to_knee = abs(knee_mid_y - hip_mid_y)
            hip_to_ankle = abs(ankle_mid_y - hip_mid_y)


            if hip_to_ankle > 1e-6:
                knee_position = hip_to_knee / hip_to_ankle
                sitting_extras["knee_position"] = knee_position


                if knee_position < 0.40:
                    sitting_score += 0.50
                elif knee_position < 0.48:
                    sitting_score += 0.35


        # --- Signal 3: Knee angle (UNIVERSAL — works for ALL sitting) ---
        # Standing: 160-180° (straight legs)
        # Sitting (any position): 120-130° (bent knees)
        if (_conf(k, L_HIP, ct) and _conf(k, L_KNEE, ct) and _conf(k, L_ANKLE, ct)):
            left_angle = _angle(k[L_HIP][:2], k[L_KNEE][:2], k[L_ANKLE][:2])
        else:
            left_angle = 180.0


        if (_conf(k, R_HIP, ct) and _conf(k, R_KNEE, ct) and _conf(k, R_ANKLE, ct)):
            right_angle = _angle(k[R_HIP][:2], k[R_KNEE][:2], k[R_ANKLE][:2])
        else:
            right_angle = 180.0


        knee_angle = min(left_angle, right_angle)
        sitting_extras["knee_angle"] = knee_angle


        if knee_angle < 110:
            sitting_score += 0.50
        elif knee_angle < 130:
            sitting_score += 0.35
        elif knee_angle < 145:
            sitting_score += 0.15


        # --- Signal 4: Body aspect ratio (UNIVERSAL — works for ALL sitting) ---
        # Standing (all heights): 3.0-6.0 (tall and narrow)
        # Sitting (any position): 1.7-2.3 (compact)
        if body_x_span > 1e-6 and body_y_span > 1e-6:
            body_aspect = body_y_span / body_x_span
            sitting_extras["body_aspect"] = body_aspect


            if body_aspect < 2.0:
                sitting_score += 0.25
            elif body_aspect < 2.5:
                sitting_score += 0.15
            elif body_aspect < 2.8:
                sitting_score += 0.05


        # --- Debug output ---
        print(f"s2t={shin_to_thigh if shin_to_thigh else 'N/A'}, "
              f"kpos={knee_position if knee_position else 'N/A'}, "
              f"kang={knee_angle:.1f}, "
              f"aspect={sitting_extras.get('body_aspect', 'N/A')}, "
              f"score={sitting_score:.2f}")


        # --- Final sitting decision ---
        if sitting_score >= self.sitting_score_thresh:
            p = float(min(0.95, 0.55 + sitting_score * 0.25))
            sitting_extras["method"] = "perspective_independent"
            sitting_extras["sitting_score"] = sitting_score
            return Classification(label="sitting", p=p, extras=sitting_extras)


        # ============================================================
        # DEFAULT: STANDING
        # ============================================================
        extras = {"sitting_score": sitting_score, "lying_score": lying_score}
        if ratio is not None:
            extras["ratio"] = ratio
        extras["shoulder_width"] = shoulder_width
        return Classification(label="standing", p=0.7, extras=extras)


    # ================================================================
    # MAIN PROCESS
    # ================================================================
    def process(self, state: FrameState) -> FrameState:
        """Classify each person's posture and action signals per frame."""
        for person in state.people:
            pose = person.pose
            if pose is None:
                person.posture = Classification(label="unknown", p=0.2)
                person.action = Classification(label="none", p=0.2)
                continue


            # Check minimum keypoints available
            k_check = pose.kpts_norm if pose.kpts_norm is not None else pose.kpts_raw
            if k_check is None or k_check.size == 0:
                person.posture = Classification(label="unknown", p=0.3)
                person.action = Classification(label="none", p=0.3)
                continue


            conf = k_check[:, 2]
            vis_count = int((conf >= self.conf_thresh).sum())
            if vis_count < 4:
                person.posture = Classification(label="unknown", p=0.3)
                person.action = Classification(label="none", p=0.3)
                continue


            # Get person ID for debug output
            person_id = str(person.track.track_id) if person.track is not None else "?"


            # Classify posture (perspective-independent)
            person.posture = self._classify_posture(pose, person_id)


            # Detect action/emergency signals (tilt, hands on neck/chest)
            k = pose.kpts_raw
            if k is not None and k.shape[0] >= 17:
                person.action = self._detect_action(k, self.conf_thresh, posture_label=person.posture.label)
            else:
                person.action = Classification(
                    label="none", p=0.3,
                    extras={"is_tilted": False, "hands_chest": False, "hands_neck": False}
                )


            # Debug dump (includes action extras)
            if DEBUG_DUMP_ENABLED and k is not None and k.shape[0] >= 17:
                _dump_keypoints(k, person_id, action_extras=person.action.extras)


        return state







