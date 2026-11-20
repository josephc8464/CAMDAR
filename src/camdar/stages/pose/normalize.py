from __future__ import annotations
"""Pose normalization utilities shared by pose/posture temporal stages."""

from typing import Optional, Tuple
import numpy as np


def normalize_kpts_center_scale(
    kpts_xyc: np.ndarray, conf_thresh: float = 0.2
) -> Tuple[Optional[np.ndarray], float]:
    """
    Normalize keypoints using:
      - translate by mean of visible keypoints
      - scale by median distance of visible keypoints to center

    Args:
      kpts_xyc: (K,3) [x, y, conf] in pixel coords
      conf_thresh: visibility threshold

    Returns:
      (kpts_norm, avg_conf)
        kpts_norm: (K,3) [x_norm, y_norm, conf] or None if insufficient signal
        avg_conf: mean conf across all joints (0..1)
    """
    if kpts_xyc is None or kpts_xyc.size == 0:
        return None, 0.0

    conf = kpts_xyc[:, 2].astype(np.float32)
    avg_conf = float(conf.mean()) if conf.size else 0.0

    visible = conf > conf_thresh
    if int(visible.sum()) < 4:
        return None, avg_conf

    pts = kpts_xyc[:, :2].astype(np.float32)
    pts_vis = pts[visible]

    center = pts_vis.mean(axis=0)
    pts_centered = pts - center

    d = np.linalg.norm(pts_vis - center, axis=1)
    scale = (
        float(np.median(d)) if float(np.median(d)) > 1e-6 else float(d.mean() + 1e-6)
    )

    pts_norm = pts_centered / scale
    kpts_norm = np.concatenate([pts_norm, conf.reshape(-1, 1)], axis=1).astype(
        np.float32
    )

    return kpts_norm, avg_conf
