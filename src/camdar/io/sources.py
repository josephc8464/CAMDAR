"""Video source helpers with GStreamer-first capture setup."""

from __future__ import annotations

import cv2
import glob
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


def _is_gst_pipeline(source: str) -> bool:
    """Return True when the source string already looks like a GStreamer pipeline."""
    return " !" in source and "appsink" in source


def _parse_udp_port(source: str, default_port: int = 5555) -> int:
    """Extract UDP port from `udp://...` URI, fallback to default when absent."""
    parsed = urlparse(source)
    if parsed.port:
        return int(parsed.port)

    # Support sources like udp://@:5555 or query-only forms.
    q = parse_qs(parsed.query)
    if "port" in q and q["port"]:
        try:
            return int(q["port"][0])
        except ValueError:
            return default_port

    return default_port


def _gst_pipeline_for_network_uri(source: str) -> list[str]:
    """
    Build candidate low-latency GStreamer pipelines for RTSP/UDP sources.

    Output format is BGR so downstream OpenCV processing can consume frames directly.
    """
    if source.startswith(("rtsp://", "rtsps://")):
        return [
            f"rtspsrc location={source} latency=100 ! "
            "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! "
            "video/x-raw,format=BGR ! appsink drop=1 sync=false"
        ]
    if source.startswith(("udp://", "udpsrc")):
        # Jetson-optimized receive path (RTP/H264 -> NVDEC -> BGR).
        port = _parse_udp_port(source)
        jetson_rtp_h264 = (
            f"udpsrc port={port} "
            "caps=\"application/x-rtp,media=video,encoding-name=H264,payload=96\" ! "
            "rtph264depay ! h264parse ! nvv4l2decoder ! nvvidconv ! "
            "video/x-raw,format=BGRx ! videoconvert ! "
            "video/x-raw,format=BGR ! appsink drop=1 sync=false"
        )

        # Fallback for raw H264-over-UDP (non-RTP), still using NVDEC.
        jetson_raw_h264 = (
            f"udpsrc port={port} ! h264parse ! nvv4l2decoder ! nvvidconv ! "
            "video/x-raw,format=BGRx ! videoconvert ! "
            "video/x-raw,format=BGR ! appsink drop=1 sync=false"
        )

        # Generic software fallback decode path.
        generic_decode = (
            f"uridecodebin uri={source} ! videoconvert ! "
            "video/x-raw,format=BGR ! appsink drop=1 sync=false"
        )

        return [jetson_rtp_h264, jetson_raw_h264, generic_decode]
    return [source]


def _gstreamer_enabled() -> bool:
    build_info = cv2.getBuildInformation()
    return "GStreamer:                   YES" in build_info or "GStreamer: YES" in build_info


def _discover_local_camera_indices(max_index: int = 10) -> list[int]:
    """
    Return candidate local camera indices.

    On Linux, prefer concrete `/dev/videoN` entries when available.
    Otherwise, fall back to scanning a bounded index range.
    """
    dev_indices: list[int] = []
    for path in glob.glob("/dev/video*"):
        suffix = Path(path).name.replace("video", "", 1)
        if suffix.isdigit():
            dev_indices.append(int(suffix))
    if dev_indices:
        return sorted(set(dev_indices))
    return list(range(max(0, int(max_index))))


def _linux_camera_name(index: int) -> str | None:
    """Return Linux V4L2 camera name when exposed in sysfs."""
    sysfs_name = Path(f"/sys/class/video4linux/video{index}/name")
    if not sysfs_name.exists():
        return None
    try:
        name = sysfs_name.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return name or None


def list_available_cameras(
    max_index: int = 10,
    network_sources: dict[str, str] | None = None,
    prefer_gstreamer: bool = True,
) -> dict[str, Any]:
    """
    Probe available camera sources and return structured availability metadata.

    `local` probes local capture indices (e.g. laptop webcam / USB cameras).
    `network` optionally probes caller-provided URIs (RTSP/UDP/GStreamer strings).
    """
    local: list[dict[str, Any]] = []
    network: list[dict[str, Any]] = []

    for index in _discover_local_camera_indices(max_index=max_index):
        cap = cv2.VideoCapture(index)
        opened = bool(cap.isOpened())
        backend = int(cap.get(cv2.CAP_PROP_BACKEND)) if opened else None
        cap.release()
        if not opened:
            continue
        local.append(
            {
                "index": index,
                "label": _linux_camera_name(index) or f"camera_{index}",
                "source": str(index),
                "backend": backend,
            }
        )

    for label, source in (network_sources or {}).items():
        cap, diagnostics = open_video_capture_with_diagnostics(
            source=source, prefer_gstreamer=prefer_gstreamer
        )
        opened = bool(cap.isOpened())
        cap.release()
        network.append(
            {
                "label": label,
                "source": source,
                "available": opened,
                "diagnostics": diagnostics,
            }
        )

    return {
        "local": local,
        "network": network,
    }


def open_video_capture_with_diagnostics(
    source: str | int, prefer_gstreamer: bool = True
) -> tuple[cv2.VideoCapture, dict[str, Any]]:
    """Open capture and return backend-attempt diagnostics."""
    source_str = str(source)
    diagnostics: dict[str, Any] = {
        "source": source_str,
        "prefer_gstreamer": bool(prefer_gstreamer),
        "opencv_version": cv2.__version__,
        "gstreamer_enabled": _gstreamer_enabled(),
        "attempts": [],
    }

    if source_str.isdigit():
        cap = cv2.VideoCapture(int(source_str))
        diagnostics["attempts"].append(
            {"backend": "camera-index", "target": source_str, "opened": bool(cap.isOpened())}
        )
        return cap, diagnostics

    if not prefer_gstreamer:
        cap = cv2.VideoCapture(source_str, cv2.CAP_FFMPEG)
        diagnostics["attempts"].append(
            {"backend": "ffmpeg", "target": source_str, "opened": bool(cap.isOpened())}
        )
        if cap.isOpened():
            return cap, diagnostics
        cap = cv2.VideoCapture(source_str)
        diagnostics["attempts"].append(
            {"backend": "default", "target": source_str, "opened": bool(cap.isOpened())}
        )
        return cap, diagnostics

    gst_candidates = (
        [source_str]
        if _is_gst_pipeline(source_str)
        else _gst_pipeline_for_network_uri(source_str)
    )
    for gst_source in gst_candidates:
        cap = cv2.VideoCapture(gst_source, cv2.CAP_GSTREAMER)
        opened = bool(cap.isOpened())
        diagnostics["attempts"].append(
            {"backend": "gstreamer", "target": gst_source, "opened": opened}
        )
        if opened:
            return cap, diagnostics

    cap = cv2.VideoCapture(source_str, cv2.CAP_FFMPEG)
    diagnostics["attempts"].append(
        {"backend": "ffmpeg", "target": source_str, "opened": bool(cap.isOpened())}
    )
    if cap.isOpened():
        return cap, diagnostics
    cap = cv2.VideoCapture(source_str)
    diagnostics["attempts"].append(
        {"backend": "default", "target": source_str, "opened": bool(cap.isOpened())}
    )
    return cap, diagnostics


def open_video_capture(source: str | int, prefer_gstreamer: bool = True) -> cv2.VideoCapture:
    """
    Open a video source.

    Strategy:
    - webcam index ("0", "1", ...) -> normal OpenCV capture
    - explicit gst pipeline or network uri -> try CAP_GSTREAMER first
    - fallback to CAP_FFMPEG, then default OpenCV backend
    """
    cap, _ = open_video_capture_with_diagnostics(
        source=source, prefer_gstreamer=prefer_gstreamer
    )
    return cap
