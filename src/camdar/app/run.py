from __future__ import annotations
"""CLI entrypoint to run the CAMDAR pipeline on a single video source."""

import argparse
import time
import json
from pathlib import Path
from typing import Any, Dict, Union

import cv2

from camdar.core.types import FramePacket, FrameState
from camdar.core.pipeline import build_pipeline_from_config
from camdar.app.settings import load_config, resolve_paths
from camdar.io.sources import list_available_cameras, open_video_capture


def parse_args() -> argparse.Namespace:
    """Parse runtime arguments for source selection and display behavior."""
    p = argparse.ArgumentParser(description="CAMDAR CV Pipeline Runner")
    p.add_argument(
        "--config", type=str, default=None, help="Path to pipeline YAML config"
    )
    p.add_argument(
        "--source",
        type=str,
        default=None,
        help="Override source (webcam index, URI, or GST pipeline). If omitted, read from config.",
    )
    p.add_argument(
        "--camera-id", type=str, default=None, help="Override camera_id from config"
    )
    p.add_argument(
        "--display", type=int, default=None, help="1 to show window, 0 for headless"
    )
    p.add_argument(
        "--max-frames", type=int, default=0, help="Stop after N frames (0 = unlimited)"
    )
    p.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available local cameras and configured network sources, then exit.",
    )
    p.add_argument(
        "--max-camera-index",
        type=int,
        default=10,
        help="Highest local camera index probe range (used when /dev/video* is unavailable).",
    )
    return p.parse_args()


def open_capture(source: str) -> cv2.VideoCapture:
    """Open capture using GStreamer-first source handling."""
    return open_video_capture(source, prefer_gstreamer=True)


def ensure_parent(path: Union[str, Path]) -> None:
    """Create parent directories for the provided file path."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Run frame capture, pipeline processing, and optional display loop."""
    args = parse_args()
    cfg: Dict[str, Any] = {}
    if args.config:
        cfg = load_config(args.config)
    elif not args.list_cameras:
        raise ValueError("--config is required unless --list-cameras is used.")

    if args.list_cameras:
        app_cfg = cfg.get("app", {}) if isinstance(cfg, dict) else {}
        sources = app_cfg.get("sources", {})
        network_sources = (
            {str(k): str(v) for k, v in sources.items()} if isinstance(sources, dict) else {}
        )
        found = list_available_cameras(
            max_index=args.max_camera_index,
            network_sources=network_sources,
            prefer_gstreamer=True,
        )
        print(json.dumps(found, indent=2))
        return

    paths = resolve_paths(Path(cfg["paths"]["repo_root"]))
    paths.logs_dir.mkdir(parents=True, exist_ok=True)

    camera_id = args.camera_id or cfg["app"]["camera_id"]
    display = cfg["app"]["display"] if args.display is None else bool(args.display)

    # Build pipeline (stages may be empty at first)
    pipeline = build_pipeline_from_config(cfg)
    pipeline.setup()

    source = args.source
    if source is None:
        app_cfg = cfg.get("app", {})
        source_select = app_cfg.get("source_select")
        sources = app_cfg.get("sources", {})
        if source_select and isinstance(sources, dict) and source_select in sources:
            source = str(sources[source_select])
        else:
            source = "0"

    cap: cv2.VideoCapture | None = None
    frame_id = 0
    t0 = time.time()

    try:
        cap = open_capture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open source: {source}")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            # timestamp: prefer capture clock if available; fallback to wall time
            ts = cap.get(cv2.CAP_PROP_POS_MSEC)
            ts = float(ts / 1000.0) if ts and ts > 0 else (time.time() - t0)

            fp = FramePacket(frame_id=frame_id, ts=ts, camera_id=camera_id, image=frame)
            state = FrameState(frame=fp)

            # Process through pipeline
            state = pipeline.process(state)
            print(
                f"frame {frame_id} processed, state: {state}, people detected: {len(state.people)}"
            )

            # If no viz stage yet, optionally show raw frame
            if display:
                cv2.imshow("CAMDAR", state.frame.image)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break

            frame_id += 1
            if args.max_frames and frame_id >= args.max_frames:
                break

    finally:
        if cap is not None:
            cap.release()
        if display:
            cv2.destroyAllWindows()
        pipeline.teardown()


if __name__ == "__main__":
    main()
