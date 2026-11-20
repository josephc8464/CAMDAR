from __future__ import annotations

"""Basic Flask frontend: live camera output with right-side status panel."""

import argparse
import atexit
import os
import signal
import threading
import time
from typing import Any

import cv2
from flask import Flask, Response, jsonify, render_template, request

from camdar.app.settings import load_config
from camdar.core.pipeline import build_pipeline_from_config
from camdar.core.types import FramePacket, FrameState
from camdar.io.sources import open_video_capture_with_diagnostics

def _resolve_source(cfg: dict[str, Any], cli_source: str | None) -> str:
    if cli_source:
        return cli_source
    app_cfg = cfg.get("app", {})
    source_select = app_cfg.get("source_select")
    sources = app_cfg.get("sources", {})
    if source_select and isinstance(sources, dict) and source_select in sources:
        return str(sources[source_select])
    return "0"


class CameraService:
    """Background capture/inference service shared by Flask routes."""

    def __init__(self, config_path: str, source: str | None = None) -> None:
        self.cfg = load_config(config_path)
        self.camera_id = str(self.cfg.get("app", {}).get("camera_id", "cam_01"))
        self.source = _resolve_source(self.cfg, source)
        self.pipeline = build_pipeline_from_config(self.cfg)
        self.pipeline.setup()

        self.cap, self._open_diag = open_video_capture_with_diagnostics(
            self.source, prefer_gstreamer=True
        )
        self.frame_id = 0
        self.reconnects = 0
        self._stop = False
        self._lock = threading.Lock()
        self._latest_jpeg = b""
        self._last_wall = None
        self._fps = 0.0
        self._status: dict[str, Any] = {
            "people_count": 0,
            "emergency_count": 0,
            "uncertain_count": 0,
            "people": [],
            "fps": 0.0,
            "latency_ms": 0.0,
            "reconnects": 0,
            "error": None,
            "open_diagnostics": self._open_diag,
        }
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _ensure_capture(self) -> bool:
        if self.cap is not None and self.cap.isOpened():
            return True
        if self.cap is not None:
            self.cap.release()
        self.cap, self._open_diag = open_video_capture_with_diagnostics(
            self.source, prefer_gstreamer=True
        )
        if self.cap.isOpened():
            self.reconnects += 1
            return True
        return False

    def _format_open_error(self) -> str:
        if not isinstance(self._open_diag, dict):
            return f"Could not open source: {self.source}"
        attempts = self._open_diag.get("attempts", [])
        summary = ", ".join(
            f"{item.get('backend')}={'ok' if item.get('opened') else 'fail'}"
            for item in attempts
        )
        return (
            f"Could not open source: {self.source} "
            f"(OpenCV {self._open_diag.get('opencv_version')}, "
            f"GStreamer enabled: {self._open_diag.get('gstreamer_enabled')}, "
            f"attempts: {summary or 'none'})"
        )

    def _update_fps(self, now: float) -> float:
        if self._last_wall is None:
            self._last_wall = now
            return self._fps
        dt = max(1e-6, now - self._last_wall)
        self._last_wall = now
        inst = 1.0 / dt
        self._fps = inst if self._fps <= 0 else (0.85 * self._fps + 0.15 * inst)
        return self._fps

    def _snapshot(self, state: FrameState, latency_ms: float, fps: float) -> dict[str, Any]:
        people = []
        emergency = 0
        uncertain = 0
        for person in state.people:
            status_label = person.status.label if person.status else "unknown"
            if status_label == "emergency":
                emergency += 1
            if status_label == "uncertain":
                uncertain += 1
            people.append(
                {
                    "track_id": int(person.track.track_id),
                    "status": status_label,
                    "status_p": float(person.status.p) if person.status else 0.0,
                    "posture": person.posture.label if person.posture else "unknown",
                    "posture_p": float(person.posture.p) if person.posture else 0.0,
                    "det_conf": float(person.track.conf),
                    "avg_kpt_conf": float(person.pose.avg_kpt_conf) if person.pose else 0.0,
                }
            )
        return {
            "people_count": len(people),
            "emergency_count": emergency,
            "uncertain_count": uncertain,
            "people": people,
            "fps": fps,
            "latency_ms": latency_ms,
            "reconnects": self.reconnects,
            "error": None,
            "open_diagnostics": self._open_diag,
        }

    def _run(self) -> None:
        while not self._stop:
            if not self._ensure_capture():
                with self._lock:
                    self._status["error"] = self._format_open_error()
                    self._status["reconnects"] = self.reconnects
                    self._status["open_diagnostics"] = self._open_diag
                time.sleep(0.25)
                continue

            ok, frame = self.cap.read()
            if not ok or frame is None:
                with self._lock:
                    self._status["error"] = "No frame read from source"
                    self._status["reconnects"] = self.reconnects
                    self._status["open_diagnostics"] = self._open_diag
                time.sleep(0.05)
                continue

            t0 = time.perf_counter()
            packet = FramePacket(
                frame_id=self.frame_id,
                ts=time.time(),
                camera_id=self.camera_id,
                image=frame,
            )
            self.frame_id += 1

            try:
                state = self.pipeline.process(FrameState(frame=packet))
                latency_ms = (time.perf_counter() - t0) * 1000.0
                fps = self._update_fps(time.time())

                draw = state.frame.image
                cv2.rectangle(draw, (10, 10), (210, 46), (0, 0, 0), -1)
                cv2.putText(draw, f"FPS: {fps:.1f}", (18, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                ok_jpg, encoded = cv2.imencode(".jpg", draw)
                if ok_jpg:
                    with self._lock:
                        self._latest_jpeg = encoded.tobytes()
                        self._status = self._snapshot(state, latency_ms=latency_ms, fps=fps)
            except Exception as e:
                with self._lock:
                    self._status["error"] = str(e)

    def mjpeg_generator(self):
        while not self._stop:
            with self._lock:
                frame = self._latest_jpeg
            if not frame:
                time.sleep(0.02)
                continue
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            )
            time.sleep(0.02)

    def status(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._status)

    def stop(self) -> None:
        if self._stop:
            return
        self._stop = True
        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
        self.pipeline.teardown()


def create_app(config_path: str, source: str | None = None) -> Flask:
    app = Flask(__name__)
    service = CameraService(config_path=config_path, source=source)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/video_feed")
    def video_feed():
        return Response(
            service.mjpeg_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/status")
    def status():
        return jsonify(service.status())

    @app.post("/shutdown")
    def shutdown():
        service.stop()
        server_shutdown = request.environ.get("werkzeug.server.shutdown")
        if callable(server_shutdown):
            threading.Thread(target=server_shutdown, daemon=True).start()
        else:
            # Fallback for non-Werkzeug launchers.
            threading.Thread(
                target=lambda: os.kill(os.getpid(), signal.SIGTERM), daemon=True
            ).start()
        return jsonify({"ok": True, "message": "Server shutting down"})

    @atexit.register
    def _cleanup():
        service.stop()

    return app


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CAMDAR Basic Web UI")
    p.add_argument("--config", type=str, default="configs/iter1.yaml")
    p.add_argument("--source", type=str, default=None, help="Optional source override")
    p.add_argument("--host", type=str, default="0.0.0.0")
    p.add_argument("--port", type=int, default=8080)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app = create_app(config_path=args.config, source=args.source)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
