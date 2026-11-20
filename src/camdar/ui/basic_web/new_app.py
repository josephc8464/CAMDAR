from __future__ import annotations

"""Low-latency Flask/Waitress frontend: decoupled capture, inference, encoding, and status serving."""

import argparse
import atexit
import copy
import os
import queue
import signal
import subprocess
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


def _kill_stale_port(port: int) -> None:
    """Best-effort cleanup of any stale process holding the target HTTP port."""
    try:
        out = subprocess.check_output(
            ["lsof", "-ti", f"tcp:{port}"], stderr=subprocess.DEVNULL, text=True
        )
        my_pid = os.getpid()
        for line in out.strip().splitlines():
            pid = int(line.strip())
            if pid != my_pid:
                os.kill(pid, signal.SIGKILL)
    except Exception:
        pass


class CameraService:
    """Decoupled capture -> inference -> preview-encode service shared by HTTP routes."""

    def __init__(self, config_path: str, source: str | None = None) -> None:
        self.cfg = load_config(config_path)
        app_cfg = self.cfg.get("app", {})
        ui_cfg = self.cfg.get("ui", {})

        self.camera_id = str(app_cfg.get("camera_id", "cam_01"))
        self.source = _resolve_source(self.cfg, source)

        # Preview encoding knobs (configurable in YAML under `ui:` or `app:`)
        self.preview_fps = float(ui_cfg.get("preview_fps", app_cfg.get("preview_fps", 15.0)))
        self.preview_max_width = int(ui_cfg.get("preview_max_width", app_cfg.get("preview_max_width", 960)))
        self.jpeg_quality = int(ui_cfg.get("jpeg_quality", app_cfg.get("jpeg_quality", 75)))

        self.pipeline = build_pipeline_from_config(self.cfg)
        self.pipeline.setup()

        self.cap, self._open_diag = open_video_capture_with_diagnostics(
            self.source, prefer_gstreamer=True
        )
        self.frame_id = 0
        self.reconnects = 0
        self._stop = False

        # 1. Latest-frame handoff slot (Capture -> Inference)
        self._raw_cond = threading.Condition()
        self._latest_raw: tuple[int, float, Any] | None = None
        self._raw_seq = 0

        # 2. Latest-annotated-frame handoff slot (Inference -> JPEG Encoder)
        self._encode_cond = threading.Condition()
        self._latest_annotated: tuple[int, Any, float] | None = None
        self._annotated_seq = 0

        # 3. Latest-JPEG handoff slot (JPEG Encoder -> MJPEG HTTP clients)
        self._jpeg_cond = threading.Condition()
        self._latest_jpeg: bytes = b""
        self._jpeg_seq = 0

        # 4. Non-blocking drop-oldest event queue (Inference -> Async Publisher)
        self._event_queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)

        # 5. Lock-protected pre-built status payloads (Inference -> GET /status & /status/hot)
        self._status_lock = threading.Lock()
        self._last_wall: float | None = None
        self._fps = 0.0

        initial_hot = {
            "camera_id": self.camera_id,
            "people_count": 0,
            "emergency_count": 0,
            "uncertain_count": 0,
            "people": [],
            "fps": 0.0,
            "latency_ms": 0.0,
            "reconnects": 0,
            "error": None,
        }
        self._hot_status: dict[str, Any] = dict(initial_hot)
        self._full_status: dict[str, Any] = {
            **initial_hot,
            "source": self.source,
            "stage_timings_ms": {},
            "open_diagnostics": self._open_diag,
        }

        # Launch decoupled background workers
        self._capture_thread = threading.Thread(
            target=self._capture_loop, name="camdar-capture", daemon=True
        )
        self._infer_thread = threading.Thread(
            target=self._inference_loop, name="camdar-infer", daemon=True
        )
        self._encode_thread = threading.Thread(
            target=self._encode_loop, name="camdar-encode", daemon=True
        )
        self._pub_thread = threading.Thread(
            target=self._publisher_loop, name="camdar-pub", daemon=True
        )

        self._capture_thread.start()
        self._infer_thread.start()
        self._encode_thread.start()
        self._pub_thread.start()
        

    def _ensure_capture(self) -> bool:
        if self.cap is not None and self.cap.isOpened():
            return True
        if self.cap is not None:
            self.cap.release()
        self.cap, self._open_diag = open_video_capture_with_diagnostics(
            self.source, prefer_gstreamer=True
        )
        if self.cap is not None and self.cap.isOpened():
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

    def _set_error(self, msg: str) -> None:
        with self._status_lock:
            self._hot_status["error"] = msg
            self._hot_status["reconnects"] = self.reconnects
            self._full_status["error"] = msg
            self._full_status["reconnects"] = self.reconnects
            self._full_status["open_diagnostics"] = self._open_diag

    def _update_fps(self, now: float) -> float:
        if self._last_wall is None:
            self._last_wall = now
            return self._fps
        dt = max(1e-6, now - self._last_wall)
        self._last_wall = now
        inst = 1.0 / dt
        self._fps = inst if self._fps <= 0 else (0.85 * self._fps + 0.15 * inst)
        return 0 #self.fps

    def _build_snapshots(
        self, state: FrameState, latency_ms: float, fps: float
    ) -> tuple[dict[str, Any], dict[str, Any], bool]:
        """Build compact hot status, full status, and alert flag in one pass."""
        people_hot = []
        people_full = []
        emergency = 0
        uncertain = 0

        for person in state.people:
            status_label = person.status.label if person.status else "unknown"
            if status_label == "emergency":
                emergency += 1
            elif status_label == "uncertain":
                uncertain += 1

            tid = int(person.track.track_id)
            posture_label = person.posture.label if person.posture else "unknown"
            status_p = round(float(person.status.p), 3) if person.status else 0.0
            posture_p = round(float(person.posture.p), 3) if person.posture else 0.0

            people_hot.append(
                {
                    "track_id": tid,
                    "status": status_label,
                    "posture": posture_label,
                }
            )
            people_full.append(
                {
                    "track_id": tid,
                    "status": status_label,
                    "status_p": status_p,
                    "posture": posture_label,
                    "posture_p": posture_p,
                    "det_conf": round(float(person.track.conf), 3),
                    "avg_kpt_conf": round(float(person.pose.avg_kpt_conf), 3)
                    if person.pose
                    else 0.0,
                }
            )

        hot = {
            "camera_id": self.camera_id,
            "people_count": len(people_hot),
            "emergency_count": emergency,
            "uncertain_count": uncertain,
            "people": people_hot,
            "fps": round(fps, 1),
            "latency_ms": round(latency_ms, 2),
            "reconnects": self.reconnects,
            "error": None,
        }

        debug_meta = getattr(state, "debug", None) or {}
        stage_timings = debug_meta.get("stage_timings_ms", {}) if isinstance(debug_meta, dict) else {}

        full = {
            **hot,
            "people": people_full,
            "source": self.source,
            "stage_timings_ms": stage_timings,
            "open_diagnostics": self._open_diag,
        }
        has_alert = emergency > 0
        return hot, full, has_alert

    def _enqueue_event_nonblocking(self, payload: dict[str, Any]) -> None:
        """Drop-oldest non-blocking push so network/DB I/O never stalls inference."""
        try:
            self._event_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            self._event_queue.put_nowait(payload)
        except queue.Full:
            pass

    # ------------------------------------------------------------------
    # Worker 1: Continuous Camera Capture (prevents GStreamer buffer lag)
    # ------------------------------------------------------------------
    def _capture_loop(self) -> None:
        while not self._stop:
            if not self._ensure_capture():
                self._set_error(self._format_open_error())
                time.sleep(0.25)
                continue

            ok, frame = self.cap.read()
            if not ok or frame is None:
                self._set_error("No frame read from source")
                time.sleep(0.02)
                continue

            fid = self.frame_id
            self.frame_id += 1
            now = time.time()

            with self._raw_cond:
                self._latest_raw = (fid, now, frame)
                self._raw_seq += 1
                self._raw_cond.notify()

    # ------------------------------------------------------------------
    # Worker 2: Pure GPU/Pipeline Inference Loop (zero JPEG/HTTP blocking)
    # ------------------------------------------------------------------
    def _inference_loop(self) -> None:
        last_seen_seq = 0
        while not self._stop:
            with self._raw_cond:
                while not self._stop and self._raw_seq == last_seen_seq:
                    self._raw_cond.wait(timeout=0.1)
                if self._stop:
                    return
                raw_item = self._latest_raw
                last_seen_seq = self._raw_seq

            if raw_item is None:
                continue

            fid, ts, frame = raw_item
            t0 = time.perf_counter()
            packet = FramePacket(
                frame_id=fid,
                ts=ts,
                camera_id=self.camera_id,
                image=frame,
            )

            try:
                state = self.pipeline.process(FrameState(frame=packet))
                latency_ms = (time.perf_counter() - t0) * 1000.0
                fps = self._update_fps(time.time())

                hot, full, has_alert = self._build_snapshots(state, latency_ms, fps)
                with self._status_lock:
                    self._hot_status = hot
                    self._full_status = full

                # Push only threshold-crossed events to background publisher
                if has_alert:
                    self._enqueue_event_nonblocking(
                        {
                            "ts": ts,
                            "camera_id": self.camera_id,
                            "frame_id": fid,
                            "emergency_count": hot["emergency_count"],
                            "people": hot["people"],
                        }
                    )

                # Hand frame to background JPEG encoder without waiting for imencode
                with self._encode_cond:
                    self._latest_annotated = (fid, state.frame.image, fps)
                    self._annotated_seq += 1
                    self._encode_cond.notify()

            except Exception as exc:
                self._set_error(str(exc))

    # ------------------------------------------------------------------
    # Worker 3: Rate-Limited JPEG Encoder (offloads CPU imencode)
    # ------------------------------------------------------------------
    def _encode_loop(self) -> None:
        last_encoded_seq = 0
        min_interval = (1.0 / self.preview_fps) if self.preview_fps > 0 else 0.0
        last_encode_time = 0.0
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), max(30, min(95, self.jpeg_quality))]

        while not self._stop:
            with self._encode_cond:
                while not self._stop and self._annotated_seq == last_encoded_seq:
                    self._encode_cond.wait(timeout=0.1)
                if self._stop:
                    return
                item = self._latest_annotated
                last_encoded_seq = self._annotated_seq

            if item is None:
                continue

            # Throttle preview encoding rate so browser preview doesn't starve CPU
            if min_interval > 0:
                elapsed = time.perf_counter() - last_encode_time
                if elapsed < min_interval:
                    time.sleep(min_interval - elapsed)

            _, draw_src, fps = item
            draw = draw_src.copy()

            # Downscale preview image if larger than preview_max_width
            h, w = draw.shape[:2]
            if self.preview_max_width > 0 and w > self.preview_max_width:
                scale = self.preview_max_width / float(w)
                draw = cv2.resize(
                    draw,
                    (self.preview_max_width, int(h * scale)),
                    interpolation=cv2.INTER_LINEAR,
                )

            cv2.rectangle(draw, (10, 10), (210, 46), (0, 0, 0), -1)
            cv2.putText(
                draw,
                f"FPS: {fps:.1f}",
                (18, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0),
                2,
            )

            ok_jpg, encoded = cv2.imencode(".jpg", draw, encode_params)
            last_encode_time = time.perf_counter()

            if ok_jpg:
                jpg_bytes = encoded.tobytes()
                with self._jpeg_cond:
                    self._latest_jpeg = jpg_bytes
                    self._jpeg_seq += 1
                    self._jpeg_cond.notify_all()

    # ------------------------------------------------------------------
    # Worker 4: Async Event Publisher (absorbs external HTTP/DB latency)
    # ------------------------------------------------------------------
    def _publisher_loop(self) -> None:
        while not self._stop:
            try:
                event = self._event_queue.get(timeout=0.25)
            except queue.Empty:
                continue
            # Hook external HTTP webhook / DB / MQTT dispatch here so it never blocks inference.
            _ = event

    # ------------------------------------------------------------------
    # Zero-Stall HTTP Reader Methods
    # ------------------------------------------------------------------
    def mjpeg_generator(self):
        last_sent_seq = 0
        while not self._stop:
            with self._jpeg_cond:
                while not self._stop and self._jpeg_seq == last_sent_seq:
                    self._jpeg_cond.wait(timeout=0.25)
                if self._stop:
                    return
                frame = self._latest_jpeg
                last_sent_seq = self._jpeg_seq

            if not frame:
                continue

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(frame)).encode("ascii") + b"\r\n\r\n"
                + frame
                + b"\r\n"
            )

    def status_hot(self) -> dict[str, Any]:
        with self._status_lock:
            return copy.deepcopy(self._hot_status)

    def status(self) -> dict[str, Any]:
        with self._status_lock:
            return copy.deepcopy(self._full_status)

    def stop(self) -> None:
        if self._stop:
            return
        self._stop = True

        with self._raw_cond:
            self._raw_cond.notify_all()
        with self._encode_cond:
            self._encode_cond.notify_all()
        with self._jpeg_cond:
            self._jpeg_cond.notify_all()

        for t in (
            self._capture_thread,
            self._infer_thread,
            self._encode_thread,
            self._pub_thread,
        ):
            if t.is_alive():
                t.join(timeout=1.0)

        if self.cap is not None:
            self.cap.release()
        self.pipeline.teardown()


def create_app(config_path: str, source: str | None = None) -> Flask:
    app = Flask(__name__)
    service = CameraService(config_path=config_path, source=source)

    @app.get("/")
    def index():
        return render_template("new_index.html")

    @app.get("/video_feed")
    def video_feed():
        return Response(
            service.mjpeg_generator(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
                "Pragma": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.get("/status")
    def status():
        return jsonify(service.status())

    @app.get("/status/hot")
    def status_hot():
        return jsonify(service.status_hot())

    @app.post("/shutdown")
    def shutdown():
        service.stop()
        server_shutdown = request.environ.get("werkzeug.server.shutdown")
        if callable(server_shutdown):
            threading.Thread(target=server_shutdown, daemon=True).start()
        else:
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
    p.add_argument(
        "--kill-stale-port",
        action="store_true",
        help="Kill any stale process listening on --port before startup",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.kill_stale_port:
        _kill_stale_port(args.port)

    app = create_app(config_path=args.config, source=args.source)
    service = app.extensions.get("camera_service")  # store service on app in create_app()

    try:
        from waitress import serve
        print(f"[CAMDAR] Serving with Waitress on http://{args.host}:{args.port}")
        serve(app, host=args.host, port=args.port, threads=8, channel_timeout=30)
    finally:
        if service is not None:
            service.stop()


if __name__ == "__main__":
    main()