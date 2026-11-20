# src/camdar/stages/pose/onnx_pose.py
import numpy as np
import onnxruntime as ort
from camdar.core.pipeline import Stage, register_stage
from camdar.core.types import FrameState, PersonState, Track, Pose
from .normalize import normalize_kpts_center_scale


@register_stage("pose.onnx")
class OnnxPoseStage(Stage):
    def __init__(self, onnx_path: str, input_size: int = 640, conf: float = 0.25, **kw):
        super().__init__(onnx_path=onnx_path, input_size=input_size, conf=conf, **kw)
        self.onnx_path = onnx_path
        self.input_size = input_size
        self.conf = conf
        self.sess = None

    def setup(self):
        self.sess = ort.InferenceSession(
            self.onnx_path, providers=["CPUExecutionProvider"]
        )

    def process(self, state: FrameState) -> FrameState:
        img = state.frame.image
        # TODO: preprocess -> run -> postprocess to (bbox, kpts)
        detections = []  # list of (bbox_xyxy, det_conf, kpts_xyc)
        people = []
        for bbox, det_conf, raw_kpts in detections:
            norm_k, avg = normalize_kpts_center_scale(raw_kpts)
            ps = PersonState(
                track=Track(track_id=-1, bbox_xyxy=bbox, conf=det_conf),
                pose=Pose(kpts_raw=raw_kpts, kpts_norm=norm_k, avg_kpt_conf=avg),
                quality={"det_conf": det_conf, "avg_kpt_conf": avg},
            )
            people.append(ps)
        state.people = people
        return state
