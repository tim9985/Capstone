"""
kp_models.py — 관절점 모델 묶음 (K3 · 10-10) — 한 크롭 / 한 그림 + 대상 박스 → (관절 17×2 그림 좌표, 확신 17) 또는 None

  위에서 내려 찍는 모델 (박스를 받는다)
    flypose_s · flypose_h   kp_flypose 의 공식 전처리 (여백 1.1 · 256×192 · RGB) · 히트맵
    vitpose_s · b · l       easy_ViTPose ONNX (COCO) · rtmlib ViTPose 전후처리 (여백 1.25 · DARK-UDP)
    rtmpose_s · m · x       OpenMMLab body7 ONNX · rtmlib RTMPose 전후처리 (여백 1.25 · SimCC)
  한 번에 찾는 모델 (크롭에서 사람을 찾아 IoU 최대)
    rtmo_s · m · l          OpenMMLab body7 ONNX · rtmlib RTMO 전처리 (640 레터박스) · 박스 · 관절 직접 꺼냄 · CPU (출력 크기가 가변이라 TensorRT 불가)
    yolo11s · m · x · yolo26x   Ultralytics · kp_flypose.yolo_kp (K1 방식)
  ONNX 는 TensorRT FP32 (3090) 로 바꿔 돌린다 (onnxruntime-gpu 는 CUDA 12 용이라 이 환경에서 GPU 안 됨) · 엔진은 ONNX 옆 *.trt3090
  rtmlib 은 설치 없이 소스 경로로 (data/raw/pose_ext/rtmlib-main · Apache-2.0)
"""
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import tensorrt as trt
import torch

BASE = Path(__file__).resolve().parent
EXT = BASE.parent / "data" / "raw" / "pose_ext"
FPW = BASE.parent / "data" / "raw" / "flypose" / "checkpoints" / "pose"
sys.path.insert(0, str(EXT / "rtmlib-main"))
from rtmlib.tools.pose_estimation.rtmo import RTMO          # noqa: E402
from rtmlib.tools.pose_estimation.rtmpose import RTMPose    # noqa: E402
from rtmlib.tools.pose_estimation.vitpose import ViTPose    # noqa: E402

import kp_flypose as KF                                     # noqa: E402

LOG = trt.Logger(trt.Logger.ERROR)


class TrtSession:
    """onnxruntime 세션 흉내 (get_inputs · get_outputs · run) — 배치 1 · TensorRT FP32"""
    class _N:
        def __init__(self, name):
            self.name = name

    def __init__(self, onnx, engine=None):
        onnx = Path(onnx); eng = Path(engine) if engine else onnx.with_suffix(".trt3090")
        if not eng.exists():
            b = trt.Builder(LOG); net = b.create_network(0); p = trt.OnnxParser(net, LOG)
            assert p.parse_from_file(str(onnx)), [p.get_error(i) for i in range(p.num_errors)]
            c = b.create_builder_config(); c.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 << 30)
            inp = net.get_input(0); shp = [1 if d < 0 else d for d in inp.shape]
            prof = b.create_optimization_profile(); prof.set_shape(inp.name, shp, shp, shp); c.add_optimization_profile(prof)
            s = b.build_serialized_network(net, c); assert s is not None, f"엔진 실패 {onnx}"
            eng.write_bytes(bytes(s))
        self.e = trt.Runtime(LOG).deserialize_cuda_engine(eng.read_bytes()); self.ctx = self.e.create_execution_context()
        names = [self.e.get_tensor_name(i) for i in range(self.e.num_io_tensors)]
        self.ins = [n for n in names if self.e.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.outs = [n for n in names if self.e.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.st = torch.cuda.Stream()

    def get_inputs(self):
        return [self._N(n) for n in self.ins]

    def get_outputs(self):
        return [self._N(n) for n in self.outs]

    def run(self, names, feed):
        x = torch.from_numpy(np.ascontiguousarray(next(iter(feed.values()), None), dtype=np.float32)).cuda()
        self.ctx.set_input_shape(self.ins[0], tuple(x.shape)); self.ctx.set_tensor_address(self.ins[0], x.data_ptr())
        outs = {}
        for n in self.outs:
            shp = tuple(self.ctx.get_tensor_shape(n)); assert all(d >= 0 for d in shp), f"출력 모양이 정해지지 않음 {n} {shp}"
            dt = torch.float32 if self.e.get_tensor_dtype(n) == trt.DataType.FLOAT else torch.int64
            outs[n] = torch.empty(shp, dtype=dt, device="cuda"); self.ctx.set_tensor_address(n, outs[n].data_ptr())
        self.ctx.execute_async_v3(self.st.cuda_stream); self.st.synchronize()
        return [outs[n].cpu().numpy() for n in (names or self.outs)]


def _iou(a, b):
    return KF.iou(a, b)


class TopDown:
    def __init__(self, kind, onnx, size=(192, 256)):
        self.kind = kind
        if kind == "flypose":
            self.sess = TrtSession(onnx, Path(onnx).parent / "flypose_h_fp32_3090.engine" if "flypose_h" in str(onnx) else None)
        else:
            cls = ViTPose if kind == "vitpose" else RTMPose
            self.tool = cls(str(onnx), model_input_size=size, backend="onnxruntime", device="cpu")
            self.tool.session = TrtSession(onnx)

    def __call__(self, frame, box):
        if self.kind == "flypose":
            t, inv = KF.fp_preprocess(frame, box)
            hm = self.sess.run(None, {"input": t})[0][0]
            return KF.fp_decode(hm, inv)
        k, s = self.tool(frame, [list(box)])
        return k[0], s[0]


class Rtmo:
    def __init__(self, onnx):
        self.tool = RTMO(str(onnx), model_input_size=(640, 640), backend="onnxruntime", device="cpu")   # 모델 안 NMS 로 출력 크기가 매번 달라 TensorRT 불가 → CPU

    def __call__(self, frame, box):
        x1, y1, x2, y2 = box; L = max(x2 - x1, y2 - y1); m = 0.5 * L
        X1, Y1 = int(max(0, x1 - m)), int(max(0, y1 - m)); X2, Y2 = int(min(frame.shape[1], x2 + m)), int(min(frame.shape[0], y2 + m))
        crop = frame[Y1:Y2, X1:X2]; f = 256 / L
        up = cv2.resize(crop, (max(1, round(crop.shape[1] * f)), max(1, round(crop.shape[0] * f))), interpolation=cv2.INTER_LINEAR)
        k = up.shape[1] / crop.shape[1]
        img, ratio = self.tool.preprocess(up); det, pose = self.tool.inference(img)
        bx = det[0, :, :4] / ratio; sc = det[0, :, 4]; kp = pose[0, :, :, :2] / ratio; ks = pose[0, :, :, 2]
        keep = sc >= 0.1
        if not keep.any():
            return None
        bx, sc, kp, ks = bx[keep], sc[keep], kp[keep], ks[keep]
        gb = np.array([x1 - X1, y1 - Y1, x2 - X1, y2 - Y1]) * k
        j = max(range(len(bx)), key=lambda i: _iou(bx[i], gb))
        if _iou(bx[j], gb) < 0.3:
            return None
        return kp[j] / k + np.array([X1, Y1]), ks[j]


class Yolo:
    def __init__(self, name):
        from ultralytics import YOLO
        self.m = YOLO(str(BASE.parent / "data" / "raw" / "pose_weights" / f"{name}-pose.pt"))

    def __call__(self, frame, box):
        return KF.yolo_kp(self.m, frame, box)


def _rtm(sub, name):
    return next((EXT / sub).glob(f"{name}*/**/end2end.onnx"))


REGISTRY = {
    "flypose_s": lambda: TopDown("flypose", FPW / "flypose_s" / "end2end.onnx"),
    "flypose_h": lambda: TopDown("flypose", FPW / "flypose_h" / "end2end.onnx"),
    "vitpose_s": lambda: TopDown("vitpose", EXT / "vitpose-s-coco.onnx"),
    "vitpose_b": lambda: TopDown("vitpose", EXT / "vitpose-b-coco.onnx"),
    "vitpose_l": lambda: TopDown("vitpose", EXT / "vitpose-l-coco.onnx"),
    "rtmpose_s": lambda: TopDown("rtmpose", _rtm(".", "rtmpose-s")),
    "rtmpose_m": lambda: TopDown("rtmpose", _rtm(".", "rtmpose-m")),
    "rtmpose_x": lambda: TopDown("rtmpose", _rtm(".", "rtmpose-x"), size=(288, 384)),
    "rtmo_s": lambda: Rtmo(_rtm(".", "rtmo-s")),
    "rtmo_m": lambda: Rtmo(_rtm(".", "rtmo-m")),
    "rtmo_l": lambda: Rtmo(_rtm(".", "rtmo-l")),
    "yolo11s": lambda: Yolo("yolo11s"),
    "yolo11m": lambda: Yolo("yolo11m"),
    "yolo11x": lambda: Yolo("yolo11x"),
    "yolo26x": lambda: Yolo("yolo26x"),
}


def timed(fn, *a):
    t0 = time.perf_counter(); r = fn(*a); return r, (time.perf_counter() - t0) * 1000
