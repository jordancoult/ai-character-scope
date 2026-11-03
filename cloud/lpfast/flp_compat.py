"""Make FasterLivePortrait's model wrappers run on TensorRT 11 / Blackwell without its grid-sample plugin.

Import this module before anything from `src.` (FasterLivePortrait). It
- puts third_party/FasterLivePortrait on sys.path;
- stubs `insightface.app.common.Face` (the only insightface symbol used; avoids a compiled dependency);
- replaces `src.models.predictor.TensorRTPredictor` with a TensorRT 10/11 version (name-based I/O, no plugin .so);
- `load_models()` builds the model wrappers on our engines in checkpoints/trt (see build_engines.py).
"""

import sys
import types
from collections import OrderedDict
from pathlib import Path

import numpy as np
import tensorrt as trt
import torch

HERE = Path(__file__).resolve().parent
FLP = HERE / "third_party" / "FasterLivePortrait"
ENG = HERE / "checkpoints" / "trt"
sys.path.insert(0, str(FLP))

if "insightface" not in sys.modules:
    class Face(dict):
        def __init__(self, d=None, **kw):
            super().__init__(d or {}, **kw)

        def __getattr__(self, k):
            return self.get(k)

        def __setattr__(self, k, v):
            self[k] = v

    _i, _a, _c = (types.ModuleType(n) for n in ("insightface", "insightface.app", "insightface.app.common"))
    _c.Face = Face
    _i.app, _a.common = _a, _c
    sys.modules.update({"insightface": _i, "insightface.app": _a, "insightface.app.common": _c})

import src.models.predictor as _P  # noqa: E402

_NP = {trt.float32: np.float32, trt.float16: np.float16, trt.int32: np.int32, trt.int64: np.int64,
       trt.int8: np.int8, trt.uint8: np.uint8, trt.bool: np.bool_}
_LOG = trt.Logger(trt.Logger.ERROR)


class TRT11Predictor(_P.TensorRTPredictor):
    def __init__(self, **kwargs):  # noqa: D401  (no plugin load, TensorRT >= 10 API)
        path = kwargs["model_path"]
        self.debug = kwargs.get("debug", False)
        self.engine = trt.Runtime(_LOG).deserialize_cuda_engine(Path(path).read_bytes())
        assert self.engine is not None, path
        self.context = self.engine.create_execution_context()
        self.inputs, self.outputs, self.tensors = [], [], OrderedDict()
        for idx in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(idx)
            shape = tuple(self.engine.get_tensor_shape(name))
            dt = _NP[self.engine.get_tensor_dtype(name)]
            b = {"index": idx, "name": name, "dtype": dt, "shape": list(shape)}
            (self.inputs if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT else self.outputs).append(b)
            self.tensors[name] = torch.empty(shape, dtype=_P.numpy_to_torch_dtype_dict[dt], device="cuda")
        order = kwargs.get("onnx_path")
        if order:  # the wrappers unpack outputs by position: follow the ONNX graph's output order
            import onnx
            names = [o.name for o in onnx.load(order, load_external_data=False).graph.output]
            self.outputs.sort(key=lambda b: names.index(b["name"]))


_P.TensorRTPredictor = TRT11Predictor


def load_models():
    """-> dict of FasterLivePortrait model wrappers on our TensorRT engines."""
    from src import models as M
    e = lambda n: str(ENG / f"{n}.engine")  # noqa: E731
    src = HERE / "checkpoints" / "liveportrait_onnx"
    onnx_of = {"warping_spade": "warping_spade-fix", "app_feat_extractor": "appearance_feature_extractor",
               "stitching_eye_retarget": "stitching_eye", "stitching_lip_retarget": "stitching_lip"}
    spec = {
        "warping_spade": ("WarpingSpadeModel", e("warping_fp16")),
        "motion_extractor": ("MotionExtractorModel", e("motion_extractor")),
        "app_feat_extractor": ("AppearanceFeatureExtractorModel", e("appearance_feature_extractor")),
        "landmark": ("LandmarkModel", e("landmark")),
        "face_analysis": ("FaceAnalysisModel", [e("retinaface_det_static"), e("face_2dpose_106_static")]),
        "stitching": ("StitchingModel", e("stitching")),
        "stitching_eye_retarget": ("StitchingModel", e("stitching_eye")),
        "stitching_lip_retarget": ("StitchingModel", e("stitching_lip")),
    }
    out = {}
    for k, (cls, p) in spec.items():
        o = None if isinstance(p, list) else str(src / f"{onnx_of.get(k, k)}.onnx")
        out[k] = getattr(M, cls)(name=cls, predict_type="trt", model_path=p, onnx_path=o)
    # MotionExtractorModel's TensorRT branch unpacks (kp, pitch, yaw, roll, t, exp, scale) (TensorRT 8 engine order)
    me = out["motion_extractor"].predictor
    me.outputs.sort(key=lambda b: ["kp", "pitch", "yaw", "roll", "t", "exp", "scale"].index(b["name"]))
    import onnx
    for pred, n in ((out["face_analysis"].face_det, "retinaface_det_static"),
                    (out["face_analysis"].face_pose, "face_2dpose_106_static")):
        names = [x.name for x in onnx.load(str(src / f"{n}.onnx"), load_external_data=False).graph.output]
        pred.outputs.sort(key=lambda b: names.index(b["name"]))
    return out
