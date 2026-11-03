"""Build TensorRT engines for LivePortrait (FasterLivePortrait ONNX files) on Blackwell (sm_120), no plugin.

FasterLivePortrait needs TensorRT 8 + a compiled grid_sample_3d plugin; TensorRT 8 has no sm_120 support and the
plugin needs nvcc. TensorRT >= 10.x parses the standard ONNX GridSample (opset 20) on 5-D input natively, so we
rewrite the two custom `GridSample3D` nodes of warping_spade-fix.onnx into `GridSample(mode=linear)` and bump the
opset; numerically identical to torch.nn.functional.grid_sample (max abs err 5e-7 in a test).

  cloud/facefix/.venv/bin/python cloud/lpfast/build_engines.py [--fp32] [--only warping_spade ...]
Engines go to cloud/lpfast/checkpoints/trt/<name>.engine (GPU-specific; rebuild per GPU / TensorRT version).
"""

import argparse
import time
from pathlib import Path

import onnx
import tensorrt as trt

HERE = Path(__file__).resolve().parent
SRC = HERE / "checkpoints" / "liveportrait_onnx"
DST = HERE / "checkpoints" / "trt"
MODELS = ["warping_spade", "appearance_feature_extractor", "motion_extractor", "stitching", "stitching_eye",
          "stitching_lip", "landmark", "retinaface_det_static", "face_2dpose_106_static"]


def patched_warping():
    out = DST / "warping_spade_gs20.onnx"
    if out.exists():
        return out
    m = onnx.load(SRC / "warping_spade-fix.onnx")
    n = 0
    for node in m.graph.node:
        if node.op_type == "GridSample3D":
            node.op_type, node.domain = "GridSample", ""
            for a in node.attribute:
                if a.name == "mode":
                    a.s = b"linear"  # opset >= 20 name for (tri)linear
            n += 1
    assert n == 2, n
    for o in m.opset_import:
        if o.domain in ("", "ai.onnx"):
            o.version = 20
    onnx.save(m, out)
    print(f"patched {n} GridSample3D -> GridSample (opset 20): {out}")
    return out


def build(name, fp16, workspace_gb=4):
    onnx_path = patched_warping() if name == "warping_spade" else SRC / f"{name}.onnx"
    eng_path = DST / f"{name}{'' if fp16 else '_fp32'}.engine"
    log = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(log)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, log)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError(f"{name}: " + "; ".join(parser.get_error(i).desc() for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    if fp16 and hasattr(trt.BuilderFlag, "FP16"):
        cfg.set_flag(trt.BuilderFlag.FP16)
    t0 = time.time()
    data = builder.build_serialized_network(net, cfg)
    if data is None:
        raise RuntimeError(f"{name}: engine build failed")
    eng_path.write_bytes(bytes(data))
    ios = [(net.get_input(i).name, tuple(net.get_input(i).shape)) for i in range(net.num_inputs)]
    print(f"{name}: {time.time() - t0:.0f}s -> {eng_path.name} ({eng_path.stat().st_size / 1e6:.0f} MB) inputs {ios}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fp32", action="store_true")
    p.add_argument("--only", nargs="*")
    a = p.parse_args()
    DST.mkdir(parents=True, exist_ok=True)
    print("TensorRT", trt.__version__)
    for name in a.only or MODELS:
        build(name, fp16=not a.fp32)


if __name__ == "__main__":
    main()
