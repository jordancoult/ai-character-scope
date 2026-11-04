"""Build TensorRT engines for LivePortrait (FasterLivePortrait ONNX files) on Blackwell (sm_120), no plugin.

FasterLivePortrait needs TensorRT 8 + a compiled grid_sample_3d plugin; TensorRT 8 has no sm_120 support and the
plugin needs nvcc. TensorRT >= 10.x parses the standard ONNX GridSample (opset 20) on 5-D input natively, so we
rewrite the two custom `GridSample3D` nodes of warping_spade-fix.onnx into `GridSample(mode=linear)` and bump the
opset; numerically identical to torch.nn.functional.grid_sample (max abs err 5e-7 in a test).

  cloud/facefix/.venv/bin/python cloud/lpfast/build_engines.py [--fp32] [--only warping_spade ...]
The warp is built in FP16 (warping_fp16.engine, the one lp_pass.py uses); the small models in FP32.
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
MODELS = ["faceparse", "warping_spade", "appearance_feature_extractor", "motion_extractor", "stitching", "stitching_eye",
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


FP32_OPS = ["GridSample", "InstanceNormalization", "Exp", "Softmax", "ReduceSum", "Pow", "Div"]


def fp16_warping():
    """TensorRT >= 11 has no FP16 builder flag (strongly typed): convert the ONNX itself, sensitive ops in FP32.
    22.5 ms vs 64 ms FP32 on the RTX 5080 Laptop; max output diff 0.022 (mean 0.001, 0..1 scale)."""
    out = DST / "warping_spade_fp16.onnx"
    if not out.exists():
        from onnxconverter_common import float16
        src = patched_warping()
        onnx.save(float16.convert_float_to_float16(onnx.load(src), keep_io_types=True, op_block_list=FP32_OPS), out)
        src.unlink()
    return out


def faceparse_onnx():
    """BiSeNet face parsing (CelebAMask-HQ 19 classes) for the mouth-cover gate: HF bluefoxcreation/Face_parsing_onnx,
    main output only, FP16 (0.6 ms on the RTX 5080 Laptop)."""
    out = DST / "faceparse_fp16.onnx"
    if not out.exists():
        from huggingface_hub import hf_hub_download
        from onnxconverter_common import float16
        p = hf_hub_download("bluefoxcreation/Face_parsing_onnx", "faceparser.onnx",
                            local_dir=str(HERE / "checkpoints" / "faceparse"), token=False)
        m = onnx.load(p)
        keep = [o for o in m.graph.output if o.name == "out"]
        del m.graph.output[:]
        m.graph.output.extend(keep)
        onnx.save(float16.convert_float_to_float16(m, keep_io_types=True, op_block_list=["Softmax", "ReduceMean",
                                                                                         "Sigmoid"]), out)
    return out


def build(name, fp16, workspace_gb=4):
    if name == "faceparse":
        onnx_path, eng_path = faceparse_onnx(), DST / "faceparse_fp16.engine"
    elif name == "warping_spade":
        onnx_path, eng_path = (fp16_warping(), DST / "warping_fp16.engine") if fp16 else \
            (patched_warping(), DST / "warping_spade_fp32.engine")
    else:  # small models: FP32 (each 0.2-3 ms)
        onnx_path, eng_path = SRC / f"{name}.onnx", DST / f"{name}.engine"
    log = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(log)
    net = builder.create_network(0)
    parser = trt.OnnxParser(net, log)
    if not parser.parse_from_file(str(onnx_path)):
        raise RuntimeError(f"{name}: " + "; ".join(parser.get_error(i).desc() for i in range(parser.num_errors)))
    cfg = builder.create_builder_config()
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)
    t0 = time.time()
    data = builder.build_serialized_network(net, cfg)
    if data is None:
        raise RuntimeError(f"{name}: engine build failed")
    eng_path.write_bytes(bytes(data))
    ios = [(net.get_input(i).name, tuple(net.get_input(i).shape)) for i in range(net.num_inputs)]
    print(f"{name}: {time.time() - t0:.0f}s -> {eng_path.name} ({eng_path.stat().st_size / 1e6:.0f} MB) inputs {ios}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fp32", action="store_true", help="warp in FP32 (3x slower; for comparisons)")
    p.add_argument("--only", nargs="*")
    a = p.parse_args()
    DST.mkdir(parents=True, exist_ok=True)
    print("TensorRT", trt.__version__)
    for name in a.only or MODELS:
        build(name, fp16=not a.fp32)


if __name__ == "__main__":
    main()
