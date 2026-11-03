"""Minimal TensorRT 10/11 runtime on torch CUDA tensors (name-based I/O, execute_async_v3, optional CUDA graph)."""

from pathlib import Path

import tensorrt as trt
import torch

_DT = {trt.float32: torch.float32, trt.float16: torch.float16, trt.int32: torch.int32, trt.int64: torch.int64,
       trt.bool: torch.bool, trt.int8: torch.int8, trt.uint8: torch.uint8}
if hasattr(trt, "bfloat16"):
    _DT[trt.bfloat16] = torch.bfloat16
_LOG = trt.Logger(trt.Logger.ERROR)


class Engine:
    """Static-shape engine with preallocated I/O buffers. `run(**inputs)` copies inputs in and returns the output
    buffers (views, overwritten by the next call: clone if you keep them)."""

    def __init__(self, path, stream=None):
        self.path = Path(path)
        self.engine = trt.Runtime(_LOG).deserialize_cuda_engine(self.path.read_bytes())
        assert self.engine is not None, path
        self.ctx = self.engine.create_execution_context()
        self.stream = stream or torch.cuda.Stream()
        self.inputs, self.outputs, self.buf = [], [], {}
        for i in range(self.engine.num_io_tensors):
            n = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(n))
            t = torch.empty(shape, dtype=_DT[self.engine.get_tensor_dtype(n)], device="cuda")
            self.buf[n] = t
            self.ctx.set_tensor_address(n, t.data_ptr())
            (self.inputs if self.engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT else self.outputs).append(n)
        self.graph = None

    def _launch(self):
        self.ctx.execute_async_v3(self.stream.cuda_stream)

    def run(self, *args, sync=False, **kw):
        feeds = dict(zip(self.inputs, args)) | kw
        cur = torch.cuda.current_stream()
        self.stream.wait_stream(cur)
        with torch.cuda.stream(self.stream):
            for n, v in feeds.items():
                self.buf[n].copy_(v, non_blocking=True)
            if self.graph is not None:
                self.graph.replay()
            else:
                self._launch()
        cur.wait_stream(self.stream)
        if sync:
            torch.cuda.synchronize()
        return {n: self.buf[n] for n in self.outputs}

    def capture_graph(self):
        """Capture the engine launch in a CUDA graph (saves launch overhead on small engines)."""
        self._launch()
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=self.stream):
            self._launch()
        self.graph = g
