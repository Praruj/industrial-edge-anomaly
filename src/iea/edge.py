"""Edge deployment: ONNX export, int8 quantization, size and latency measurement."""
from __future__ import annotations

import os
import platform
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.quantization import (CalibrationDataReader, QuantFormat, QuantType,
                                      quantize_dynamic, quantize_static)


def export_onnx(model: torch.nn.Module, example: torch.Tensor, path: Path, dynamic_batch: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    axes = {"input": {0: "batch"}, "output": {0: "batch"}} if dynamic_batch else None
    torch.onnx.export(model.eval(), example, str(path), input_names=["input"], output_names=["output"],
                      dynamic_axes=axes, opset_version=17)
    return path


class _Reader(CalibrationDataReader):
    def __init__(self, batches: list[np.ndarray]):
        self.it = iter([{"input": b} for b in batches])

    def get_next(self):
        return next(self.it, None)


def quantize_int8_static(src: Path, dst: Path, calib: list[np.ndarray]) -> Path:
    """Static int8 (QDQ, per-channel weights) for conv nets, calibrated on good samples only."""
    quantize_static(str(src), str(dst), _Reader(calib), quant_format=QuantFormat.QDQ,
                    per_channel=True, weight_type=QuantType.QInt8, activation_type=QuantType.QUInt8)
    return dst


def quantize_int8_dynamic(src: Path, dst: Path) -> Path:
    """Dynamic int8 for small dense nets (weights int8, activations quantized at runtime)."""
    quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8)
    return dst


def session(path: Path, threads: int = 1) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])


def run(sess: ort.InferenceSession, x: np.ndarray) -> np.ndarray:
    return sess.run(None, {"input": x})[0]


def latency_ms(fn, warmup: int = 10, iters: int = 100) -> dict:
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(iters):
        s = time.perf_counter(); fn(); t.append((time.perf_counter() - s) * 1000)
    t = np.array(t)
    return {"p50_ms": float(np.median(t)), "p95_ms": float(np.percentile(t, 95))}


def size_mb(*paths: Path) -> float:
    return sum(os.path.getsize(p) for p in paths) / 1e6


def device_info() -> dict:
    return {"cpu": platform.processor() or platform.machine(), "os": platform.platform(),
            "onnxruntime": ort.__version__, "threads": 1,
            "note": "Laptop CPU limited to one thread as a stand-in for an edge device."}
