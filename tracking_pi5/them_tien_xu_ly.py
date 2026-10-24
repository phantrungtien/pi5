#!/usr/bin/env python3
"""Gan tien xu ly vao DAU model ONNX: nhan anh uint8 BGR [1,H,W,3] thay vi float NCHW.

    python3 them_tien_xu_ly.py yolo11n-pose-1280x736-raw.onnx   # -> ...-raw-u8.onnx

Vi sao: o 1280x736, doi BGR->RGB + /255 + HWC->NCHW bang numpy/cv2 mat 15-40 ms CPU
moi khung (vuot ngan sach 33 ms cua 30 fps). Dua vao graph thi TensorRT lam tren GPU,
CPU chi con letterbox (~2 ms). track.Detector tu nhan ra input uint8.
"""
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

src = sys.argv[1]
m = onnx.load(src)
g = m.graph
old = g.input[0]
_, c, h, w = [d.dim_value for d in old.type.tensor_type.shape.dim]
assert c == 3, old

g.initializer.extend([
    numpy_helper.from_array(np.array([2, 1, 0], np.int64), "pre_bgr2rgb"),
    numpy_helper.from_array(np.array(1 / 255.0, np.float32), "pre_scale"),
])
pre = [
    helper.make_node("Gather", ["images_u8", "pre_bgr2rgb"], ["pre_rgb"], axis=3),
    helper.make_node("Cast", ["pre_rgb"], ["pre_f"], to=TensorProto.FLOAT),
    helper.make_node("Mul", ["pre_f", "pre_scale"], ["pre_norm"]),
    helper.make_node("Transpose", ["pre_norm"], [old.name], perm=[0, 3, 1, 2]),
]
nodes = pre + list(g.node)
del g.node[:]
g.node.extend(nodes)
g.input.remove(old)
g.input.insert(0, helper.make_tensor_value_info("images_u8", TensorProto.UINT8, [1, h, w, 3]))
onnx.checker.check_model(m)
dst = src.replace(".onnx", "-u8.onnx")
onnx.save(m, dst)
print(dst)
