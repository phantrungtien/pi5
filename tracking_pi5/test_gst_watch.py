#!/usr/bin/env python3
"""python3 test_gst_watch.py - GstCap._watch: camera con day khung JPEG hong thi KHONG dung lai gst."""
import sys, pathlib, types
sys.path[:0] = [str(pathlib.Path(__file__).parent)]
sys.modules.setdefault("mjpeg_server", types.ModuleType("mjpeg_server"))   # run_gui import luc nap
sys.modules.setdefault("latency_stamp", types.ModuleType("latency_stamp"))
from run_gui import restart_due

assert not restart_due(1.0, 1.5, 99, True)          # chua qua han
assert restart_due(2.0, 1.5, 99, True)              # im that (khong khung nao, ke ca hong): dung lai
assert not restart_due(2.0, 1.5, 0.1, True)         # 29/10: chi khung hong -> doi, khong dung lai
assert restart_due(11.0, 1.5, 0.1, True)            # hong lien 11 s: van dung lai
assert restart_due(25.0, 20.0, 0.1, False)          # chua tung co khung (first_grace): giu nhu cu
assert restart_due(0.5, 0, 0.1, True)               # vua cam camera vao (han 0): mo ngay
print("test_gst_watch OK")
