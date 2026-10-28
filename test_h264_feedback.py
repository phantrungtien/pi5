#!/usr/bin/env python3
"""python3 test_h264_feedback.py - dem mat goi TS + dieu khien bitrate H264Out."""
import sys, pathlib
sys.path[:0] = [str(pathlib.Path(__file__).parent), str(pathlib.Path(__file__).parent / "tracking_pi5")]
from h264_feedback import count_loss


def ts(pid, cc, payload=True):
    return bytes([0x47, pid >> 8, pid & 0xFF, (0x10 if payload else 0x20) | cc]) + bytes(184)


last = {}
assert count_loss(ts(256, 0) + ts(256, 1), last) == (2, 0)
assert count_loss(ts(256, 4), last) == (1, 2)               # mat cc 2,3
assert count_loss(ts(256, 4), last) == (1, 0)               # goi lap
assert count_loss(ts(256, 0), last) == (1, 11)              # quay vong 4 -> 0: mat 5..15
assert count_loss(ts(256, 9, payload=False) + ts(0x1FFF, 3), last) == (0, 0)
assert count_loss(ts(257, 7), last) == (1, 0)               # PID moi

import threading
threading.Thread.start = lambda self: None                   # khong chay ffmpeg/socket
from run_gui import H264Out
h = H264Out("127.0.0.1:1", 1500, "superfast", kbps_min=400)
assert h.adapt(0.10, 10.0) == 1050                           # mat -> x0.7
assert h.adapt(0.10, 11.0) == 1050                           # chua du DOWN_GAP
assert h.adapt(0.10, 12.0) == 735
assert h.adapt(0.0, 15.0) == 735                             # chua du UP_AFTER
assert h.adapt(0.0, 17.0) == 885                             # +150
for t in range(30, 300, 6):
    h.adapt(0.0, t)
assert h.kbps == 1500                                        # tran = --h264-kbps
for t in range(300, 340, 3):
    h.adapt(1.0, t)
assert h.kbps == 400                                         # san = --h264-kbps-min

import run_gui                                               # bat ffmpeg = CC ve 0 -> bo qua bao mat gia
run_gui.subprocess.Popen = lambda *a, **k: None
h2 = H264Out("127.0.0.1:1", 1500, "superfast")
h2.size = (2, 2)
h2._spawn()
assert h2.adapt(0.10, h2.last_change + 1.0) == 1500          # vua khoi dong service: khong ha
assert h2.adapt(0.10, h2.last_change + 2.0) == 1050          # het DOWN_GAP: mat that thi ha

import numpy as np                                           # tu phoi sang uu tien vung sang
from run_gui import HighlightAE
ae = HighlightAE.__new__(HighlightAE)
ae.e_max, ae.e_min, ae.e = 100, 3, 50.0
img = lambda med, clip: np.r_[np.full(int(1000 * (1 - clip)), med), np.full(int(1000 * clip), 255)].astype(np.uint8)
assert ae.step(img(120, 0.05)) < 50                          # den chay 5% -> toi di
assert ae.step(img(40, 0.05)) > 50                           # ca khung qua toi -> sang len du den chay
assert ae.step(img(120, 0.0)) > 50                           # khong chay gi -> sang len
assert ae.step(img(120, 0.01)) == 50                         # trong vung on dinh -> giu
assert HighlightAE.__init__.__defaults__ == (1,)             # san 100 us: XC-TECH 4K min = 1, khong phai 3
ae.e = 3.0
assert ae.step(img(30, 0.0)) > 3.4                           # khong ket o e nho
ae.e = 100.0
assert ae.step(img(30, 0.0)) > 100                           # o e_max van toi -> sang tiep bang gain
ae.e = 200.0
assert ae.step(img(30, 0.0)) == 200                          # tran = e_max + G_MAX
assert ae.split(80.0) == (80, 1) and ae.split(130.0) == (100, 30)   # exposure truoc, gain sau
assert ae.split(100.3) == (100, 1)                           # bay 27/10: gain 0 = XC-TECH tu chinh gain -> trang
print("OK")
