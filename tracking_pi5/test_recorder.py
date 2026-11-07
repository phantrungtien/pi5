"""Kiem Recorder: ghi tu luc chay -> dung khi ha canh (DISARM sau ARM) -> ARM lai = file moi;
khoi dong xoa file > 3 ngay, giu file moi. Chay: python3 test_recorder.py   (~1 s)
"""
import os, tempfile, time

from run_gui import Recorder

d = tempfile.mkdtemp()
old, new = os.path.join(d, "20000101_000000.ts"), os.path.join(d, "20000102_000000.ts")
for f in (old, new):
    open(f, "wb").close()
os.utime(old, (time.time() - 4 * 86400,) * 2)
armed = [False]
r = Recorder(d, lambda: armed[0])
assert not os.path.exists(old) and os.path.exists(new), "xoa file > 3 ngay, giu file moi"

pkt = b"\x47" * 1128
r.write(pkt)                                   # vua bat nguon, chua bay: ghi
armed[0] = True
r.write(pkt)                                   # dang bay: ghi
assert r.f and os.path.getsize(r.f.name) == 2 * 1128
first = r.f.name
armed[0] = False
r.write(pkt)                                   # ha canh: dung, dong file
assert r.f is None and os.path.getsize(first) == 2 * 1128
r.write(pkt)                                   # nam tren dat sau ha canh: khong ghi
assert r.f is None
time.sleep(1.1)                                # ten file theo giay
armed[0] = True
r.write(pkt)                                   # bay lan 2: file moi
assert r.f and r.f.name != first and os.path.getsize(r.f.name) == 1128
print("OK")
