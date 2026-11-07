"""Kiem Recorder bang luong H.264 MPEG-TS that (~15 s):
- khoi dong xoa file > 3 ngay, giu file moi
- ghi tu luc chay -> dung khi ha canh (DISARM sau ARM) -> ARM lai = file moi
- PTS reset giua luong (H264Out khoi dong lai ffmpeg khi doi bitrate) -> file ghi van tang deu
  (ghi tho thi Windows khong mo duoc, 07/11/2026)
Chay tren Jetson: python3 test_recorder.py
"""
import os, subprocess, tempfile, time

from run_gui import Recorder


def clip(sec):
    """Mot lan chay ffmpeg = PTS bat dau lai tu ~0, nhu H264Out._spawn(). 720p co nhieu + 1200k + PES
    co do dai nhu luong that: khung ~5 KB trai nhieu goi, demux nha khung ngay khi du."""
    return subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
         f"testsrc2=size=1280x720:rate=30:duration={sec},noise=alls=20:allf=t",
         "-c:v", "libx264", "-preset", "superfast", "-tune", "zerolatency", "-g", "30",
         "-b:v", "1200k", "-maxrate", "1200k", "-bufsize", "1200k", "-omit_video_pes_length", "0",
         "-f", "mpegts", "pipe:1"], capture_output=True, check=True).stdout


def feed(r, data, sec):
    """Day nhu _fanout: goi 1128 B (6 goi TS), deu theo thoi gian that."""
    n = len(data) // 1128
    t0 = time.monotonic()
    for i in range(n):
        r.write(data[i * 1128:(i + 1) * 1128])
        time.sleep(max(0.0, t0 + (i + 1) * sec / n - time.monotonic()))


def dts(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries", "packet=dts",
                          "-of", "csv=p=0", path], capture_output=True, text=True).stdout
    return [int(x) for x in out.split() if x.strip().lstrip("-").isdigit()]


def wait_closed(r):
    t = time.monotonic()
    while r.f is not None and time.monotonic() - t < 10:
        time.sleep(0.1)
    assert r.f is None, "ha canh phai dong file"


d = tempfile.mkdtemp()
old, new = os.path.join(d, "20000101_000000.ts"), os.path.join(d, "20000102_000000.ts")
for f in (old, new):
    open(f, "wb").close()
os.utime(old, (time.time() - 4 * 86400,) * 2)
armed = [False]
r = Recorder(d, lambda: armed[0])
assert not os.path.exists(old) and os.path.exists(new), "xoa file > 3 ngay, giu file moi"
os.remove(new)

a, b = clip(3), clip(3)
feed(r, a, 3)                                  # vua bat nguon, chua bay: ghi
armed[0] = True
feed(r, b, 3)                                  # dang bay; PTS cua b bat dau lai tu 0
armed[0] = False
r.write(b"\x47" * 1128)                        # ha canh
wait_closed(r)
files = sorted(os.listdir(d))
assert len(files) == 1, files
t = dts(os.path.join(d, files[0]))
assert len(t) >= 150, f"chi {len(t)} khung / 180"
assert all(y > x for x, y in zip(t, t[1:])), "DTS phai tang ngat"
gaps = [y - x for x, y in zip(t, t[1:])]
# Khung cuoi bi cat do luc ngat ghi -> ffmpeg chi nha ra luc dong file (~SYNC s sau): bo qua no.
assert max(gaps[:-1]) < 45000, f"nhay {max(gaps[:-1]) / 90000:.1f} s"  # khong nhay cho PTS reset
tiny = sum(g < 900 for g in gaps)                                     # < 10 ms
assert tiny < len(gaps) * 0.1, f"{tiny}/{len(gaps)} khung sat nhau"   # goi don dong luc dau file thi co vai

r.write(b"\x47" * 1128)                        # nam tren dat sau ha canh: khong ghi
time.sleep(2.5)
assert r.f is None and len(os.listdir(d)) == 1
time.sleep(1.0)                                # ten file theo giay
armed[0] = True
feed(r, clip(2), 2)                            # bay lan 2: file moi
armed[0] = False
r.write(b"\x47" * 1128)                        # ha canh lan 2 (that: goi den lien tuc)
wait_closed(r)
assert len(os.listdir(d)) == 2, os.listdir(d)
print(f"OK  {len(t)} khung, buoc DTS {min(gaps) / 90:.0f}-{max(gaps) / 90:.0f} ms, {tiny} khung sat nhau")
