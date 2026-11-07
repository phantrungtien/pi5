"""Kiem Recorder4K bang khung JPEG 4K that (ffmpeg testsrc2, ~20 s):
- chua ARM: khong ghi gi; ARM: file *_4k.mkv; DISARM: dong file; ARM lai: file moi
- file DANG ghi (nhu rut pin giua chuyen bay) van doc duoc
- doc lai: MJPEG 3840x2160, dai dung thoi gian that (dau thoi gian theo luc toi)
Chay: python3 test_recorder4k.py   (laptop hay Jetson, can ffmpeg)
"""
import json, os, shutil, subprocess, tempfile, time

from run_gui import Recorder4K


def frames(n):
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-f", "lavfi", "-i",
         f"testsrc2=size=3840x2160:rate=30,noise=alls=10:allf=t", "-frames:v", str(n),
         "-c:v", "mjpeg", "-q:v", "4", "-f", "image2pipe", "pipe:1"], capture_output=True, check=True).stdout
    return [b"\xff\xd8" + p for p in raw.split(b"\xff\xd8")[1:]]


def probe(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v", "-count_packets",
                          "-show_entries", "stream=codec_name,width,height,nb_read_packets:format=duration",
                          "-of", "json", path], capture_output=True, text=True).stdout
    j = json.loads(out)
    return j["streams"][0], float(j["format"].get("duration", 0))


def feed(r, fr, sec):
    t0 = time.monotonic()
    i = 0
    while time.monotonic() - t0 < sec:
        r.write(fr[i % len(fr)])
        i += 1
        time.sleep(max(0.0, t0 + i / 30 - time.monotonic()))


d = tempfile.mkdtemp()
armed = [False]
r = Recorder4K(d, lambda: armed[0])
fr = frames(30)
mb = sum(map(len, fr)) / len(fr) / 1e6
feed(r, fr, 1.0)
time.sleep(0.5)
assert not os.listdir(d), "chua ARM ma da ghi"

armed[0] = True
feed(r, fr, 3.0)
files = sorted(os.listdir(d))
assert len(files) == 1 and files[0].endswith("_4k.mkv"), files
part = os.path.join(d, "dang_ghi.mkv")
shutil.copy(os.path.join(d, files[0]), part)          # rut pin luc nay: file do mo duoc khong
st, _ = probe(part)
assert st["codec_name"] == "mjpeg" and int(st["nb_read_packets"]) > 30, st
os.remove(part)

armed[0] = False
feed(r, fr, 0.5)                                       # gst_zoom thoi day khung sau DISARM
time.sleep(r.SYNC + 1.5)
st, dur = probe(os.path.join(d, files[0]))
assert (st["width"], st["height"]) == (3840, 2160), st
assert 2.5 < dur < 4.0 and 75 < int(st["nb_read_packets"]) <= 95, (dur, st)

time.sleep(1.1)                                        # ten file theo giay
armed[0] = True
feed(r, fr, 1.0)
armed[0] = False
time.sleep(r.SYNC + 1.5)
assert len(os.listdir(d)) == 2, os.listdir(d)
print(f"OK  khung {mb:.2f} MB, file 3 s: {dur:.2f} s, {st['nb_read_packets']} khung 4K")
