"""Kiem H264Out: IDR moi ~1 s theo dong ho, ca khi camera tut con 3 Hz (bay 29/10: dong co chay ->
camera bo khung hong; -g 30 cu = IDR moi 10 s, tablet mat hinh lau) lan khi 30 fps (khong duoc
chen 2 IDR sat nhau - ton 4G).

Chay tren Jetson (ffmpeg 4.4 nhu service): python3 test_h264_keyframe.py   (~30 s)
"""
import json, socket, subprocess, tempfile, threading, time

import numpy as np

from run_gui import H264Out


def keyframe_gaps(fps, seconds):
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2.0)
    buf = bytearray()

    def _recv():
        try:
            while True:
                buf.extend(rx.recv(2048))
        except socket.timeout:
            pass

    th = threading.Thread(target=_recv, daemon=True)
    th.start()
    out = H264Out(f"127.0.0.1:{rx.getsockname()[1]}", 900, "superfast")
    # Anh gan nhu dung yen (nen + o vuong troi): anh nhieu ngau nhien lam x264 tu chen I vi
    # "doi canh" (scenecut) -> test PASS ca voi code cu.
    bg = np.tile(np.linspace(0, 255, 960, dtype=np.uint8), (540, 1))[..., None].repeat(3, 2)
    for i in range(fps * seconds):
        f = bg.copy()
        x = (5 * i) % 900
        f[200:260, x:x + 60] = 255
        out.push(f)
        time.sleep(1 / fps)
    time.sleep(1.5)
    out.p.stdin.close()
    out.p.wait()
    th.join()

    with tempfile.NamedTemporaryFile(suffix=".ts") as ts:
        ts.write(buf)
        ts.flush()
        fr = json.loads(subprocess.check_output(
            ["ffprobe", "-v", "error", "-select_streams", "v", "-show_entries",
             "frame=key_frame,pkt_pts_time,best_effort_timestamp_time", "-of", "json",
             ts.name]))["frames"]
    t = [float(x.get("pkt_pts_time") or x["best_effort_timestamp_time"]) for x in fr]
    keys = [ti for ti, x in zip(t, fr) if x["key_frame"]]
    gaps = [round(b - a, 2) for a, b in zip(keys, keys[1:])]
    print(f"{fps} fps: {len(fr)} khung, {len(keys)} IDR, cach nhau {gaps}")
    assert len(fr) >= fps * (seconds - 1), "thieu khung"
    return gaps


for fps, seconds in ((3, 14), (30, 6)):
    g = keyframe_gaps(fps, seconds)
    assert g and max(g) <= 1.5, f"{fps} fps: IDR cach nhau {max(g or [0])} s > 1.5 s"
    assert min(g) >= 0.5, f"{fps} fps: 2 IDR sat nhau ({min(g)} s) - ton bang thong"
print("PASS")
