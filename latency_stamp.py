#!/usr/bin/env python3
"""Dau thoi gian trong khung video de do do tre chup -> giai ma.

Drone (run_gui.py --stamp): ve dai o den/trang goc duoi trai = gio chup (ms, 40 bit) + checksum.
Tram mat dat: doc lai dai o, tre = gio luc giai ma xong - gio chup.

    python3 latency_stamp.py udp://@:5610        # sau h264_feedback.py (no chiem 5600)
    python3 latency_stamp.py udp://@:5600        # khong chay h264_feedback

Hai dong ho lech bao nhieu ms thi so do lech bay nhieu. Them user@host cua drone de tu do
do lech qua ssh va bu (1 lan hoi mat khau):

    python3 latency_stamp.py udp://@:5600 robomango@100.117.32.3
So do KHONG gom thoi gian man hinh hien thi (~1 khung ve 60 Hz) va phoi sang/USB camera.
"""
import sys, time

import numpy as np

CELL, COLS, ROWS = 16, 24, 2               # o 16 px song sot x264 400 kbps; 384x32 vua ca khung 640
BITS = COLS * ROWS                         # 40 bit ms + 8 bit checksum
MASK = (1 << 40) - 1                       # ms quay vong ~35 nam, hai dau cung cat nhu nhau


def _sum(v):
    # dao bit: khung den tron (v=0, s=0) khong lot qua checksum
    return ~sum(v.to_bytes(5, "big")) & 0xFF


def _bits(v):
    v &= MASK
    s = _sum(v)
    x = (v << 8) | s
    return [(x >> (BITS - 1 - i)) & 1 for i in range(BITS)]


def stamp(frame, t_wall):
    """Ve gio t_wall (giay epoch) vao goc duoi trai, ghi de len frame."""
    h = frame.shape[0]
    y0 = h - ROWS * CELL
    for i, b in enumerate(_bits(int(t_wall * 1000))):
        r, c = divmod(i, COLS)
        frame[y0 + r * CELL:y0 + (r + 1) * CELL, c * CELL:(c + 1) * CELL] = 255 if b else 0


def read(frame):
    """Tra ve gio chup (giay epoch) hoac None neu dai o hong/khong co."""
    g = frame.mean(axis=2) if frame.ndim == 3 else frame
    h = g.shape[0]
    if h < ROWS * CELL or g.shape[1] < COLS * CELL:
        return None
    y0, q = h - ROWS * CELL, CELL // 4    # lay tam o, tranh vien bi nhoe
    x = 0
    for i in range(BITS):
        r, c = divmod(i, COLS)
        y, xx = y0 + r * CELL, c * CELL
        x = (x << 1) | int(g[y + q:y + CELL - q, xx + q:xx + CELL - q].mean() > 128)
    v, s = x >> 8, x & 0xFF
    if _sum(v) != s:
        return None
    # khoi phuc phan cao bi cat MASK tu dong ho may nay
    now_ms = int(time.time() * 1000)
    ms = (now_ms & ~MASK) | v
    if ms > now_ms + (MASK >> 1):
        ms -= MASK + 1
    return ms / 1000


def clock_offset(cmd, n=20):
    """Dong ho may kia - dong ho may nay (giay), kieu NTP qua mot tien trinh hoi-dap.
    Lay mau RTT nho nhat: sai so <= RTT/2 (LAN ~2 ms)."""
    import subprocess
    p = subprocess.Popen(cmd + ["python3 -u -c 'import sys,time; [print(time.time(), flush=True) for _ in sys.stdin]'"],
                         stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    best = (float("inf"), 0.0)
    for _ in range(n):
        t0 = time.time()
        p.stdin.write("\n"); p.stdin.flush()
        tr = float(p.stdout.readline())
        t1 = time.time()
        best = min(best, (t1 - t0, tr - (t0 + t1) / 2))
    p.stdin.close(); p.wait()
    return best[1], best[0]


def main(url, ssh=None):
    off = 0.0
    if ssh:
        off, rtt = clock_offset(["ssh", ssh])
        print(f"[tre] dong ho {ssh} lech {off * 1000:+.1f} ms (RTT {rtt * 1000:.1f} ms) - da bu", flush=True)
    # ffmpeg low-delay chu khong cv2.VideoCapture: do 16/10/2026 cv2 tu cong them ~170 ms
    # (300 vs 120 ms cung mot luong) - do bang cv2 la do cai dau thu chu khong phai duong truyen.
    import subprocess
    # chi cat dai o ra Python: chuyen ca khung 720p sang BGR khong kip -> tre tang ~1 s/s (do)
    w, h = COLS * CELL, ROWS * CELL
    p = subprocess.Popen(["ffmpeg", "-loglevel", "fatal", "-fflags", "nobuffer", "-flags", "low_delay",
                          "-probesize", "32", "-analyzeduration", "0", "-i", url,
                          "-vf", f"crop={w}:{h}:0:ih-{h}", "-vsync", "passthrough", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                         stdout=subprocess.PIPE)
    n_b = w * h * 3
    lat, bad, t_rep = [], 0, time.monotonic() + 5
    while True:
        buf = p.stdout.read(n_b)
        if len(buf) < n_b:
            sys.exit("ffmpeg dung - sai url?")
        t = read(np.frombuffer(buf, np.uint8).reshape(h, w, 3))
        if t is None:
            bad += 1
        else:
            lat.append((time.time() + off - t) * 1000)
        if time.monotonic() >= t_rep:
            t_rep += 5
            if lat:
                lat.sort()
                n = len(lat)
                print(f"[tre] {n} khung  p50 {lat[n // 2]:.0f} ms  p90 {lat[int(n * 0.9)]:.0f} ms"
                      f"  min {lat[0]:.0f}  max {lat[-1]:.0f}  (doc hong {bad})", flush=True)
            else:
                print(f"[tre] khong doc duoc dau thoi gian ({bad} khung) - thieu --stamp?",
                      flush=True)
            lat, bad = [], 0


def _selftest():
    t = time.time()
    f = np.full((540, 960, 3), 90, np.uint8)
    stamp(f, t)
    assert abs(read(f) - t) < 0.001
    f[-1, 0, :] ^= 0xFF; f[-20, 5 * CELL + 8, :] ^= 0xFF          # nhieu le te van doc dung
    assert abs(read(f) - t) < 0.001
    y, x = f.shape[0] - ROWS * CELL, 3 * CELL                       # hong 1 o -> checksum bat
    f[y:y + CELL, x:x + CELL] = 255 - f[y:y + CELL, x:x + CELL]
    assert read(f) is None
    assert read(np.zeros((540, 960, 3), np.uint8)) is None             # khong co dai o
    assert read(np.full((540, 960, 3), 255, np.uint8)) is None
    off, rtt = clock_offset(["bash", "-c"])                 # thay ssh bang bash cung may
    assert abs(off) < 0.01 and rtt < 0.1                      # cung may: lech ~0
    print("OK")


if __name__ == "__main__":
    if sys.argv[1:] == ["--test"]:
        _selftest()
    else:
        main(*(sys.argv[1:] or ["udp://@:5610"]))
