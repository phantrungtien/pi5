#!/usr/bin/env python3
"""Chay TREN PI. Do lop camera->server qua localhost (khong dinh WiFi) + trang
thai WiFi. Moi 5 giay mot dong. Muc dich: khi laptop thay khung hinh khung, biet
duoc luc do phia Pi con san khung deu hay khong."""
import subprocess
import sys
import time
import urllib.request

DUR = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0


def wifi():
    try:
        out = subprocess.run(["iw", "dev", "wlan0", "link"], capture_output=True,
                             text=True, timeout=3).stdout
        sig = rx = tx = "?"
        for ln in out.splitlines():
            ln = ln.strip()
            if ln.startswith("signal:"):
                sig = ln.split()[1]
            elif ln.startswith("rx bitrate:"):
                rx = ln.split()[2]
            elif ln.startswith("tx bitrate:"):
                tx = ln.split()[2]
        return f"sig={sig}dBm rx={rx} tx={tx}"
    except Exception as e:
        return f"wifi? {type(e).__name__}"


def read_frame(resp):
    while True:
        line = resp.readline()
        if not line:
            raise EOFError
        if line.startswith(b"--"):
            break
    length = None
    while True:
        line = resp.readline()
        if not line:
            raise EOFError
        if line in (b"\r\n", b"\n"):
            break
        if line.lower().startswith(b"content-length:"):
            length = int(line.split(b":", 1)[1])
    return resp.read(length)


t_end = time.monotonic() + DUR
resp = None
bucket, gaps, last = 0, [], None
t_win = time.monotonic()
print(f"# bat dau {time.strftime('%H:%M:%S')}, chay {DUR:.0f}s", flush=True)

while time.monotonic() < t_end:
    try:
        if resp is None:
            resp = urllib.request.urlopen("http://127.0.0.1:8080/stream", timeout=5)
            print(f"{time.strftime('%H:%M:%S')}  [noi lai localhost]", flush=True)
        read_frame(resp)
        now = time.monotonic()
        if last is not None:
            gaps.append((now - last) * 1000)
        last = now
        bucket += 1
    except Exception as e:
        print(f"{time.strftime('%H:%M:%S')}  LOI localhost: {type(e).__name__} {e}", flush=True)
        resp = None
        last = None
        time.sleep(1)

    if time.monotonic() - t_win >= 5.0:
        g = sorted(gaps) if gaps else [0]
        print(f"{time.strftime('%H:%M:%S')}  PI  {bucket/5.0:5.1f} fps  "
              f"p50 {g[len(g)//2]:6.0f}ms  MAX {g[-1]:7.0f}ms   {wifi()}", flush=True)
        bucket, gaps = 0, []
        t_win = time.monotonic()
print("# xong", flush=True)
