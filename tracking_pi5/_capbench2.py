"""Tran duong ong KHI KHONG CO MODEL: doc gst, roi doc+nen JPEG."""
import subprocess, time
import numpy as np, cv2
W, H, N = 960, 540, 960 * 540 * 3

def doc_du(f, n):
    b = b""
    while len(b) < n:
        c = f.read(n - len(b))
        if not c:
            return None
        b += c
    return b

def chay(nhan, nen, fps=15, secs=12):
    p = subprocess.Popen(
        ["gst-launch-1.0", "-q", "v4l2src", "device=/dev/video0", "!",
         f"image/jpeg,width={W},height={H},framerate={fps}/1", "!",
         "jpegdec", "!", "videoconvert", "!", "video/x-raw,format=BGR", "!",
         "fdsink", "fd=1", "sync=false"], stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL)
    for _ in range(10):
        if doc_du(p.stdout, N) is None:
            print(nhan, "gst chet", flush=True); return
    t0, k, cpu0 = time.perf_counter(), 0, time.process_time()
    while time.perf_counter() - t0 < secs:
        b = doc_du(p.stdout, N)
        if b is None:
            break
        if nen:
            f = np.frombuffer(b, np.uint8).reshape(H, W, 3)
            cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 70])
        k += 1
    dt = time.perf_counter() - t0
    cpu = (time.process_time() - cpu0) / max(k, 1) * 1000
    p.terminate()
    print(f"{nhan:<34} {k/dt:5.1f} fps   CPU {cpu:5.1f} ms/khung", flush=True)

chay("chi doc gst (jpegdec) 15Hz", False, 15)
chay("doc + nen JPEG q70   15Hz", True, 15)
chay("doc + nen JPEG q70   12Hz", True, 12)
