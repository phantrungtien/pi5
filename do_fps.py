import cv2, time
for want in (0, 10, 30):
    c = cv2.VideoCapture("/dev/video0")
    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    c.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    c.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    c.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if want:
        c.set(cv2.CAP_PROP_FPS, want)
    c.read()
    ts = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < 6.0:
        ok, f = c.read()
        if ok:
            ts.append(time.monotonic())
    g = sorted((ts[i] - ts[i-1]) * 1000 for i in range(1, len(ts)))
    n = len(g)
    lbl = want if want else "mac dinh"
    print(f"dat={lbl} -> driver bao {c.get(cv2.CAP_PROP_FPS):.0f} | THUC {len(ts)/6.0:.1f} fps | p50 {g[n//2]:.0f}ms p90 {g[int(n*0.9)]:.0f}ms max {g[-1]:.0f}ms")
    c.release()
