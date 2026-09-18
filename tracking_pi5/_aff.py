"""Pose 3 luong + lua 1 luong tren 4 loi: co ghim loi thi hon khong?

Lua chi chay 3 lan/giay ma lam pose cham 15 ms/khung. Neu la do trinh dieu phoi
day luong lua vao dung loi dang chay pose, thi ghim moi ben mot coi se het.
"""
import os, threading, time
import numpy as np, onnxruntime as ort

D = "/app/tracking_pi5/yolo11n-pose-320.onnx"
F = "/app/fire-models/best_nano_111_320.onnx"


def sess(path, threads):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    i = s.get_inputs()[0]
    sh = [d if isinstance(d, int) else 1 for d in i.shape]
    return s, i.name, np.random.rand(*sh).astype(np.float32)


def do(nhan, ghim, secs=12):
    if ghim:
        os.sched_setaffinity(0, {0, 1, 2})      # pose: loi 0-2
    else:
        os.sched_setaffinity(0, {0, 1, 2, 3})
    ps, pn, px = sess(D, 3)
    ps.run(None, {pn: px}); ps.run(None, {pn: px})
    stop = threading.Event()

    def lua():
        if ghim:
            os.sched_setaffinity(0, {3})        # lua: rieng loi 3
        fs, fn, fx = sess(F, 1)
        fs.run(None, {fn: fx})
        while not stop.is_set():
            t = time.perf_counter()
            fs.run(None, {fn: fx})
            time.sleep(max(0, 1 / 3 - (time.perf_counter() - t)))

    threading.Thread(target=lua, daemon=True).start()
    time.sleep(1.0)
    t0, n, lat = time.perf_counter(), 0, []
    while time.perf_counter() - t0 < secs:
        a = time.perf_counter(); ps.run(None, {pn: px}); lat.append(time.perf_counter() - a)
        n += 1
    dt = time.perf_counter() - t0
    stop.set(); time.sleep(0.4)
    lat.sort()
    print(f"{nhan:<30} {n/dt:5.2f} fps   p50 {lat[len(lat)//2]*1000:6.1f} ms  "
          f"p90 {lat[int(len(lat)*0.9)]*1000:6.1f} ms", flush=True)
    os.sched_setaffinity(0, {0, 1, 2, 3})


do("khong ghim (nhu hien tai)", False)
do("ghim: pose 0-2, lua 3", True)
do("khong ghim (do lai)", False)
