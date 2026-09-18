"""Hai session ONNX tranh 4 loi — dung nhu production. Do anh huong cua spin-wait.

ONNX Runtime cho luong tho QUAY TIT (busy-wait) cho viec sau moi lan chay. Mot
session thi khong sao; hai session tren 4 loi thi luong dang quay cua ben nay an
mat loi cua ben kia. `session.intra_op.allow_spinning=0` bat chung ngu.
"""
import threading, time
import numpy as np, onnxruntime as ort

D = "/app/tracking_pi5/yolo11n-pose-320.onnx"
F = "/app/fire-models/best_nano_111_320.onnx"


def sess(path, threads, spin):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if not spin:
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    i = s.get_inputs()[0]
    sh = [d if isinstance(d, int) else 1 for d in i.shape]
    return s, i.name, np.random.rand(*sh).astype(np.float32)


def do(nhan, spin, co_lua, secs=12):
    ps, pn, px = sess(D, 3, spin)
    ps.run(None, {pn: px}); ps.run(None, {pn: px})
    stop = threading.Event()

    def lua():
        fs, fn, fx = sess(F, 1, spin)
        fs.run(None, {fn: fx})
        while not stop.is_set():          # 3 Hz nhu --fire-hz 3
            t = time.perf_counter()
            fs.run(None, {fn: fx})
            time.sleep(max(0, 1/3 - (time.perf_counter() - t)))

    th = threading.Thread(target=lua, daemon=True)
    if co_lua:
        th.start(); time.sleep(1.0)
    t0, n, lat = time.perf_counter(), 0, []
    while time.perf_counter() - t0 < secs:
        a = time.perf_counter(); ps.run(None, {pn: px}); lat.append(time.perf_counter() - a)
        n += 1
    dt = time.perf_counter() - t0
    stop.set(); time.sleep(0.4)
    lat.sort()
    print(f"{nhan:<34} {n/dt:5.2f} fps   pose p50 {lat[len(lat)//2]*1000:6.1f} ms  "
          f"p90 {lat[int(len(lat)*0.9)]*1000:6.1f} ms", flush=True)


print("=== POSE MOT MINH (tran) ===", flush=True)
do("spin bat (mac dinh)", True, False)
do("spin tat", False, False)
print("=== POSE + LUA 3 Hz (nhu that) ===", flush=True)
do("spin bat (mac dinh)", True, True)
do("spin tat", False, True)
