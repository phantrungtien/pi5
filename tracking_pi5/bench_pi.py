import numpy as np, onnxruntime as ort, time
D = "/app/tracking_pi5/"; F = "/app/fire-models/"

def bench(path, n=15, threads=4, shape=None):
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    i = s.get_inputs()[0]
    sh = [d if isinstance(d, int) else 1 for d in (shape or i.shape)]
    x = np.random.rand(*sh).astype(np.float32)
    s.run(None, {i.name: x}); s.run(None, {i.name: x})
    t = []
    for _ in range(n):
        a = time.perf_counter(); s.run(None, {i.name: x}); t.append(time.perf_counter() - a)
    return float(np.median(t) * 1000), sh[2]

print("=== POSE NGUOI (4 luong) ===", flush=True)
for f in ("yolo11n-pose-320.onnx", "yolo11n-pose-416.onnx", "yolo11n-pose.onnx"):
    ms, sz = bench(D + f)
    print(f"  {f:24s} imgsz={sz:3d}  {ms:6.1f} ms  -> tran {1000/ms:4.1f} fps", flush=True)
print("=== POSE 320 theo so luong ===", flush=True)
for th in (1, 2, 3, 4):
    ms, _ = bench(D + "yolo11n-pose-320.onnx", threads=th)
    print(f"  {th} luong: {ms:6.1f} ms  -> tran {1000/ms:4.1f} fps", flush=True)
print("=== ReID osnet_x0_25 (4 luong) ===", flush=True)
for k in (1, 2, 6):
    ms, _ = bench(D + "osnet_x0_25.onnx", n=10, shape=(k, 3, 256, 128))
    print(f"  {k} nguoi: {ms:6.1f} ms", flush=True)
print("=== MODEL LUA (2 luong, nhu dang chay) ===", flush=True)
for f in ("best_nano_111_320.onnx", "best_nano_111_640.onnx"):
    ms, sz = bench(F + f, threads=2)
    print(f"  {f:24s} imgsz={sz:3d}  {ms:6.1f} ms", flush=True)
