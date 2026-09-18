import numpy as np, onnxruntime as ort, time
D = "/app/tracking_pi5/"
def bench(path, n=20, threads=3):
    so = ort.SessionOptions(); so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    i = s.get_inputs()[0]
    sh = [d if isinstance(d, int) else 1 for d in i.shape]
    x = np.random.rand(*sh).astype(np.float32)
    s.run(None, {i.name: x}); s.run(None, {i.name: x})
    t = []
    for _ in range(n):
        a = time.perf_counter(); s.run(None, {i.name: x}); t.append(time.perf_counter() - a)
    return float(np.median(t) * 1000)
for f in ("yolo11n-pose-320.onnx", "yolo11n-pose-320-int8.onnx"):
    for th in (3, 4):
        print(f"  {f:30s} {th} luong: {bench(D+f, threads=th):6.1f} ms", flush=True)
for f in ("best_nano_111_320-int8.onnx",):
    print(f"  {f:30s} 1 luong: {bench(D+f, threads=1):6.1f} ms", flush=True)
