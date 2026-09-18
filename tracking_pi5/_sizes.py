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
    return float(np.median(t) * 1000), sh[2]
for f in ("yolo11n-pose-320.onnx", "yolo11n-pose-288.onnx",
          "yolo11n-pose-256.onnx", "yolo11n-pose-224.onnx"):
    ms, sz = bench(D + f)
    print(f"  imgsz {sz:3d}: {ms:6.1f} ms  -> tran {1000/ms:5.1f} fps", flush=True)
