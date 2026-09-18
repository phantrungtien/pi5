import numpy as np, onnxruntime as ort, time

def bench(path, n=15, threads=4):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    i = s.get_inputs()[0]
    x = np.random.rand(*i.shape).astype(np.float32)
    s.run(None, {i.name: x}); s.run(None, {i.name: x})
    t = []
    for _ in range(n):
        a = time.perf_counter(); s.run(None, {i.name: x}); t.append(time.perf_counter() - a)
    return np.median(t) * 1000, i.shape[2]

for f in ["yolo11n-pose.onnx", "yolo11n-pose-416.onnx", "yolo11n-pose-320.onnx"]:
    ms, sz = bench("/app/t/" + f)
    print(f"{f:24s} imgsz={sz:3d}  {ms:6.1f} ms  -> tran {1000/ms:4.1f} fps", flush=True)

so = ort.SessionOptions(); so.intra_op_num_threads = 4
r = ort.InferenceSession("/app/t/osnet_x0_25.onnx", so, providers=["CPUExecutionProvider"])
nm = r.get_inputs()[0].name
for k in (1, 2, 6):
    x = np.random.rand(k, 3, 256, 128).astype(np.float32)
    r.run(None, {nm: x})
    t = []
    for _ in range(10):
        a = time.perf_counter(); r.run(None, {nm: x}); t.append(time.perf_counter() - a)
    print(f"osnet ReID {k} nguoi{'':11s}{np.median(t)*1000:6.1f} ms", flush=True)
