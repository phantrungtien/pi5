"""So phat hien giua cac co model tren CUNG mot bo khung that.

Chi nhanh hon ma sot nguoi thi khong dung duoc. Do: so nguoi bat duoc, do tin
cay, va dien tich hop nho nhat — hop nho = nguoi o xa, do la cho model nho gay.
"""
import glob, sys
import numpy as np, onnxruntime as ort, cv2

D = "/app/tracking_pi5/"
CONF = 0.35


def chay(model, anh):
    so = ort.SessionOptions(); so.intra_op_num_threads = 3
    s = ort.InferenceSession(D + model, so, providers=["CPUExecutionProvider"])
    i = s.get_inputs()[0]
    sz = i.shape[2]
    tong, confs, dien = 0, [], []
    for f in anh:
        im = cv2.imread(f)
        h, w = im.shape[:2]
        r = min(sz / w, sz / h)
        nw, nh = int(w * r), int(h * r)
        pad = np.full((sz, sz, 3), 114, np.uint8)
        pad[:nh, :nw] = cv2.resize(im, (nw, nh))
        x = pad[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = s.run(None, {i.name: x})[0][0]        # [300, 57]
        keep = out[out[:, 4] >= CONF]
        tong += len(keep)
        for d in keep:
            confs.append(float(d[4]))
            dien.append(float((d[2] - d[0]) * (d[3] - d[1])) / (sz * sz))
    return sz, tong, confs, dien


anh = sorted(glob.glob("/app/tracking_pi5/_khung/*.jpg"))
print(f"{len(anh)} khung, nguong conf {CONF}")
for m in sys.argv[1:]:
    sz, n, c, a = chay(m, anh)
    print(f"  imgsz {sz:3d}: {n:3d} nguoi | conf tb {np.mean(c) if c else 0:.3f} "
          f"| hop nho nhat {min(a)*100 if a else 0:.2f}% khung", flush=True)
