#!/usr/bin/env python3
"""Person tracking on Raspberry Pi 5: yolov8n-drone.onnx + IoU tracker + OpenCV debug GUI.

    python3 track.py --source 0            # webcam
    python3 track.py --source vid.mp4      # file
    python3 track.py --selftest            # no camera needed

Keys: q quit | space pause | s save frame | d toggle debug overlay | t toggle trails
"""
import argparse, collections, time
import cv2
import numpy as np

IMGSZ = 640  # chi la mac dinh cho letterbox(); Detector doc kich thuoc THAT tu model


def letterbox(img, size=IMGSZ):
    """size: int (vuong) hoac (H, W) — model export chu nhat 16:9 khong phi diem anh vao vien."""
    sh, sw = (size, size) if isinstance(size, int) else size
    h, w = img.shape[:2]
    r = min(sh / h, sw / w)
    nh, nw = round(h * r), round(w * r)
    pad_t, pad_l = (sh - nh) // 2, (sw - nw) // 2
    out = np.full((sh, sw, 3), 114, np.uint8)
    out[pad_t:pad_t + nh, pad_l:pad_l + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, pad_l, pad_t


class Detector:
    """ONNX export has NMS baked in: output0 is [1,300,6] = x1,y1,x2,y2,conf,cls in 640-space."""

    def __init__(self, path, threads=4):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads  # Pi 5 has 4 cores
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # Jetson Orin: CUDA 11 ms vs CPU 31 ms (320). Pi chi co CPU -> ort tu bo CUDA.
        avail = ort.get_available_providers()
        self.sess = ort.InferenceSession(path, so, providers=[
            p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in avail])
        # Hai kieu export: CO NMS -> [1,300,6+51] da loc; KHONG NMS ("raw") -> [1,4+nc+51,N]
        # va NMS lam o day bang cv2. Raw de chay TensorRT: NMS trong graph lam TRT loi
        # ("GetSupportedList"), con raw thi TRT FP16 nhanh hon CUDA nhieu lan.
        shp = self.sess.get_outputs()[0].shape
        self.raw = shp[1] < shp[2]
        kpt = self.sess.get_modelmeta().custom_metadata_map.get("kpt_shape")
        self.nk = int(np.prod(eval(kpt))) if kpt else 0  # noqa: S307 - metadata cua chinh model
        self.nc = shp[1] - 4 - self.nk
        if self.raw and "TensorrtExecutionProvider" in avail:
            import os
            cache = os.path.join(os.path.dirname(os.path.abspath(path)), "trt_cache")
            try:
                # Lan dau build engine vai phut (stream chua len trong luc do), sau doc cache.
                self.sess = ort.InferenceSession(path, so, providers=[
                    ("TensorrtExecutionProvider", {
                        "trt_fp16_enable": True, "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": cache, "trt_timing_cache_enable": True,
                        "trt_timing_cache_path": cache}),
                    "CUDAExecutionProvider", "CPUExecutionProvider"])
            except Exception as e:  # noqa: BLE001 - TRT hong thi van chay CUDA, cham hon
                print(f"[det] TensorRT loi ({type(e).__name__}) -> CUDA", flush=True)
        print(f"[det] {path} {'raw+NMS cv2' if self.raw else 'NMS trong model'} "
              f"-> {self.sess.get_providers()[0]}", flush=True)
        inp = self.sess.get_inputs()[0]
        self.iname = inp.name
        # Model export half=True (FP16) nhan va tra float16; con lai float32.
        self.dtype = np.float16 if "float16" in inp.type else np.float32
        # Model qua them_tien_xu_ly.py: nhan thang uint8 BGR [1,H,W,3], GPU tu doi.
        self.u8 = "uint8" in inp.type
        # Export dung dynamic=False nen shape la SO THAT, khong phai chuoi. Doc tu day
        # de doi model 640/416/320 chi la doi duong dan file, khong phai sua code.
        self.imgsz = ((int(inp.shape[1]), int(inp.shape[2])) if self.u8
                      else (int(inp.shape[2]), int(inp.shape[3])))

    def __call__(self, frame, conf=0.35, classes=None):
        lb, r, pl, pt = letterbox(frame, self.imgsz)
        if self.u8:
            x = lb[None]
        else:
            x = (np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)[None])
                 .astype(np.float32) / 255.0).astype(self.dtype)
        out = self.sess.run(None, {self.iname: x})[0][0].astype(np.float32)
        if self.raw:
            out = self._nms(out, conf, classes)
        out = out[out[:, 4] >= conf]
        if classes is not None and len(out):
            out = out[np.isin(out[:, 5].astype(int), classes)]
        if not len(out):
            return out  # already (0, ncols) — reshape(0, -1) can't infer -1 from size 0
        out = out.copy()
        out[:, [0, 2]] = (out[:, [0, 2]] - pl) / r
        out[:, [1, 3]] = (out[:, [1, 3]] - pt) / r
        h, w = frame.shape[:2]
        np.clip(out[:, [0, 2]], 0, w - 1, out[:, [0, 2]])
        np.clip(out[:, [1, 3]], 0, h - 1, out[:, [1, 3]])
        if out.shape[1] > 6:  # pose export: 17 keypoints (x,y,conf) live in cols 6..57
            k = out[:, 6:].reshape(len(out), -1, 3)
            k[:, :, 0] = (k[:, :, 0] - pl) / r
            k[:, :, 1] = (k[:, :, 1] - pt) / r
        return out

    def _nms(self, out, conf, classes, iou=0.45, top=300):
        """[4+nc+nk, N] tho -> [M, 6+nk] = x1,y1,x2,y2,conf,cls,kpts giong export co NMS.
        iou 0.45 chu khong 0.7 (mac dinh ultralytics): o 0.7 hop GOP hai nguoi dung sat
        nhau (IoU ~0.5 voi tung nguoi) song sot -> dem du. Do 90 khung that: 0.45 bo het
        hop gop, khong mat nguoi xa nao."""
        o = out.T
        sc = o[:, 4:4 + self.nc]
        cls = sc.argmax(1)
        s = sc[np.arange(len(o)), cls]
        m = s >= conf
        if classes is not None:
            m &= np.isin(cls, classes)
        o, s, cls = o[m], s[m], cls[m]
        if not len(o):
            return np.zeros((0, 6 + self.nk), np.float32)
        cx, cy, bw, bh = o[:, 0], o[:, 1], o[:, 2], o[:, 3]
        keep = np.array(cv2.dnn.NMSBoxes(
            np.stack([cx - bw / 2, cy - bh / 2, bw, bh], 1).tolist(), s.tolist(), conf, iou),
            int).reshape(-1)[:top]
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
        return np.concatenate([xyxy, s[:, None], cls[:, None], o[:, 4 + self.nc:]], 1)[keep]


def iou_matrix(a, b):
    """a:(N,4) b:(M,4) xyxy -> (N,M)"""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)), np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


class Track:
    def __init__(self, tid, det):
        self.id = tid
        self.set(det)
        self.age = 0          # frames since last matched detection
        self.hits = 1
        self.trail = collections.deque(maxlen=48)
        self.push()

    def set(self, det, smooth=0.0):
        """det = one detector row: box, conf, cls (+ 51 keypoint values on a pose model).

        smooth > 0: EMA voi khung truoc (trong so cua cu = smooth) cho hop VA khop.
        Khop cu duoi 0.3 la rac (model doan) - lay thang khop moi, khong tron vao.
        """
        if smooth and hasattr(self, "det") and len(det) == len(self.det):
            old, det = self.det, det.copy()
            det[:5] = smooth * old[:5] + (1 - smooth) * det[:5]
            if len(det) > 6:
                o, n = old[6:].reshape(-1, 3), det[6:].reshape(-1, 3)
                ok = o[:, 2] >= 0.3
                n[ok] = smooth * o[ok] + (1 - smooth) * n[ok]   # view -> ghi thang vao det
        self.det, self.box = det, det[:4]
        self.conf, self.cls = float(det[4]), int(det[5])

    @property
    def kpts(self):
        """(17,3) x,y,conf -- or None on a plain detect model."""
        return self.det[6:].reshape(-1, 3) if len(self.det) > 6 else None

    def push(self):
        x1, y1, x2, y2 = self.box
        self.trail.append((int((x1 + x2) / 2), int(y2)))  # feet point, stable under a drone view


class IoUTracker:
    """Greedy IoU association, no motion model.
    ponytail: no Kalman/ReID -- ids swap when people cross or the drone yaws fast.
    Upgrade path: swap this class for BoT-SORT/ByteTrack if id switches actually hurt.
    """

    def __init__(self, iou_thr=0.3, max_age=15, min_hits=3, smooth=0.0, hold=0):
        """smooth: EMA hop/khop (0 = tat, 0.5 = tre ~1 khung). hold: van tra vet da mat
        toi `hold` khung - hop khong chop tat khi model hut mot khung. Ca hai MAC DINH TAT:
        vong FOLLOW can toa do tuoi, khong tre."""
        self.iou_thr, self.max_age, self.min_hits = iou_thr, max_age, min_hits
        self.smooth, self.hold = smooth, hold
        self.tracks, self._next = [], 1

    def update(self, dets):
        M = iou_matrix(np.array([t.box for t in self.tracks], np.float32).reshape(-1, 4), dets[:, :4])
        used_d, used_t = set(), set()
        for ti, di in sorted(np.ndindex(M.shape), key=lambda p: -M[p]):
            if M[ti, di] < self.iou_thr:
                break
            if ti in used_t or di in used_d:
                continue
            t = self.tracks[ti]
            t.set(dets[di], self.smooth)
            t.age, t.hits = 0, t.hits + 1
            t.push()
            used_t.add(ti); used_d.add(di)

        for ti, t in enumerate(self.tracks):
            if ti not in used_t:
                t.age += 1
        for di in range(len(dets)):
            if di not in used_d:
                self.tracks.append(Track(self._next, dets[di]))
                self._next += 1
        self.tracks = [t for t in self.tracks if t.age <= self.max_age]
        return [t for t in self.tracks if t.hits >= self.min_hits and t.age <= self.hold]


def color(i):
    return tuple(int(c) for c in ((37 * i) % 255, (17 * i + 90) % 255, (29 * i + 160) % 255))


def draw(frame, tracks, dets, hud, show_debug, show_trails):
    for t in tracks:
        x1, y1, x2, y2 = map(int, t.box)
        c = color(t.id)
        cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
        label = f"ID{t.id} {t.conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), c, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        if show_trails and len(t.trail) > 1:
            cv2.polylines(frame, [np.array(t.trail, np.int32)], False, c, 2)
    if show_debug:
        for d in dets:  # raw detections, before association
            cv2.rectangle(frame, (int(d[0]), int(d[1])), (int(d[2]), int(d[3])), (160, 160, 160), 1)
        for i, line in enumerate(hud):
            cv2.putText(frame, line, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(frame, line, (8, 20 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    return frame


def run(a):
    det = Detector(a.model, a.threads)
    tracker = IoUTracker(a.iou, a.max_age, a.min_hits)
    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src)
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
    if not cap.isOpened():
        raise SystemExit(f"cannot open source {a.source!r}")

    win = "tracking (q quit, space pause, s save, d debug, t trails)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.createTrackbar("conf x100", win, int(a.conf * 100), 99, lambda v: None)
    cv2.createTrackbar("iou x100", win, int(a.iou * 100), 99, lambda v: None)

    classes = None if a.classes == "" else [int(c) for c in a.classes.split(",")]
    fps, paused, debug, trails, frame, n = 0.0, False, True, True, None, 0
    while True:
        if not paused or frame is None:
            ok, frame = cap.read()
            if not ok:
                break
            n += 1
            t0 = time.perf_counter()
            conf = max(cv2.getTrackbarPos("conf x100", win), 1) / 100
            tracker.iou_thr = max(cv2.getTrackbarPos("iou x100", win), 1) / 100
            dets = det(frame, conf, classes)
            t_inf = (time.perf_counter() - t0) * 1000
            live = tracker.update(dets)
            fps = 0.9 * fps + 0.1 * (1000 / max(t_inf, 1e-3)) if fps else 1000 / max(t_inf, 1e-3)

        hud = [f"frame {n}  {t_inf:.0f} ms  {fps:.1f} fps",
               f"det {len(dets)}  live {len(live)}  pool {len(tracker.tracks)}",
               f"conf {conf:.2f}  iou {tracker.iou_thr:.2f}  max_age {tracker.max_age}",
               "PAUSED" if paused else ""]
        cv2.imshow(win, draw(frame.copy(), live, dets, hud, debug, trails))

        k = cv2.waitKey(1) & 0xFF
        if k == ord("q") or k == 27:
            break
        if k == ord(" "):
            paused = not paused
        if k == ord("d"):
            debug = not debug
        if k == ord("t"):
            trails = not trails
        if k == ord("s"):
            cv2.imwrite(f"frame_{n:06d}.jpg", draw(frame.copy(), live, dets, hud, debug, trails))
            print(f"saved frame_{n:06d}.jpg")
    cap.release()
    cv2.destroyAllWindows()


def selftest():
    t = IoUTracker(iou_thr=0.3, max_age=2, min_hits=2)
    box = lambda x: np.array([[x, 100, x + 50, 200, 0.9, 0]], np.float32)
    assert t.update(box(100)) == []                      # min_hits not reached yet
    live = t.update(box(105))
    assert len(live) == 1 and live[0].id == 1, live      # same object keeps id 1
    assert len(t.update(box(400))) == 0                  # jump -> new track, still young
    assert len(t.tracks) == 2                            # old one coasting on age
    for _ in range(3):
        t.update(np.zeros((0, 6), np.float32))
    assert t.tracks == [], t.tracks                      # all expired past max_age

    a = np.array([[0, 0, 10, 10]], np.float32)
    assert abs(iou_matrix(a, a)[0, 0] - 1.0) < 1e-6
    assert iou_matrix(a, np.array([[20, 20, 30, 30]], np.float32))[0, 0] == 0
    assert iou_matrix(a, np.zeros((0, 4), np.float32)).shape == (1, 0)

    img = np.zeros((480, 640, 3), np.uint8)
    lb, r, pl, pt = letterbox(img)
    assert lb.shape == (640, 640, 3) and abs(r - 1.0) < 1e-9 and (pl, pt) == (0, 80)
    print("selftest ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8n-drone.onnx")
    p.add_argument("--source", default="0")
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--iou", type=float, default=0.3)
    p.add_argument("--max-age", type=int, default=15)
    p.add_argument("--min-hits", type=int, default=3)
    p.add_argument("--classes", default="", help="comma list, e.g. 0 -- empty = all classes")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    selftest() if a.selftest else run(a)
