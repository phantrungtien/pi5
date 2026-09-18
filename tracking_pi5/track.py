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
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nh, nw = round(h * r), round(w * r)
    pad_t, pad_l = (size - nh) // 2, (size - nw) // 2
    out = np.full((size, size, 3), 114, np.uint8)
    out[pad_t:pad_t + nh, pad_l:pad_l + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, pad_l, pad_t


class Detector:
    """ONNX export has NMS baked in: output0 is [1,300,6] = x1,y1,x2,y2,conf,cls in 640-space."""

    def __init__(self, path, threads=4):
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads  # Pi 5 has 4 cores
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.iname = inp.name
        # Export dung dynamic=False nen shape la SO THAT, khong phai chuoi. Doc tu day
        # de doi model 640/416/320 chi la doi duong dan file, khong phai sua code.
        self.imgsz = int(inp.shape[2])

    def __call__(self, frame, conf=0.35, classes=None):
        lb, r, pl, pt = letterbox(frame, self.imgsz)
        x = np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32) / 255.0
        out = self.sess.run(None, {self.iname: x})[0][0]
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

    def set(self, det):
        """det = one detector row: box, conf, cls (+ 51 keypoint values on a pose model)."""
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

    def __init__(self, iou_thr=0.3, max_age=15, min_hits=3):
        self.iou_thr, self.max_age, self.min_hits = iou_thr, max_age, min_hits
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
            t.set(dets[di])
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
        return [t for t in self.tracks if t.hits >= self.min_hits and t.age == 0]


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
