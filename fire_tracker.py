#!/usr/bin/env python3
"""Bam vet lua/khoi bang YOLO11n ONNX, ve thang len khung hinh MJPEG.

    python3 fire_tracker.py --model ~/firedrop_sim/tracking/best_nano_111_640.onnx
    (tu kiem, khong can camera - xem demo() cuoi file)

Dung that thi khong goi truc tiep, ma qua may chu video:

    python3 mjpeg_server.py --ros-topic /rgb/image \
        --detect ~/firedrop_sim/tracking/best_nano_111_640.onnx

--------------------------------------------------------------------------
VI SAO VE VAO KHUNG HINH CHU KHONG GUI TOA DO QUA WEBSOCKET 8765
--------------------------------------------------------------------------
Duong lenh 8765 co san `envelope` (core/adapters/remote.py) nen gui hop bao
qua do la lam duoc. Nhung hop va khung hinh khi do di HAI duong khac nhau,
moi duong mot do tre: khung anh qua HTTP 8080, hop qua WebSocket 8765. Drone
bay 5 m/s o cao 30 m, lech 100 ms la hop tut sau ngon lua ~10 px va no LUON
tut ve mot phia - nhin ra ngay va nhin nhu thuat toan bam sai, trong khi thuat
toan dung. Ve vao chinh khung hinh vua suy dien thi hop va anh KHONG THE lech
nhau, bat ke mang tre bao nhieu.

Doi lai: GUI khong doc duoc so lieu hop (khong loc, khong bam nut). Neu sau nay
can thi them topic "vision/tracks" vao WS, NHUNG van giu hop ve san o day -
mot cai de nhin, mot cai de may doc.

--------------------------------------------------------------------------
BAM VET, KHONG PHAI CHI PHAT HIEN
--------------------------------------------------------------------------
YOLO chay doc lap tung khung: cung mot dam chay se nhay so hieu, va cu vai
khung lai mat mot khung roi hien lai - hop nhap nhay. `Tracker` ghep hop giua
cac khung bang IoU, giu so hieu, va:

  - `confirm_hits` khung lien tiep moi ve  -> nhieu mot khung khong len man hinh
  - con song them `max_miss` khung sau khi mat -> lua bi khoi che thoang qua
    khong lam mat hop

Day la bam vet DUOI DAT thi kem (khong co du doan chuyen dong), nhung o day
camera nadir va lua thi DUNG YEN duoi dat - thu chuyen dong la camera, va no
cham hon nhieu so voi nhip khung. Ghep bang IoU la du. Can hon thi Kalman,
dung viet truoc khi do duoc la thieu.

--------------------------------------------------------------------------
⚠ DO TREN LUA CUA GAZEBO: MODEL NAY KHONG AN LUA MO PHONG
--------------------------------------------------------------------------
best_nano_111_640.onnx hoc tu anh CHAY THAT (Fire-Smoke-Detection-Yolov11).
Lua trong Gazebo la mot khoi phat sang mo, khong co van ngon lua. Do tren 717
khung cua tron mot chuyen bay o firedrop_city (bay thang, tha, ASSESS, RTL):

    dam chay THAT, luc gan nhat      diem Fire dinh 0.25   (4/717 khung >= 0.20)
    nen be tong xam sau khi dau      diem Fire dinh 0.607  <- DUONG TINH GIA
    moi khung con lai                diem Fire <= 0.03

Tuc la o nguong 0.35 mac dinh, thu DUY NHAT ca chuyen bay hien ra la mot dam
chay khong ton tai. Vi vay:

  1. `max_box_frac` vut hop phu qua 25% khung hinh - dung cai duong tinh gia
     tren, no phu KIN man hinh.
  2. `run_sim.sh` cham nguong xuong 0.20 CHI cho mo phong (TRACK_CONF). Mac
     dinh 0.35 o day giu nguyen cho camera that, la thu model duoc hoc.

Va nhac lai cho ro: hop nay chi de NGUOI XEM. Quyet dinh tha bong van do
`fire_detector` (nguong HSV do duoc) - dung model nay de nham muc tieu la
lay mot bo phat hien 4/717 khung thay muc tieu, thay cho mot bo dang chay dung.
Muon dung that thi phai hoc them tren chinh khung hinh Gazebo.
"""

import argparse
import time

import cv2
import numpy as np

# Ten lop lay tu metadata cua chinh file .onnx (xem `names` trong metadata_props).
# KHONG viet cung o day: xuat lai model voi bo lop khac ma quen sua thi hop cu
# nhan nham ten - hong cam, kieu te nhat.
CLASS_FALLBACK = {0: "Fire", 1: "Smoke"}
COLORS = {"Fire": (40, 90, 230), "Smoke": (190, 170, 120)}  # BGR
COLOR_OTHER = (200, 200, 200)


def letterbox(img, size=640, pad=114):
    """Resize giu ti le roi dem vien - dung phep tien xu ly cua Ultralytics.

    Tra ve (anh_640, ti_le, dx, dy) de doi nguoc toa do hop ve anh goc. Resize
    thang (khong giu ti le) cung chay va cung ra hop, chi la hop lech he thong
    theo phuong bi keo gian: anh 640x480 keo thanh 640x640 la moi vat cao them
    33%, model thay mot ngon lua deo hon moi ngon lua no tung hoc.
    """
    h, w = img.shape[:2]
    r = min(size / w, size / h)
    nw, nh = round(w * r), round(h * r)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    out = np.full((size, size, 3), pad, dtype=np.uint8)
    out[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, r, dx, dy


def _ghim(bx, by, bx2, by2, w, h, max_box_frac):
    """Cat hop ve trong khung va vut hop qua to. None = bo hop nay.

    Cat ve trong khung: hop trum ra ngoai mep anh la binh thuong (vat bi cat boi
    mep), nhung ve ra ngoai thi OpenCV im lang bo net.

    VUT HOP QUA TO. Cung luat `max_blob_frac` ma fire_detector ben firedrop_sim
    dung, va vi cung mot ly do da do duoc: diem "Fire" CAO NHAT cua ca mot chuyen
    bay la 0.607 tren mot khung phu kin man hinh nen be tong xam, luc drone da dau
    va camera nhin canh quat cua chinh no. Dam chay THAT trong cung chuyen do chi
    duoc 0.25. Ha nguong de thay dam chay ma khong co luat nay thi thu duy nhat
    GUI hien ra la cai dam chay khong ton tai, voi diem cao gap doi dam chay that.
    """
    x1, y1 = max(0, int(bx)), max(0, int(by))
    x2, y2 = min(w - 1, int(bx2)), min(h - 1, int(by2))
    if x2 <= x1 or y2 <= y1:
        return None
    if (x2 - x1) * (y2 - y1) > max_box_frac * w * h:
        return None
    return x1, y1, x2, y2


def decode(raw, r, dx, dy, w, h, conf_min, iou_nms, max_box_frac=0.25, keep_cls=None):
    """Dau ra YOLO -> [(x1, y1, x2, y2, lop, diem)] tren toa do anh goc.

    Nhan HAI dang, phan biet bang shape chu khong bang co dat co:

      [1, 4+nc, 8400]  xuat nms=False — NMS phai lam o day
      [1, N, 6]        xuat nms=True  — x1,y1,x2,y2,diem,lop, da loc trung roi

    8400 hop la 80x80 + 40x40 + 20x20 o luoi ba tang; gan het la rac diem ~0,
    loc nguong TRUOC roi moi NMS thi nhanh hon hai bac.
    """
    # Model da co NMS ben trong. KHONG chay NMS lan hai: no khong lam sach hon,
    # chi ton thoi gian va an mat may hop nam sat nhau ma lan dau da co y giu.
    if raw.ndim == 3 and raw.shape[2] == 6:
        out = []
        for x1, y1, x2, y2, score, cls in raw[0]:
            if score < conf_min:
                continue  # [1,300,6] dem du 300 cho, cho thua lap day bang diem 0
            if keep_cls is not None and int(cls) not in keep_cls:
                continue
            # Toa do o he 640 DA letterbox — phai go dem roi chia ti le, giong
            # het nhanh duoi. Quen buoc nay thi hop van hien, chi la lech deu.
            g = _ghim((x1 - dx) / r, (y1 - dy) / r, (x2 - dx) / r, (y2 - dy) / r,
                      w, h, max_box_frac)
            if g:
                out.append((*g, int(cls), float(score)))
        return out

    p = raw[0].T                      # [8400, 4+nc]
    scores = p[:, 4:]
    best = scores.max(axis=1)
    keep = best >= conf_min
    if not keep.any():
        return []
    p, best = p[keep], best[keep]
    cls = scores[keep].argmax(axis=1)

    # Loc lop TRUOC NMS: NMS o duoi KHONG phan biet lop, nen mot hop Smoke diem
    # cao nam de len dam lua se nuot hop Fire roi den luot no bi vut -> mat dung
    # cai dang can tim, va khong mot dong nao bao. Loc theo lop DA argmax chu
    # khong xoa cot Smoke truoc khi lay max: model bao "day la khoi" thi bo han,
    # khong di tim diem Fire thap hon trong cung o de goi no la lua.
    if keep_cls is not None:
        m = np.isin(cls, list(keep_cls))
        if not m.any():
            return []
        p, best, cls = p[m], best[m], cls[m]

    # cx,cy,w,h (he 640, da letterbox) -> x,y,w,h (he anh goc)
    cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
    x = (cx - bw / 2 - dx) / r
    y = (cy - bh / 2 - dy) / r
    bw, bh = bw / r, bh / r

    boxes = np.stack([x, y, bw, bh], axis=1)
    idx = cv2.dnn.NMSBoxes(boxes.tolist(), best.tolist(), conf_min, iou_nms)
    out = []
    for i in np.array(idx).flatten():
        bx, by, bwi, bhi = boxes[i]
        g = _ghim(bx, by, bx + bwi, by + bhi, w, h, max_box_frac)
        if g:
            out.append((*g, int(cls[i]), float(best[i])))
    return out


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    return inter / ((ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter)


class Track:
    __slots__ = ("id", "box", "cls", "score", "hits", "miss")

    def __init__(self, tid, box, cls, score):
        self.id, self.box, self.cls, self.score = tid, box, cls, score
        self.hits, self.miss = 1, 0


class Tracker:
    """Ghep hop giua cac khung bang IoU. Xem docstring dau file ve pham vi."""

    def __init__(self, iou_min=0.2, confirm_hits=2, max_miss=5):
        self.iou_min, self.confirm_hits, self.max_miss = iou_min, confirm_hits, max_miss
        self.tracks: list[Track] = []
        self._next = 1

    def update(self, dets):
        """dets = [(x1,y1,x2,y2,cls,score)] -> danh sach Track DA XAC NHAN."""
        chua_ghep = list(self.tracks)
        # Ghep tham lam theo diem giam dan: hop chac an duoc chon truoc, nen mot
        # hop rac diem thap khong the cuop mat so hieu cua vet dang bam tot.
        for x1, y1, x2, y2, cls, score in sorted(dets, key=lambda d: -d[5]):
            box = (x1, y1, x2, y2)
            hop = [t for t in chua_ghep if t.cls == cls]
            t = max(hop, key=lambda t: iou(t.box, box), default=None)
            if t is not None and iou(t.box, box) >= self.iou_min:
                t.box, t.score, t.miss = box, score, 0
                t.hits += 1
                chua_ghep.remove(t)
            else:
                self.tracks.append(Track(self._next, box, cls, score))
                self._next += 1

        for t in chua_ghep:
            t.miss += 1
        self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]
        return [t for t in self.tracks if t.hits >= self.confirm_hits and t.miss == 0]


class FireTracker:
    """Nap model mot lan, moi khung goi `draw(img)`. An toan de goi tu mot thread.

    Suy dien CHAM (YOLO11n 640 tren CPU ~40-60 ms). Goi tu dung thread dang doc
    camera la co y: no lam cham vong doc, ma vong doc thi chi giu KHUNG MOI NHAT
    (QoS depth 1 ben ROS, CAP_PROP_BUFFERSIZE 1 ben USB) - nen anh huong duy nhat
    la fps giam, khong bao gio la do tre troi. Dua sang thread rieng se cho fps
    cao hon nhung hop ve len khung KHAC voi khung da suy dien, tuc lech - dung
    cai thu ma ca file nay sinh ra de tranh.
    """

    def __init__(self, model_path, conf=0.35, iou_nms=0.45, max_box_frac=0.25,
                 providers=None, threads=0, confirm_hits=2, max_miss=5,
                 classes=("Fire",)):
        self.conf, self.iou_nms, self.max_box_frac = conf, iou_nms, max_box_frac

        # HAI backend, MOT ham infer(): .pt di qua ultralytics/torch, .onnx chay
        # thang onnxruntime. Ca hai tra ve cung dinh dang dets nen Tracker, HUD
        # va mjpeg_server KHONG doi mot dong - doi backend la doi mot nhanh if.
        self.pt = str(model_path).endswith(".pt")
        if self.pt:
            import os

            # ultralytics thu pip install moi lan nap neu thieu goi phu; tren Pi
            # khong mang thi treo ~60s roi van chay. Phai dat TRUOC khi import.
            os.environ.setdefault("YOLO_AUTOINSTALL", "false")
            import torch
            from ultralytics import YOLO

            if threads:
                torch.set_num_threads(threads)
            self.model = YOLO(str(model_path))
            self.size = 640
            self.names = dict(self.model.names)
            # Lan suy dien dau cham gap 10-30 lan (cap phat buffer). Dot o day,
            # khong thi no roi vao khung dau va HUD bao vai nghin ms mot lan.
            self.model(np.zeros((self.size, self.size, 3), np.uint8), verbose=False)
        else:
            import onnxruntime as ort

            # `threads`: do tren Pi5/yolov8n-640 — 1 luong 542ms, 2 luong 347ms,
            # 4 luong 426ms. Bon luong CHAM hon hai vi con camera va nen JPEG cung
            # dang an core. De 0 la giao cho onnxruntime tu chon (may khac may khac).
            opt = ort.SessionOptions()
            if threads:
                opt.intra_op_num_threads = threads
            self.sess = ort.InferenceSession(
                str(model_path), opt, providers=providers or [
                    p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                    if p in ort.get_available_providers()]
            )
            self.inp = self.sess.get_inputs()[0].name
            self.size = self.sess.get_inputs()[0].shape[2] or 640
            self.names = self._names()
        # confirm_hits/max_miss dem theo KHUNG SUY DIEN, khong theo giay. Chay
        # moi khung o 15 fps thi max_miss=5 la 1/3 giay; chay o 2,4 fps (yolov8n
        # tren CPU Pi) thi cung con so do thanh HAI GIAY hop ma o do khong con
        # gi — dung cai loi "ve thu khong ton tai" ma file nay chong. Nen ai goi
        # o nhip cham phai ha xuong, xem mjpeg_server.py.
        # ponytail: mac dinh CHI giu Fire. Drone di dap lua, con Smoke o conf 0.2
        # an false-positive nang (tuong sang, nang chieu, bui). Muon lai ca hai:
        # FireTracker(..., classes=None), hoac classes=("Fire", "Smoke").
        if classes is None:
            self.keep_cls = None
        else:
            self.keep_cls = {i for i, n in self.names.items() if n in classes}
            if not self.keep_cls:
                # Im lang o day = model chay binh thuong ma KHONG BAO GIO ra hop.
                raise ValueError(
                    f"khong co lop nao ten {tuple(classes)} trong {self.names}")

        # Nhan dau dong HUD: dem nguoi (--classes person) thi ghi NGUOI, con lai LUA.
        self.hud_label = "NGUOI" if classes and set(classes) == {"person"} else "LUA"
        self.tracker = Tracker(confirm_hits=confirm_hits, max_miss=max_miss)
        self.tracks = []  # vet da xac nhan cua lan suy dien GAN NHAT
        self.ms = 0.0  # thoi gian suy dien khung cuoi, de hien len HUD

    def _names(self):
        meta = self.sess.get_modelmeta().custom_metadata_map or {}
        try:
            return {int(k): v for k, v in eval(meta["names"]).items()}  # noqa: S307
        except Exception:
            # Model khong kem metadata (hay kem kieu la) van chay duoc, chi la
            # hop mang ten mac dinh. Bo qua trong im lang thi sai o day se doi
            # lot thanh "model phat hien nham lop" - nen phai keu.
            print("[track] khong doc duoc `names` trong model - dung ten mac dinh",
                  flush=True)
            return dict(CLASS_FALLBACK)

    def infer(self, img):
        """BGR uint8 -> danh sach Track da xac nhan cua khung nay."""
        t0 = time.monotonic()
        h, w = img.shape[:2]
        if self.pt:
            # ultralytics tu letterbox va tu NMS, va tra hop DA ve he anh goc -
            # nen khong goi decode() o day. Van qua _ghim de ghim trong khung va
            # bo hop qua to, giong het nhanh ONNX: hai backend phai loc nhu nhau
            # thi so lieu so sanh moi co nghia.
            res = self.model(img, conf=self.conf, iou=self.iou_nms, verbose=False)[0]
            dets = []
            for (x1, y1, x2, y2), c, sc in zip(res.boxes.xyxy.tolist(),
                                               res.boxes.cls.tolist(),
                                               res.boxes.conf.tolist()):
                if self.keep_cls is not None and int(c) not in self.keep_cls:
                    continue
                g = _ghim(x1, y1, x2, y2, w, h, self.max_box_frac)
                if g:
                    dets.append((*g, int(c), float(sc)))
        else:
            lb, r, dx, dy = letterbox(img, self.size)
            blob = np.ascontiguousarray(
                lb[:, :, ::-1].transpose(2, 0, 1)[None].astype(np.float32) / 255.0
            )
            raw = self.sess.run(None, {self.inp: blob})[0]
            dets = decode(raw, r, dx, dy, w, h, self.conf, self.iou_nms,
                          self.max_box_frac, self.keep_cls)
        self.ms = (time.monotonic() - t0) * 1000.0
        self.tracks = self.tracker.update(dets)
        return self.tracks

    def draw(self, img):
        """Suy dien roi ve TAI CHO len `img`. Tra ve so vet dang bam."""
        self.infer(img)
        return self.draw_cached(img)

    def draw_cached(self, img, hud=None):
        """Ve cac vet CUA LAN SUY DIEN GAN NHAT, khong chay model.

        Tach ra vi tren Pi mot lan suy dien mat ~410 ms con khung hinh ve moi
        ~67 ms: goi `draw()` tung khung la keo ca luong video xuong 2,4 fps.
        Luong YOLO chay rieng, con day chi ve lai cai no vua tim duoc.
        """
        for t in self.tracks:
            name = self.names.get(t.cls, f"lop{t.cls}")
            color = COLORS.get(name, COLOR_OTHER)
            x1, y1, x2, y2 = t.box
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            _nhan(img, f"#{t.id} {name} {t.score:.2f}", x1, y1, color)
        if hud is not None:
            # run_gui dua ca dong ("NGUOI 3 | 30 FPS"): ms/FPS cua model LUA o day
            # bi doc nham la fps video. Toc do lua van in trong log ("yolo NN ms").
            _hud(img, hud)
            return len(self.tracks)
        # FPS suy dien = 1000/ms, de so sanh cac backend/kich thuoc model tren HUD
        fps = 1000.0 / self.ms if self.ms > 0 else 0.0
        _hud(img, f"{self.hud_label} {len(self.tracks)} | {self.ms:.0f} ms | {fps:.1f} FPS")
        return len(self.tracks)


def _co_chu(img):
    """(co chu, do day net) theo chieu cao anh.

    Net 1 px bi JPEG q70 (va chroma subsampling cua no) xoa nhoe thanh vet mo.
    Net 2 px tro len moi song sot qua nen. Co theo chieu cao de 480p va 1080p
    nhin cung mot kich thuoc tren GUI.
    """
    k = img.shape[0] / 480
    return 0.6 * k, max(2, round(2 * k))  # ponytail: hang so chinh tay, to/nho thi sua 0.6


def _nhan(img, text, x, y, color):
    """Nhan tren canh tren cua hop, lat xuong duoi khi hop cham mep tren anh."""
    fs, t = _co_chu(img)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fs, t)
    ty = y - th - 6 if y - th - 6 >= 0 else y + 2
    cv2.rectangle(img, (x, ty), (x + tw + 6, ty + th + 6), color, -1)
    cv2.putText(img, text, (x + 3, ty + th + 2), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (20, 20, 20), t, cv2.LINE_AA)


def _hud(img, text):
    """Mot dong o goc duoi-trai. KHONG co hop nao cung phai hien, no la bang
    chung "model dang chay ma khong thay gi" - khac han voi "model chet"."""
    h = img.shape[0]
    fs, t = _co_chu(img)
    cv2.putText(img, text, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (0, 0, 0), t + 3, cv2.LINE_AA)
    cv2.putText(img, text, (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, fs,
                (255, 255, 255), t, cv2.LINE_AA)


def demo(model=None):
    """Tu kiem. Phan hinh hoc chay khong can model; co model thi nap va do luon."""
    # letterbox: giu ti le, va doi nguoc toa do phai ve dung cho cu
    img = np.zeros((480, 640, 3), np.uint8)
    lb, r, dx, dy = letterbox(img, 640)
    assert lb.shape == (640, 640, 3) and (dx, dy) == (0, 80), (lb.shape, dx, dy)
    assert abs(r - 1.0) < 1e-9, r

    # mot hop giua anh 640 phai giai ma ve dung giua anh 640x480 goc
    raw = np.zeros((1, 6, 1), np.float32)
    raw[0, :4, 0] = [320, 320, 64, 64]   # cx, cy, w, h trong he 640 da letterbox
    raw[0, 4, 0] = 0.9                   # lop 0 = Fire
    (x1, y1, x2, y2, cls, score), = decode(raw, r, dx, dy, 640, 480, 0.35, 0.45)
    assert (x1, y1, x2, y2) == (288, 208, 352, 272), (x1, y1, x2, y2)
    assert cls == 0 and abs(score - 0.9) < 1e-6

    # diem duoi nguong thi khong ra hop nao
    raw[0, 4, 0] = 0.10
    assert decode(raw, r, dx, dy, 640, 480, 0.35, 0.45) == []

    # hop phu qua max_box_frac bi vut, du diem RAT cao - day la duong tinh gia
    # 0.607 tren nen be tong do duoc that (xem docstring dau file)
    to = np.zeros((1, 6, 1), np.float32)
    to[0, :4, 0] = [320, 320, 640, 640]   # phu kin ca khung
    to[0, 4, 0] = 0.99
    assert decode(to, r, dx, dy, 640, 480, 0.35, 0.45) == []
    # cung hop do nhung tat luat (frac 1.0) thi phai ra - de chac la bi vut
    # VI QUA TO, khong phai vi mot loi hinh hoc nao khac
    assert len(decode(to, r, dx, dy, 640, 480, 0.35, 0.45, max_box_frac=1.0)) == 1

    # DANG THU HAI: model xuat kem NMS -> [1, N, 6] = x1,y1,x2,y2,diem,lop.
    # Cung mot hop, cung ket qua nhu nhanh tren — neu hai nhanh ra khac nhau thi
    # doi model se lam hop lech ma khong ai doi nghi vao decode().
    nms = np.zeros((1, 3, 6), np.float32)
    nms[0, 0] = [288, 288, 352, 352, 0.9, 0]  # he 640 da letterbox
    nms[0, 1] = [288, 288, 352, 352, 0.0, 0]  # cho thua: [1,300,6] luon dem du 300
    nms[0, 2] = [0, 0, 640, 640, 0.99, 1]     # phu kin khung -> phai bi vut vi qua to
    (x1, y1, x2, y2, cls, score), = decode(nms, r, dx, dy, 640, 480, 0.35, 0.45)
    assert (x1, y1, x2, y2) == (288, 208, 352, 272), (x1, y1, x2, y2)
    assert cls == 0 and abs(score - 0.9) < 1e-6

    # draw_cached: chay moi khung hinh nen mot loi ten bien o day lam CHET
    # luong doc camera va video dung han — da xay ra that. Goi duoc no ma khong
    # can .onnx bang cach muon ham cho mot vat gia du thuoc tinh.
    class _Gia:
        names, ms = {0: "Fire"}, 12.3
        tracks = [Track(1, (10, 10, 50, 50), 0, 0.9)]
    _Gia.draw_cached = FireTracker.draw_cached
    anh = np.zeros((120, 160, 3), np.uint8)
    assert _Gia().draw_cached(anh) == 1
    assert anh.any(), "phai ve duoc hop len anh, khong duoc de anh den"

    assert abs(iou((0, 0, 10, 10), (0, 0, 10, 10)) - 1.0) < 1e-9
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - (50 / 150)) < 1e-9

    # Tracker: phai giu so hieu qua cac khung, va chi ve sau `confirm_hits`
    tr = Tracker(confirm_hits=2, max_miss=2)
    d = [(10, 10, 50, 50, 0, 0.9)]
    assert tr.update(d) == [], "khung dau tien chua duoc xac nhan"
    (t,) = tr.update([(12, 12, 52, 52, 0, 0.9)])
    assert t.id == 1
    (t2,) = tr.update([(14, 14, 54, 54, 0, 0.9)])
    assert t2.id == 1, "cung mot dam chay phai giu nguyen so hieu"

    # mat dau -> khong ve nua, nhung con song du `max_miss` khung roi moi xoa
    assert tr.update([]) == [] and len(tr.tracks) == 1
    assert tr.update([]) == [] and len(tr.tracks) == 1
    assert tr.update([]) == [] and tr.tracks == []

    # lop khac thi KHONG ghep, du hop trung khit
    tr2 = Tracker(confirm_hits=1)
    (a,) = tr2.update([(0, 0, 40, 40, 0, 0.9)])
    ids = {t.id for t in tr2.update([(0, 0, 40, 40, 1, 0.9)])}
    assert a.id not in ids, "Fire va Smoke chong nhau khong duoc dung chung vet"

    # keep_cls: hop Smoke diem cao nam de len hop Fire. Khong loc thi NMS (khong
    # phan biet lop) giu Smoke va vut Fire -> ra 0 dam lua.
    chong = np.zeros((1, 6, 2), np.float32)
    chong[0, :4, 0] = [320, 320, 64, 64]      # lop 0 = Fire
    chong[0, 4, 0] = 0.6
    chong[0, :4, 1] = [322, 322, 66, 66]      # gan trung -> IoU cao
    chong[0, 5, 1] = 0.95                     # lop 1 = Smoke, diem cao hon
    (bo,) = decode(chong, r, dx, dy, 640, 480, 0.35, 0.45)
    assert bo[4] == 1, "khong loc thi NMS phai giu Smoke"
    (giu,) = decode(chong, r, dx, dy, 640, 480, 0.35, 0.45, keep_cls={0})
    assert giu[4] == 0 and abs(giu[5] - 0.6) < 1e-6, "loc roi phai con dung hop Fire"
    assert decode(nms, r, dx, dy, 640, 480, 0.35, 0.45, keep_cls={1}) == [], \
        "dang [1,N,6] cung phai loc theo lop"

    print("fire_tracker (hinh hoc + bam vet): TAT CA DAT")

    if model:
        ft = FireTracker(model)
        anh = np.random.randint(0, 255, (480, 640, 3), np.uint8)
        t0 = time.monotonic()
        n = ft.draw(anh)
        print(f"nap {model}")
        print(f"lop: {ft.names}")
        print(f"mot khung 640x480: {(time.monotonic()-t0)*1000:.0f} ms, {n} vet")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", default=None, help="duong dan .onnx de do luon toc do")
    demo(p.parse_args().model)
