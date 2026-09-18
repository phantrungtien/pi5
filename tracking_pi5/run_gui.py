#!/usr/bin/env python3
"""person_follow_reid trong container drone-cam: /image_raw -> bám người -> MJPEG :8080.

Chạy CÙNG container với v4l2_camera_node, thay chỗ của `mjpeg_server.py --detect`:

    python3 run_gui.py --device /dev/video0        # MJPG 1280x720 thang tu webcam
    python3 run_gui.py --topic /image_raw           # doc topic ROS (can v4l2_camera_node)

Vì sao nằm trong container chứ không chạy trần trên Pi:
  * cv2 (cả trên host lẫn trong image) build KHÔNG có V4L2 — `VideoCapture` mở
    /dev/video0 luôn thất bại, nên nguồn khung phải đi vòng qua gst hoặc ROS.
  * rclpy chỉ có trong container, nên `/follow/command` + `/follow/target` mới
    publish thật thay vì chỉ in ra console.
  * Cổng 8080 giữ nguyên -> GUI_NATIVE không sửa một dòng nào.

HAI NGUỒN, cùng một đầu ra:
  `--device`  gst đọc MJPG thẳng từ webcam. USB đã nén sẵn nên chỉ còn jpegdec;
              rẻ hơn nhiều so với v4l2_camera_node chuyển YUYV->RGB8 720p bằng CPU
              (đo thật: YUYV 720p qua ROS tụt còn 0.6-0.8 Hz). ĐANG DÙNG.
  `--topic`   subscriber sensor_msgs/Image, cần v4l2_camera_node chạy kèm. Giữ lại
              cho lúc nguồn là camera mô phỏng Gazebo chứ không phải webcam thật.

Hai chỗ được vá, `person_follow_reid.py` giữ nguyên:
  cv2.VideoCapture -> GstCap hoặc RosCap
  cv2.imshow       -> nén JPEG rồi mjpeg_server._luu(); Handler cũ phát ra HTTP
"""
import argparse, subprocess, sys, threading, time
from http.server import ThreadingHTTPServer

import cv2, numpy as np

sys.path[:0] = ["/app", "/home/hoaibac"]        # mjpeg_server.py mount ở /app trong container
import mjpeg_server as mj

_fire_tracker = None
_fire_period = 0.0
_last_fire_offer = 0.0


def _offer_fire_frame(frame):
    """Give the fire thread the newest raw camera frame when it is ready."""
    global _last_fire_offer
    if _fire_tracker is None:
        return
    now = time.monotonic()
    if _fire_period > 0 and now - _last_fire_offer < _fire_period:
        return
    with mj._lock:
        if mj._raw is None:
            mj._raw = frame.copy()
            _last_fire_offer = now


class GstCap:
    """Thay cv2.VideoCapture: gst-launch -> pipe raw BGR. Luồng đọc giữ khung MỚI NHẤT.

    Xin MJPG chứ không YUYV: webcam nén sẵn trong phần cứng nên USB chỉ tải ~1/10
    băng thông, và `jpegdec` rẻ hơn nhiều so với chuyển màu YUYV 720p bằng CPU.
    """

    def __init__(self, device, w, h, fps, stale=2.0, restart_after=5.0, first_grace=20.0):
        self.w, self.h, self.n = w, h, w * h * 3
        self.dev, self.fps = device, fps
        # Do that 18/09/2026: webcam UVC (Microdia 0c45:636b) ngung day khung luc
        # 15:07:37 ma KHONG bao loi gi — nhan khong ghi mot dong dmesg nao, thiet bi
        # van liet ke tren USB, fd van mo. `v4l2src` ket trong poll() vinh vien, nen
        # `_pump` ket trong stdout.read() va `self.frame` dung yen. Truoc ban va nay,
        # read() van tra ve khung cu do => may chu phat lai mai mot khung (do that:
        # 12 khung trong 1,03 s trung khit tung byte) trong khi moi chi so deu dep:
        # 11,6 fps o day, 11,6 fps o GUI, yolo 191 ms khong doi. Dong `[mjpeg] nguon
        # 0.0 Hz` da noi dung su that suot 28 phut ma khong ai doc.
        #
        # `stale`: qua ngan thi nhip model chem (~9 fps) cung bi tinh la chet.
        # `restart_after` > `stale`: bao "khong co hinh" TRUOC, dung lai pipeline SAU.
        # `first_grace` rieng va DAI hon `restart_after`: MJPG 960x540 mat ~8 s ra
        # khung dau (thuong luong USB). Tinh luc chua co khung nao la "im vo han" thi
        # canh sat se giet gst moi giay truoc khi no kip noi — da dinh dung bay do.
        self.stale, self.restart_after, self.first_grace = stale, restart_after, first_grace
        self.frame, self.alive, self.t, self.t0 = None, True, 0.0, 0.0
        self.p = None
        self._spawn()
        threading.Thread(target=self._watch, daemon=True).start()
        print(f"[follow] gst {device} MJPG {w}x{h}@{fps} "
              f"(cu >{stale:g}s = mat hinh, >{restart_after:g}s = dung lai)", flush=True)

    def _spawn(self):
        self.p = subprocess.Popen(
            ["gst-launch-1.0", "-q", "v4l2src", f"device={self.dev}", "!",
             f"image/jpeg,width={self.w},height={self.h},framerate={self.fps}/1", "!",
             "jpegdec", "!", "videoconvert", "!", "video/x-raw,format=BGR", "!",
             "fdsink", "fd=1", "sync=false"],
            # KHONG dat bufsize=0: stdout khi do la FileIO tho, read(n) tra ve dung
            # nhung gi pipe dang co (<=64 KB) chu khong doi du n. Mot khung 720p BGR
            # la 2.7 MB nen se KHONG BAO GIO du -> _pump thoat ngay o khung dau.
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.t0 = time.monotonic()               # moc de tinh han cho KHUNG DAU
        threading.Thread(target=self._pump, args=(self.p,), daemon=True).start()

    def _pump(self, proc):
        while self.alive and proc is self.p:
            buf = proc.stdout.read(self.n)
            if len(buf) < self.n:
                # pipe dut: gst chet, hay _watch vua giet. Phai wait() de thu xac: khong
                # co camera thi gst chet ngay, va _watch dung lai moi 20 s — khong thu
                # xac thi moi lan la mot tien trinh ma.
                try:
                    proc.wait(timeout=3)
                except Exception:
                    pass
                break
            self.frame = np.frombuffer(buf, np.uint8).reshape(self.h, self.w, 3)
            self.t = time.monotonic()
            _offer_fire_frame(self.frame)
            with mj._lock:
                mj._in += 1                      # để dòng [mjpeg] đếm đúng nhịp nguồn

    def age(self):
        """Khung hien tai gia bao lau. Chua co khung nao -> vo cung."""
        return float("inf") if not self.t else time.monotonic() - self.t

    def _watch(self):
        """Nguon im qua lau thi dung lai gst. Giet tien trinh cu lam dut pipe, nho the
        `_pump` cu thoat khoi stdout.read() dang ket — khong co cach nao danh thuc no
        tu ben ngoai."""
        while self.alive:
            time.sleep(1.0)
            # Chua co khung nao thi dem tu luc dung gst va cho `first_grace`; co roi
            # thi dem tu khung cuoi va cho `restart_after`.
            if self.t:
                quiet, han = time.monotonic() - self.t, self.restart_after
            else:
                quiet, han = time.monotonic() - self.t0, self.first_grace
            if quiet <= han:
                continue
            print(f"[follow] nguon im {quiet:.0f}s — dung lai gst", flush=True)
            old = self.p
            self.p = None                        # bao _pump cu dung lai
            try:
                old.kill()
                old.wait(timeout=3)
            except Exception:
                pass
            self.t = 0.0
            self._spawn()

    def isOpened(self):
        for _ in range(400):                     # đo thật: MJPG 720p mất ~8 s ra khung đầu
                                                 # (thương lượng USB), đừng hạ xuống dưới 15 s
            if self.frame is not None:
                return True
            if not self.alive:
                return False
            time.sleep(0.05)
        return False

    def read(self):
        # VAN tra ve khung cu khi qua han: `person_follow_reid` lam `if not ok: break`,
        # tra False o day la giet ca chuong trinh. Chan o cho phat (xem `show` trong
        # main) — luong HTTP im, GUI roi dung vao nhanh "khong co video" san co.
        f = self.frame
        return (f is not None), (None if f is None else f.copy())

    def set(self, *_):  return True              # phân giải đã cố định trong pipeline gst
    def get(self, *_):  return 0.0

    def release(self):
        self.alive = False
        self.p.terminate()


class RosCap:
    """Thay cv2.VideoCapture: giữ khung MỚI NHẤT của một topic sensor_msgs/Image.

    ponytail: không xếp hàng khung. Inference ~5 fps mà camera 30 fps thì hàng đợi
    biến thành độ trễ vài giây — thà bỏ khung còn hơn trả khung quá date (cùng lý do
    mjpeg_server chỉ giữ `_latest`).
    """

    def __init__(self, topic):
        import rclpy
        from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
        from sensor_msgs.msg import Image

        self.rclpy, self.frame, self.alive = rclpy, None, True
        if not rclpy.ok():                       # pf.init_ros() thường đã init trước
            rclpy.init()
        self.node = rclpy.create_node("person_follow_cam")
        # BEST_EFFORT: camera ROS2 publish BEST_EFFORT, subscriber để RELIABLE thì
        # KHÔNG BAO GIỜ khớp và không có một dòng lỗi nào — chỉ là không có khung.
        qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                         history=QoSHistoryPolicy.KEEP_LAST)
        self.node.create_subscription(Image, topic, self._on, qos)
        threading.Thread(target=self._spin, daemon=True).start()
        print(f"[follow] đọc ROS2 topic {topic}", flush=True)

    def _on(self, msg):
        img = mj._imgmsg_to_bgr(msg)             # None nếu encoding lạ -> bỏ, đừng đoán
        if img is None:
            return
        self.frame = np.ascontiguousarray(img)   # rgb8 trả về view lật kênh + read-only
        _offer_fire_frame(self.frame)
        with mj._lock:
            mj._in += 1                          # để dòng [mjpeg] đếm đúng nhịp nguồn

    def _spin(self):
        try:
            self.rclpy.spin(self.node)
        except Exception:
            pass                                 # shutdown_ros() kéo context đi trước

    def isOpened(self):
        for _ in range(200):                     # v4l2_camera_node mất vài giây mới lên
            if self.frame is not None:
                return True
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
        return False

    def read(self):
        f = self.frame
        return (f is not None), (None if f is None else f.copy())

    def set(self, *_):  return True              # phân giải do v4l2_camera_node quyết
    def get(self, *_):  return 0.0

    def release(self):
        self.alive = False
        self.node.destroy_node()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", default=None, help="vd /dev/video0 -> nguồn gst MJPG")
    p.add_argument("--topic", default="/image_raw", help="dùng khi không có --device")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=15, help="fps xin từ camera")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--quality", type=int, default=60)
    p.add_argument("--stream-every", type=int, default=1,
                   help="chi nen 1 trong N khung ra HTTP; vong dieu khien van chay du nhip")
    p.add_argument("--model", default=None,
                   help="onnx pose, mac dinh pf.POSE_ONNX (416). 320: nhanh ~2x, sot nguoi o xa")
    p.add_argument("--conf", type=float, default=None)
    p.add_argument("--detect-only", action="store_true",
                   help="detect tat ca nguoi, bo ReID/FOLLOW/ROS2")
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--fire-model", default=None,
                   help="onnx fire/smoke model chay song song va ve hop len stream")
    p.add_argument("--fire-conf", type=float, default=0.2)
    p.add_argument("--fire-threads", type=int, default=2)
    p.add_argument("--fire-hz", type=float, default=0.0,
                   help="gioi han nhip suy dien fire; 0 = nhanh nhat co the")
    a = p.parse_args()

    import person_follow_reid as pf

    global _fire_tracker, _fire_period
    if a.fire_model:
        from fire_tracker import FireTracker

        _fire_period = 1.0 / a.fire_hz if a.fire_hz > 0 else 0.0
        _fire_tracker = FireTracker(a.fire_model, conf=a.fire_conf,
                                    threads=a.fire_threads,
                                    confirm_hits=1, max_miss=2)
        threading.Thread(target=mj._yolo_loop, args=(_fire_tracker,), daemon=True).start()
        lop = ([_fire_tracker.names[i] for i in sorted(_fire_tracker.keep_cls)]
               if _fire_tracker.keep_cls is not None
               else list(_fire_tracker.names.values()))
        print(f"[fire] model {a.fire_model} conf={a.fire_conf} "
              f"threads={a.fire_threads} hz={a.fire_hz:g} lop={lop}", flush=True)

    cap_ref = []                                 # `show` can hoi tuoi khung cua chinh no

    def _make_cap(*_, **__):
        c = GstCap(a.device, a.width, a.height, a.fps) if a.device else RosCap(a.topic)
        cap_ref.append(c)
        return c

    cv2.VideoCapture = _make_cap

    # Chi phat khung pf DA VE XONG: hop/xuong khop 100% voi hinh, video di theo nhip
    # model (~9 fps voi model 320). Da thu tach hinh ra 15 fps va dan lop phu len khung
    # moi hon (2026-09-15): lop phu tre 160-290 ms, lech ro khi vung tay/di nhanh; bu
    # bang optical flow (9.8 vs 10.1 px) hay ngoai suy van toc (18.2 vs 16.5 px) deu
    # KHONG hon khong bu — o cu ly gan hop doi hinh dang (tay, chan) chu khong truot
    # deu. Muon hinh nhanh hon ma van khop thi phai lam model nhanh hon.
    n_frame = [0]

    def show(_win, frame):
        n_frame[0] += 1
        if n_frame[0] % a.stream_every:
            return
        # Nguon chet thi THOI PHAT, chu khong phat lai khung cuoi. Mot luong im la
        # thu GUI phat hien duoc (no canh khoang ngat 4 s roi bao "khong co video"
        # kem dia chi); mot luong dep dan toan anh cu thi khong ai phat hien duoc.
        cap = cap_ref[0] if cap_ref else None
        if cap is not None and getattr(cap, "age", None) and cap.age() > cap.stale:
            return
        if _fire_tracker is not None:
            frame = np.ascontiguousarray(frame)
            _fire_tracker.draw_cached(frame)
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, a.quality])
        if ok:
            mj._luu(jpg.tobytes())
    cv2.imshow = show
    cv2.waitKey = lambda *_: -1                  # không có cửa sổ: phím 'q'/'r' vô hiệu
    cv2.destroyAllWindows = lambda *_: None

    srv = ThreadingHTTPServer(("0.0.0.0", a.port), mj.Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    threading.Thread(target=mj._stats_loop, args=(5.0, _fire_tracker), daemon=True).start()
    print(f"[mjpeg] http://0.0.0.0:{a.port}/stream  <- person_follow_reid "
          f"q{a.quality} 1/{a.stream_every} khung", flush=True)

    args = argparse.Namespace(model=a.model or pf.POSE_ONNX, reid=pf.REID_ONNX, source="0",
                              conf=a.conf if a.conf is not None else pf.CONF,
                              threads=a.threads, width=a.width, height=a.height,
                              detect_only=a.detect_only)
    try:
        pf.main(args)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
