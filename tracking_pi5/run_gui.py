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
import argparse, os, subprocess, sys, threading, time
from http.server import ThreadingHTTPServer

import cv2, numpy as np

# /app: container tren Pi. Thu muc cha: chay thang tu repo (Jetson).
sys.path[:0] = ["/app", str(__import__("pathlib").Path(__file__).resolve().parent.parent)]
import mjpeg_server as mj
import latency_stamp

_fire_tracker = None
_pub_count = None  # std_msgs/Int32 /vision/person_count, xem _init_count_pub()


def _init_count_pub():
    """Publisher so nguoi. Khong co rclpy (chay tran ngoai container) thi bo qua, van stream."""
    global _pub_count
    try:
        import rclpy
        from std_msgs.msg import Int32
    except ImportError as e:
        print(f"[ROS] khong co rclpy ({e}) -> KHONG publish /vision/person_count", flush=True)
        return
    try:
        if not rclpy.ok():                   # follow mode: pf.init_ros() da init truoc
            # Khong de rclpy bat SIGTERM: no chi tat ROS ma KHONG thoat, `systemctl
            # restart` treo video ~13 s roi chet o publish ("context is invalid").
            from rclpy.signals import SignalHandlerOptions
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
        node = rclpy.create_node("vision_counter")
    except Exception as e:  # noqa: BLE001
        # CycloneDDS chet khi tailscale0 chua len. Mat topic con hon mat ca video.
        print(f"[ROS] khong tao duoc node ({e}) -> KHONG publish so nguoi", flush=True)
        return
    _pub_count = (node.create_publisher(Int32, "/vision/person_count", 10), Int32)
    print("[ROS] publish /vision/person_count (std_msgs/Int32)", flush=True)
_fire_period = 0.0
_last_fire_offer = 0.0
_seg = {}  # --stamp: ten doan -> [ms], show() in p50/p90 moi 5 s


def _seg_add(k, sec):
    _seg.setdefault(k, []).append(sec * 1000)


def _seg_report():
    global _seg
    d, _seg = _seg, {}
    print("[doan] " + "  ".join(f"{k} {sorted(v)[len(v) // 2]:.1f}/{sorted(v)[int(len(v) * .9)]:.1f}"
                                for k, v in d.items()) + "  (ms p50/p90)", flush=True)


_fire_slot = None  # (khung, meta) cho _fire_loop: meta (gio chup, vung cat) di CUNG khung
_fire_ros = None   # _FireRos khi co ROS
_fire_saver = None  # _FireSaver khi --fire-save
_fire_verify = None  # fire_verify.FireVerify khi --fire-verify: Gemini loc vet truoc khi ra ROS
_fire_scan = 0.0     # --fire-scan N: Gemini quet nguyen khung moi N s khi drone arm / /fire/scan true


class _FireSaver:
    """--fire-save DIR: khung CO hop lua -> DIR/<ngay>/<gio>_x<zoom>_d<diem>.jpg + .txt nhan YOLO
    (lop cx cy w h, chuan hoa) cua chinh cac hop model doan. Train lai: mo bang cong cu gan nhan,
    XOA hop sai (vd thanh an toan vang cua robot bi bao Fire 18/19 khung, 23/10/2026) -> anh am,
    sua hop dung -> anh duong. Luu NGUYEN khung model thay (chua ve hop), khong cat quanh hop:
    YOLO hoc ca nen xung quanh, anh cat thi mat ngu canh do."""
    PERIOD = 1.0     # ponytail: <= 1 khung/s (~150 KB) ~ 0.5 GB/gio thay lua lien tuc
    MAX_BYTES = 5e9  # ponytail: moi lan chay luu toi da 5 GB; chuyen thu muc di roi chay lai
    MIN_FREE = 10e9  # chua bao gio lam day dia: con < 10 GB trong thi thoi luu

    def __init__(self, root, cam_w, names):
        self.root, self.cam_w, self.t, self.bytes, self.on = root, cam_w, 0.0, 0, True
        os.makedirs(root, exist_ok=True)
        with open(os.path.join(root, "classes.txt"), "w") as f:  # thu tu lop cho LabelImg/YOLO
            f.write("\n".join(names[i] for i in sorted(names)) + "\n")
        print(f"[fire] luu khung co lua -> {root} (<= 1 khung/s)", flush=True)

    def __call__(self, img, tracks, meta):
        now = time.monotonic()
        if not (self.on and tracks) or now - self.t < self.PERIOD:
            return
        self.t = now
        zoom = self.cam_w / meta[1][2] if meta else 0.0  # 0 = khung ngay sau doi zoom
        d = os.path.join(self.root, time.strftime("%Y%m%d"))
        base = os.path.join(d, time.strftime("%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
                            f"_x{zoom:.1f}_d{max(t.score for t in tracks):.2f}")
        h, w = img.shape[:2]
        try:  # moi loi dia (ca disk_usage khi thu muc mat) chi dung luu, khong giet luong lua
            import shutil
            if self.bytes > self.MAX_BYTES or shutil.disk_usage(self.root).free < self.MIN_FREE:
                self.on = False
                print(f"[fire] DUNG luu khung: da luu {self.bytes / 1e9:.1f} GB / dia sap day",
                      flush=True)
                return
            os.makedirs(d, exist_ok=True)
            ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            with open(base + ".jpg", "wb") as f:
                f.write(jpg.tobytes())
            with open(base + ".txt", "w") as f:
                for t in tracks:
                    x1, y1, x2, y2 = t.box
                    f.write(f"{t.cls} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
                            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}\n")
            self.bytes += len(jpg)
        except OSError as e:
            self.on = False  # loi ghi dia khong duoc lam chet luong lua
            print(f"[fire] DUNG luu khung: {e}", flush=True)


def _offer_fire_frame(frame, meta=None):
    """Give the fire thread the newest raw camera frame when it is ready."""
    global _last_fire_offer, _fire_slot
    if _fire_tracker is None:
        return
    now = time.monotonic()
    if _fire_period > 0 and now - _last_fire_offer < _fire_period:
        return
    with mj._lock:
        if _fire_slot is None:
            _fire_slot = (frame.copy(), meta)
            _last_fire_offer = now


def _fire_loop(tracker):
    """Nhu mj._yolo_loop, them: hop lua cua khung co meta -> /fire/detections (xem _FireRos).
    Khung va meta lay CUNG mot lan tu _fire_slot: hai bien rieng thi zoom doi giua chung la hop
    bi doi nguoc theo vung cat cua khung khac - lua lech hang chuc met ma khong ai thay."""
    global _fire_slot
    while True:
        with mj._lock:
            item, _fire_slot = _fire_slot, None
        if item is None:
            time.sleep(0.005)
            continue
        img, meta = item
        try:
            tracks = tracker.infer(img)
        except Exception as e:  # noqa: BLE001
            print(f"[track] loi suy dien: {type(e).__name__}: {e}", flush=True)
            time.sleep(1.0)
            continue
        if _fire_saver is not None:
            _fire_saver(img, tracks, meta)  # ca hop Gemini bac: chinh la anh bao nham can de train lai
        if _fire_verify is not None:
            if _fire_scan and _fire_ros is not None and (_fire_ros.armed or _fire_ros.scan_cmd):
                _fire_verify.scan(img, tracks, _fire_scan, meta)  # hop YOLO CHUA loc: so ca hop Gemini bac
            tracks = _fire_verify.filter(img, tracks)
        if meta is not None and _fire_ros is not None:
            _fire_ros.publish_fire(tracks, meta, img.shape, tracker.names)


PICTURE_FLAG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "fire_picture")


class _PictureFlag:
    """FireVerify.picture: --fire-picture HOAC co file ~/jetson/fire_picture. Doc lai moi lan Gemini hoi nen bat/tat
    bang `touch` / `rm` qua ssh, KHONG khoi dong lai service (dong lenh service nam o /etc, sua phai sudo)."""

    def __init__(self, on):
        self.on = on

    def __bool__(self):
        return self.on or os.path.exists(PICTURE_FLAG)


def _gemini_fire(boxes, meta, shape):
    """Gemini quet ca khung thay lua YOLO SOT -> /fire/detections nhu hop YOLO, diem 1.0. Den tre 1-7 s:
    stamp = gio CHUP khung (meta), fire_locator lay pose drone dung luc do."""
    if _fire_ros is None or meta is None:
        return
    from types import SimpleNamespace
    cls = min(_fire_tracker.keep_cls) if _fire_tracker.keep_cls else 0
    _fire_ros.publish_fire([SimpleNamespace(box=b, score=1.0, cls=cls) for b in boxes], meta, shape,
                           _fire_tracker.names)


class _FireRos:
    """ROS cho task Chua chay (fire_drone tren Jetson), mot node + mot luong:

      /fire/zoom (Float32, latched) <- fire_mission: doi vung cat camera LUC DANG BAY
      /fire/detections (vision_msgs/Detection2DArray) -> fire_locator: tam hop lua theo diem anh
          KHUNG GOC cam_w x cam_h (he calib drone_cam_3840x2160.yaml) - doi nguoc zoom o day
      ground_range (fire_drone) chay NGAY TRONG tien trinh nay, an nhanh xam toan canh cua
          gst_zoom.py: 960x540 x 30 Hz qua DDS tailscale (goi 1200 B) la nghen, mat khung luc ha.

    Mot luong vua spin vua dua khung xam cho ground_range: khong khoa, khong dua nhau.
    Node khac tren Jetson (MAVROS, fire_mission) phai cung RMW + CYCLONEDDS_URI moi thay topic.
    """

    def __init__(self, cap_ref):
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from std_msgs.msg import Float32
        from vision_msgs.msg import Detection2D, Detection2DArray, ObjectHypothesisWithPose
        self.rclpy, self.cap_ref = rclpy, cap_ref
        self.Detection2D, self.Detection2DArray, self.Hyp = Detection2D, Detection2DArray, ObjectHypothesisWithPose
        self.node = rclpy.create_node("fire_camera")
        self.pub = self.node.create_publisher(Detection2DArray, "/fire/detections", 10)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.node.create_subscription(Float32, "/fire/zoom", self._on_zoom, latched)
        # --fire-scan chi chay khi dang bay (arm) hoac bat tay: ros2 topic pub -t 3 /fire/scan std_msgs/msg/Bool
        # "{data: true}". Best effort: khop ca publisher reliable lan best effort (MAVROS).
        from rclpy.qos import qos_profile_sensor_data
        from std_msgs.msg import Bool
        self.armed = self.scan_cmd = False
        self.node.create_subscription(Bool, "/fire/scan", self._on_scan, qos_profile_sensor_data)
        try:
            from mavros_msgs.msg import State
            self.node.create_subscription(State, "/mavros/state", self._on_state, qos_profile_sensor_data)
        except ImportError:
            pass
        self.ex = SingleThreadedExecutor()
        self.ex.add_node(self.node)
        try:
            from fire_drone.ground_range import GroundRange
            self.gr = GroundRange()
            self.ex.add_node(self.gr)
        except ImportError as e:
            self.gr = None
            print(f"[fire] khong co fire_drone ({e}) -> KHONG do mat dat. "
                  "source ~/Firefighting-drone/install/setup.bash truoc", flush=True)
        threading.Thread(target=self._loop, daemon=True).start()
        print("[ROS] /fire/zoom -> camera, lua -> /fire/detections"
              + (", ground_range trong tien trinh" if self.gr else ""), flush=True)

    def _cap(self):
        return self.cap_ref[0] if self.cap_ref else None

    def _on_scan(self, msg):
        if bool(msg.data) != self.scan_cmd:
            print(f"[scan] /fire/scan {'BAT' if msg.data else 'TAT'}", flush=True)
        self.scan_cmd = bool(msg.data)

    def _on_state(self, msg):
        if bool(msg.armed) != self.armed:
            print(f"[scan] drone {'ARM' if msg.armed else 'DISARM'}", flush=True)
        self.armed = bool(msg.armed)

    def _on_zoom(self, msg):
        cap = self._cap()
        if hasattr(cap, "set_zoom"):
            cap.set_zoom(float(msg.data))

    def _loop(self):
        n, t_log = 0, time.monotonic() + 5
        while self.rclpy.ok():
            on = self.gr is not None and self.gr.enabled
            self.ex.spin_once(timeout_sec=0.0 if on else 0.1)
            cap = self._cap()
            if self.gr is None or not hasattr(cap, "pull_gray"):
                continue
            cap.set_gray(on)
            if on:
                g = cap.pull_gray(0.05)
                if g is not None:
                    self.gr.on_gray(g)
                    n += 1
                if time.monotonic() >= t_log:
                    # 0 khung luc dang ha = ground_range mu, drone ha theo muc cat canh
                    print(f"[fire] do mat dat: {n / 5:.1f} khung xam/s", flush=True)
                    n, t_log = 0, time.monotonic() + 5

    def publish_fire(self, tracks, meta, shape, names):
        t_wall, (l, t, cw, ch) = meta
        h, w = shape[:2]
        sx, sy = cw / w, ch / h
        msg = self.Detection2DArray()
        msg.header.stamp.sec = int(t_wall)
        msg.header.stamp.nanosec = int((t_wall % 1) * 1e9)
        msg.header.frame_id = "drone_cam"
        for tr in tracks:
            x1, y1, x2, y2 = tr.box
            d = self.Detection2D()
            # tam diem anh lech 0.5 nhu marker_detector.zoom_k: cat roi co gian
            d.bbox.center.position.x = l + ((x1 + x2) / 2 + 0.5) * sx - 0.5
            d.bbox.center.position.y = t + ((y1 + y2) / 2 + 0.5) * sy - 0.5
            d.bbox.size_x, d.bbox.size_y = (x2 - x1) * sx, (y2 - y1) * sy
            hyp = self.Hyp()
            hyp.hypothesis.class_id = names.get(tr.cls, str(tr.cls))
            hyp.hypothesis.score = float(tr.score)
            d.results.append(hyp)
            msg.detections.append(d)
        self.pub.publish(msg)  # ca khi rong: fire_locator bo qua, con ros2 topic hz thay song


class H264Out:
    """Nen H.264 bang x264 (CPU) va DAY MPEG-TS qua UDP toi tram mat dat.

    Orin Nano khong co bo nen phan cung, nen x264 superfast: do 960x540@30 tren 90 khung
    that ~1,1 nhan, 1,5 Mbps ra 35,9 dB (ultrafast 34,9 dB, veryfast 36,0 dB ma ton them
    15% CPU). Nhe hon MJPEG ~13 lan - danh cho 4G.

    Luong rieng + giu KHUNG MOI NHAT: ffmpeg cham thi bo khung, KHONG bao gio chan vong
    nhan dien. Timestamp theo dong ho that nen bo khung khong lam video chay nhanh.
    pkt_size 1128 = 6 goi TS 188 B: + header IP/UDP van duoi MTU 1280 cua Tailscale
    (1316 mac dinh thi bi phan manh, mat 1 manh la mat ca goi).

    fb_port: tram mat dat (h264_feedback.py) bao "loss=<ti le>" moi giay ve cong UDP nay.
    Mat > LOSS_HI -> x0.7 bitrate; sach < LOSS_LO lien UP_AFTER s -> +UP_STEP, tran la
    --h264-kbps. Chua nhan bao cao nao thi giu bitrate co dinh (GUI cu van chay).
    ponytail: ffmpeg 4.4 khong doi bitrate luc chay -> doi = khoi dong lai ffmpeg (~1 khung
    IDR moi, giat nhe), nen co DOWN_GAP/UP_AFTER chong doi lien tuc.
    """
    LOSS_HI, LOSS_LO = 0.02, 0.005
    # Khoi dong lai ffmpeg reset CC cua TS -> tram bao "mat" gia trong ~1 s ke tiep; _spawn()
    # dat last_change va ca hai khoang cho deu >= 2 s nen bao cao do bi bo qua.
    DOWN_GAP, UP_AFTER, UP_STEP = 2.0, 5.0, 150
    SILENT = 5.0                       # da tung co bao cao ma im 5 s = mat het -> ha

    def __init__(self, dest, kbps, preset, fb_port=0, kbps_min=400, gamma=1.0, ir=False):
        self.dest, self.kbps, self.preset, self.ir = dest, kbps, preset, ir
        # Nhieu dich "A:5600,B:5600" (laptop + dien thoai): ffmpeg van MOT lenh nhu cu (giu
        # muxdelay/PES da do tre) nhung day vao cong noi bo, luong _fanout chep tung goi ra
        # moi dich. Ca hai tram cung bao mat goi ve fb_port -> bitrate theo tram te hon
        # (chung mot duong len 4G). ponytail: khong dung -f tee vi chua chac tuy chon mux
        # xuong toi tung dich.
        if "," in dest:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.bind(("127.0.0.1", 0))
            self.dest = f"127.0.0.1:{s.getsockname()[1]}"
            outs = [(socket.gethostbyname(h), int(p))
                    for h, p in (d.strip().rsplit(":", 1) for d in dest.split(","))]
            threading.Thread(target=self._fanout, args=(s, outs), daemon=True).start()
            print(f"[h264] chep ra {len(outs)} dich: {dest}", flush=True)
        # Gamma >1 nang vung toi, giu vung sang - di kem HighlightAE (phoi sang theo den nen
        # khung toi). LUT ~1 ms 720p, chi anh video; model van nhan khung goc. Da thu CLAHE
        # (15/10/2026): nhieu gap doi (Laplacian 421 -> 719), x264 1,5 Mbps nhoe -> bo.
        self.lut = ((np.arange(256) / 255) ** (1 / gamma) * 255).astype(np.uint8) \
            if gamma != 1.0 else None
        self.kbps_max, self.kbps_min, self.run_kbps = kbps, kbps_min, None
        self.p, self.size, self.frame, self.t_push = None, None, None, None
        self.last_change = self.last_rx = 0.0
        self.cv = threading.Condition()
        threading.Thread(target=self._pump, daemon=True).start()
        if fb_port:
            threading.Thread(target=self._feedback, args=(fb_port,), daemon=True).start()

    def adapt(self, loss, now):
        """Mot buoc dieu khien AIMD; tra ve kbps moi (= cu neu khong doi)."""
        if loss > self.LOSS_HI and now - self.last_change >= self.DOWN_GAP:
            k = max(self.kbps_min, int(self.kbps * 0.7))
        elif loss < self.LOSS_LO and now - self.last_change >= self.UP_AFTER:
            k = min(self.kbps_max, self.kbps + self.UP_STEP)
        else:
            return self.kbps
        if k != self.kbps:
            print(f"[h264] loss {loss:.1%} -> {self.kbps} => {k} kbps", flush=True)
            self.kbps, self.last_change = k, now
        return k

    def _feedback(self, port):
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.bind(("0.0.0.0", port))
        s.settimeout(1.0)
        print(f"[h264] nghe phan hoi mat goi o udp :{port}", flush=True)
        while True:
            try:
                msg = s.recv(64)
                loss = float(msg.decode().split("=", 1)[1])
                self.last_rx = time.monotonic()
            except socket.timeout:
                if not self.last_rx or time.monotonic() - self.last_rx < self.SILENT:
                    continue
                loss = 1.0
            except (ValueError, IndexError, UnicodeDecodeError):
                continue
            self.adapt(loss, time.monotonic())

    @staticmethod
    def _fanout(s, outs):
        import socket
        o = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        while True:
            pkt = s.recv(2048)
            for d in outs:
                try:
                    o.sendto(pkt, d)
                except OSError:
                    pass  # mot tram tat Tailscale khong duoc cat hinh cua tram kia

    def _spawn(self):
        w, h = self.size
        self.run_kbps = self.kbps
        # Moi lan bat ffmpeg (khoi dong service, doi bitrate, ffmpeg chet, doi co khung) CC
        # cua TS ve 0 -> tram bao mat gia. Tinh nhu mot lan "doi" de adapt() bo qua bao cao do.
        self.last_change = time.monotonic()
        br = f"{self.kbps}k"
        # --h264-ir: intra-refresh = lam moi dan tung cot trong 30 khung thay IDR moi giay, VBV
        # 100 ms. Do 27/10 (testsrc 960x540 900k): khung lon nhat 13,6 -> 8,7 KB, khong con cuc
        # IDR don len 4G. Vao giua luong cham hon ~1 s (doi SPS + het mot vong lam moi).
        ir = ["-intra-refresh", "1"] if self.ir else []
        vbv = f"{self.kbps // 10}k" if self.ir else br
        self.p = subprocess.Popen(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-f", "rawvideo",
             "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-use_wallclock_as_timestamps", "1",
             "-i", "-", "-c:v", "libx264", "-preset", self.preset, "-tune", "zerolatency",
             "-pix_fmt", "yuv420p", "-b:v", br, "-maxrate", br, "-bufsize", vbv, *ir,
             "-g", "30", "-threads", "2", "-vsync", "passthrough",  # ffmpeg 4.4 (22.04): chua co -fps_mode
             # muxdelay mac dinh 0.7 s: TS giu goi lai -> tre ~680 ms (do). Tat thi ~1 khung.
             "-muxdelay", "0", "-muxpreload", "0", "-flush_packets", "1",
             # PES video mac dinh khong ghi do dai -> demux tram phai doi goi KHUNG SAU moi
             # biet khung nay het: +1 khung (loopback 16/10/2026: 86 -> 53 ms).
             "-omit_video_pes_length", "0",
             "-f", "mpegts", f"udp://{self.dest}?pkt_size=1128"],
            stdin=subprocess.PIPE)
        print(f"[h264] x264 {self.preset} {w}x{h} {self.kbps} kbps{' intra-refresh' if self.ir else ''}"
              f" -> udp://{self.dest}", flush=True)

    def push(self, frame, t_push=None):
        with self.cv:
            self.frame, self.t_push = frame, t_push
            self.cv.notify()

    def _pump(self):
        while True:
            with self.cv:
                while self.frame is None:
                    self.cv.wait()
                f, self.frame = self.frame, None
                t_push = self.t_push
            t_got = time.monotonic()
            if (self.p is None or self.p.poll() is not None or f.shape[1::-1] != self.size
                    or self.run_kbps != self.kbps):
                if self.p is not None:
                    self.p.kill()
                    self.p.wait()
                self.size = f.shape[1::-1]
                self._spawn()
            try:
                if self.lut is not None:
                    f = cv2.LUT(f, self.lut)
                self.p.stdin.write(f.tobytes())
                if t_push:
                    # ghi = LUT + day 2,7 MB vao pipe; ffmpeg dang nen khung truoc thi cho ca do
                    _seg_add("hang", t_got - t_push)
                    _seg_add("ghi", time.monotonic() - t_got)
            except (BrokenPipeError, OSError) as e:
                print(f"[h264] ffmpeg chet ({e}) - dung lai o khung sau", flush=True)
                self.p = None
                time.sleep(1.0)


_CID = {"auto_exposure": 0x009A0901, "exposure_time_absolute": 0x009A0902,
        "gain": 0x00980913, "backlight_compensation": 0x0098091C}


def v4l2_set(dev, **ctrls):
    """Ghi control camera bang ioctl VIDIOC_S_CTRL - chi SET dung cai can.

    KHONG goi v4l2-ctl: moi lan no doc lai MOI control (GET_CUR) va webcam nay chiu khong
    noi - 15/10/2026: v4l2-ctl ngay truoc khi gst mo camera -> USB reset moi lan khoi dong
    service; hoi don dap -> tra -71 roi tu rut khoi USB, phai cam lai bang tay.
    """
    import fcntl, os, struct
    try:
        fd = os.open(dev, os.O_RDWR | os.O_NONBLOCK)
    except OSError:
        return                                   # camera dang rut
    try:
        for k, v in ctrls.items():
            try:                                 # _IOWR('V', 28, struct v4l2_control)
                fcntl.ioctl(fd, 0xC008561C, struct.pack("Ii", _CID[k], v))
            except OSError:
                pass
    finally:
        os.close(fd)


def find_camera():
    """Camera USB dau tien (node index 0 = node chup; index 1 la metadata) hoac None.

    Chi doc sysfs, KHONG mo/hoi camera (xem v4l2_set: hoi nhieu -> webcam tu rut USB).
    """
    import glob, os
    for d in sorted(glob.glob("/sys/class/video4linux/video*"),
                    key=lambda d: int(d.rsplit("video", 1)[1])):
        try:
            if (open(d + "/index").read().strip() == "0"
                    and "/usb" in os.path.realpath(d + "/device")):
                return "/dev/" + os.path.basename(d)
        except OSError:
            pass                                 # vua rut giua chung
    return None


def resolve_dev(dev):
    return find_camera() if dev == "auto" else dev


# Jetson co giai ma JPEG phan cung (Pi thi khong -> jpegdec CPU)
HW_JPEG = subprocess.run(["gst-inspect-1.0", "nvv4l2decoder"], capture_output=True).returncode == 0


class HighlightAE:
    """Tu phoi sang UU TIEN VUNG SANG: den/troi khong chay, vung toi de --gamma keo len.

    Auto cua webcam do theo trung binh ca khung -> den chay loang tron (do 15/10/2026:
    3,5% diem anh >=250, p99 254). Webcam khong co exposure bias/HDR, nen tat auto va tu
    chinh exposure_time_absolute moi PERIOD s, nhan/chia STEP:
      median < MED_MIN            -> sang len (khong de ca khung den vi mot ngon den)
      chay > CLIP_HI va median du -> toi di
      chay < CLIP_LO              -> sang len (con du cho, lay them chi tiet vung toi)
    Tran e_max (100 = 10 ms) chong nhoe chuyen dong; auto chon ~15,6 ms trong phong.
    Mot bien "muc sang" self.e: e_min..e_max la exposure, phan vuot e_max la gain (0..G_MAX)
    -> toi thi dai exposure truoc roi moi tang gain (nhieu); sang thi ha gain truoc.
    Do 15/10/2026 chieu: phong toi hon, e_max 100 gain 0 chi ra median ~36 -> can gain.

    Moi RESEND lan gui lai ca bon gia tri: camera bi rut/cam lai thi ve mac dinh (auto),
    khong gui lai thi AE chinh vao khoang khong.
    """
    # MED_MIN 95 chu khong 55: YOLO + model lua nhan khung GOC (gamma chi cho video). 55
    # (15/10/2026) toi qua -> bong den sang giua nen toi bi model lua bat thanh "lua".
    CLIP_HI, CLIP_LO, MED_MIN, STEP, PERIOD, G_MAX, RESEND = 0.02, 0.008, 95, 1.15, 0.5, 100, 20

    def __init__(self, device, cap_ref, e_max, e_min=3):
        self.dev, self.cap_ref, self.e_max, self.e_min = device, cap_ref, e_max, e_min
        self.e = e_max                           # ghi camera o khung dau trong _loop, KHONG o day
        threading.Thread(target=self._loop, daemon=True).start()
        print(f"[ae] uu tien vung sang, exposure {e_min}..{e_max} (x100 us) + gain 0..{self.G_MAX}",
              flush=True)

    def _set(self, **ctrls):
        cap = self.cap_ref[0] if self.cap_ref else None
        dev = getattr(cap, "dev", None) or resolve_dev(self.dev)   # --device auto: theo node gst dang mo
        if dev:
            v4l2_set(dev, **ctrls)

    def step(self, gray):
        """Mot buoc dieu khien tren anh xam; tra ve muc sang moi."""
        clip, med = np.mean(gray >= 250), np.median(gray)
        e = self.e
        if med < self.MED_MIN:
            e *= self.STEP
        elif clip > self.CLIP_HI and med > self.MED_MIN * 1.2:   # 1.2: vung dem, khong dao dong
            e /= self.STEP
        elif clip < self.CLIP_LO:
            e *= self.STEP
        return min(self.e_max + self.G_MAX, max(self.e_min, e))   # so thuc: lam tron thi ket o e nho

    def split(self, level=None):
        """muc sang -> (exposure, gain) nguyen."""
        level = self.e if level is None else level
        return round(min(level, self.e_max)), round(max(0.0, level - self.e_max))

    def _loop(self):
        n = 0
        while True:
            time.sleep(self.PERIOD)
            cap = self.cap_ref[0] if self.cap_ref else None
            # Khung CU (camera chet) thi dung yen: chinh theo anh dong bang da day gain len 100.
            if cap is None or cap.frame is None or cap.age() > cap.stale:
                continue
            g = cv2.cvtColor(cap.frame[::8, ::8], cv2.COLOR_BGR2GRAY)
            old, self.e = self.split(), self.step(g)
            (e, gn), n = self.split(), n + 1
            if n % 30 == 0:
                print(f"[ae] exposure {e} gain {gn}  chay {np.mean(g >= 250):.1%}  "
                      f"median {np.median(g):.0f}", flush=True)
            if n % self.RESEND == 1:                 # n=1: khung dau, camera da dang phat
                self._set(auto_exposure=1, exposure_time_absolute=e, gain=gn, backlight_compensation=0)
            elif (e, gn) != old:
                self._set(exposure_time_absolute=e, gain=gn)


try:  # gi chi can cho gst_zoom.py; khong co thi GstCap dung gst-launch nhu cu (zoom co dinh)
    import gi
    gi.require_version("Gst", "1.0")
    HAVE_GI = True
except (ImportError, ValueError):
    HAVE_GI = False
GST_ZOOM = str(__import__("pathlib").Path(__file__).resolve().parent / "gst_zoom.py")
GRAY = (960, 540)  # nhanh xam toan canh cua gst_zoom.py, cho ground_range


class GstCap:
    """Thay cv2.VideoCapture: gst-launch -> pipe raw BGR. Luồng đọc giữ khung MỚI NHẤT.

    Xin MJPG chứ không YUYV: webcam nén sẵn trong phần cứng nên USB chỉ tải ~1/10
    băng thông, và `jpegdec` rẻ hơn nhiều so với chuyển màu YUYV 720p bằng CPU.
    """

    def __init__(self, device, w, h, fps, stale=1.0, restart_after=1.5, first_grace=20.0,
                 cam_w=None, cam_h=None, zoom=1.0):
        self.w, self.h, self.n = w, h, w * h * 3
        # zoom: quay cam_w x cam_h roi cat GIUA 1/zoom moi chieu, ve lai w x h. Quay
        # 1080p, zoom 2 -> cat 960x540 diem anh THAT (khong phong anh), nguoi o xa to
        # gap doi trong mat model. Doi lai mat 3/4 goc nhin, xem HFOV trong main().
        self.cam_w, self.cam_h, self.zoom = cam_w or w, cam_h or h, zoom
        self.zoom_t, self.gray_on = 0.0, False
        self._gray, self._gray_new = None, threading.Event()
        # want: duong dan hoac "auto" (tim lai camera USB moi lan dung gst -> cam lai, doi cong,
        # doi han camera khac deu tu len hinh). dev: node dang mo that.
        self.want, self.dev, self.fps = device, None, fps
        self.mode = 0                            # chi so MODES: camera khong ra khung thi thu cai sau
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
        self._t_read = 0.0                       # dau thoi gian cua khung read() tra lan truoc
        self.p = None
        self._spawn()
        threading.Thread(target=self._watch, daemon=True).start()
        print(f"[follow] gst {device} MJPG {w}x{h}@{fps} "
              f"(cu >{stale:g}s = mat hinh, >{restart_after:g}s = dung lai)", flush=True)

    def _spawn(self):
        dev = resolve_dev(self.want)
        if dev != self.dev:
            print(f"[follow] camera: {dev or 'KHONG THAY camera USB - doi cam vao'}", flush=True)
        self.dev = dev
        if HW_JPEG and HAVE_GI and self.mode == 0:
            # gst_zoom.py = gst-launch + doi zoom luc chay + nhanh xam (xem docstring file do).
            # Tien trinh CON chu khong gi trong tien trinh: nvjpegdec thinh thoang SEGV.
            r, w = os.pipe()
            self.p = subprocess.Popen(
                [sys.executable, GST_ZOOM, dev or "/dev/null", str(self.cam_w), str(self.cam_h),
                 str(self.fps), str(self.w), str(self.h), *map(str, self._box()), str(w)],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, pass_fds=(w,))
            os.close(w)
            self.p.gray = os.fdopen(r, "rb")
            self._send(f"gray {int(self.gray_on)}")
            threading.Thread(target=self._pump_gray, args=(self.p,), daemon=True).start()
            self.t0 = time.monotonic()
            threading.Thread(target=self._pump, args=(self.p,), daemon=True).start()
            return
        # KHONG decodebin: tren Jetson no chon giai ma phan cung NVMM -> videoconvert "not-negotiated".
        # ponytail: hai che do du phong bo zoom (videocrop can kich thuoc goc ma o day khong biet)
        src = ([f"image/jpeg,width={self.cam_w},height={self.cam_h},framerate={self.fps}/1", "!",
                *self._decode()],
               ["image/jpeg", "!", "jpegdec", "!", "videoscale", "!"],    # MJPG kich thuoc bat ky
               ["video/x-raw", "!", "videoscale", "!"],                   # camera khong co MJPG
               )[self.mode] + ["videoconvert", "!"]
        self.p = subprocess.Popen(
            # io-mode=2 (mmap): de mac dinh thi nvjpegdec treo, khong ra khung nao
            ["gst-launch-1.0", "-q", "v4l2src", f"device={dev or '/dev/null'}",
             *(["io-mode=2"] if HW_JPEG else []), "!", *src,
             f"video/x-raw,format=BGR,width={self.w},height={self.h}", "!",
             "fdsink", "fd=1", "sync=false"],
            # KHONG dat bufsize=0: stdout khi do la FileIO tho, read(n) tra ve dung
            # nhung gi pipe dang co (<=64 KB) chu khong doi du n. Mot khung 720p BGR
            # la 2.7 MB nen se KHONG BAO GIO du -> _pump thoat ngay o khung dau.
            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        self.t0 = time.monotonic()               # moc de tinh han cho KHUNG DAU
        threading.Thread(target=self._pump, args=(self.p,), daemon=True).start()

    def _decode(self):
        """Giai ma MJPG + cat giua theo zoom + scale ve w x h. zoom 1 -> chi scale.

        Jetson: nvv4l2decoder + nvvidconv (phan cung). Do 19/10/2026 webcam 4K: jpegdec +
        videoscale 2K->720p chi 20 fps (kin 1 nhan CPU), phan cung 2K/4K->720p 30 fps, gst 33% CPU.
        """
        l, t, cw, ch = self._box()
        if HW_JPEG:
            # nvvidconv left/right/top/bottom la TOA DO x0,x1,y0,y1 (gst-inspect ghi "pixels to crop"
            # nhung right=0 -> rong 0 -> VIC loi, ra khung rac); videocrop moi la le cat bo
            # nvjpegdec chu khong nvv4l2decoder mjpeg=1: webcam nen YUV 4:2:2, nvv4l2decoder ra sai
            # mau (vang -> tim) + lech ngang ~27 px o 4K
            return ["nvjpegdec", "!", "video/x-raw(memory:NVMM)", "!",
                    "nvvidconv", f"left={l}", f"right={l + cw}",
                    f"top={t}", f"bottom={t + ch}", "!",
                    f"video/x-raw,format=BGRx,width={self.w},height={self.h}", "!"]
        cut = [f"left={l}", f"right={self.cam_w - cw - l}", f"top={t}", f"bottom={self.cam_h - ch - t}"]
        return ["jpegdec", "!", *(["videocrop", *cut, "!"] if self.zoom > 1 else []), "videoscale", "!"]

    def _box(self):
        """Vung cat giua 1/zoom cua khung camera: (x0, y0, rong, cao) theo diem anh goc."""
        cw, ch = round(self.cam_w / self.zoom), round(self.cam_h / self.zoom)
        return (self.cam_w - cw) // 2, (self.cam_h - ch) // 2, cw, ch

    def _send(self, cmd):
        """Lenh cho gst_zoom.py. Che do gst-launch (khong co stdin) / con vua chet: bo qua."""
        try:
            self.p.stdin.write(cmd.encode() + b"\n")
            self.p.stdin.flush()
        except (AttributeError, OSError, ValueError):
            pass

    def set_zoom(self, z):
        """Doi zoom luc dang chay (/fire/zoom). Che do du phong gst-launch thi zoom moi chi co
        hieu luc lan dung lai pipeline sau."""
        z = min(max(1.0, z), 8.0)  # ponytail: tran 8 = 480x270 diem anh 4K, hon nua la phong anh
        if z == self.zoom:
            return
        self.zoom, self.zoom_t = z, time.monotonic()
        self._send("crop %d %d %d %d" % self._box())
        print(f"[follow] zoom x{z:g} -> cat {self._box()}", flush=True)

    def set_gray(self, on):
        if on != self.gray_on:
            self.gray_on = on
            self._send(f"gray {int(on)}")
            self._gray = None

    def _pump_gray(self, proc):
        n = GRAY[0] * GRAY[1]
        with proc.gray as f:
            while True:
                buf = f.read(n)
                if len(buf) < n:
                    return                       # con chet / bi giet: _watch dung lai
                self._gray = np.frombuffer(buf, np.uint8).reshape(GRAY[1], GRAY[0])
                self._gray_new.set()

    def pull_gray(self, timeout):
        """Khung xam toan canh 960x540 MOI (chua tra lan nao) sau set_gray(True), hoac None."""
        if not self._gray_new.wait(timeout):
            return None
        self._gray_new.clear()
        return self._gray

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
            # ~70 ms dau sau khi doi zoom khung con la vung cat CU: khong gui hop lua
            _offer_fire_frame(self.frame, None if self.t - self.zoom_t < 0.2
                              else (time.time(), self._box()))
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
            time.sleep(0.25)  # 23/10: nhip kiem 1 s lam mo lai cham toi 1 s
            # Chua co khung nao thi dem tu luc dung gst va cho `first_grace`; co roi
            # thi dem tu khung cuoi va cho `restart_after`.
            if self.t:
                quiet, han = time.monotonic() - self.t, self.restart_after
            else:
                quiet, han = time.monotonic() - self.t0, self.first_grace
            if not self.dev and resolve_dev(self.want):
                han = 0                          # dang khong co camera ma vua cam vao: mo ngay
            if quiet <= han:
                continue
            if not self.t and self.dev:
                # camera co ma chua ra khung nao: co the khong co MJPG dung w x h@fps -> doi che do
                self.mode = (self.mode + 1) % 3
            print(f"[follow] nguon im {quiet:.0f}s — dung lai gst (che do {self.mode})", flush=True)
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
        # đo thật: MJPG 720p mất ~8 s ra khung đầu (thương lượng USB), đừng hạ first_grace
        # dưới 15 s. Chờ đủ cả 3 chế độ (_watch đổi sau mỗi first_grace), không thì chương
        # trình thoát ở chế độ 0, systemd chạy lại cũng từ 0 -> không bao giờ tới dự phòng.
        t_end = time.monotonic() + 3 * self.first_grace + 5
        while time.monotonic() < t_end:
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
        #
        # CHO khung moi (toi da `stale`): Jetson chay model ~15 ms < 67 ms/khung camera,
        # khong cho thi xu ly + phat lai cung mot khung ~4 lan (do: nguon 15 Hz, nen
        # 59 fps, 3.4 MB/s). Pi cham hon camera nen chua bao gio dinh.
        t_end = time.monotonic() + self.stale
        while self.t == self._t_read and time.monotonic() < t_end:
            time.sleep(0.003)
        self._t_read, self._t_ret = self.t, time.monotonic()
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
        # Khung nguyên, không cắt: hộp lửa ra /fire/detections theo điểm ảnh của chính topic
        # (sim Gazebo 1920x1080). Thiếu meta thì _fire_loop thấy lửa mà không gửi gì.
        h, w = self.frame.shape[:2]
        _offer_fire_frame(self.frame, (time.time(), (0, 0, w, h)))
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
    p.add_argument("--device", default=None,
                   help="vd /dev/video0, hoac auto = camera USB dau tien, tu tim lai khi cam lai")
    p.add_argument("--topic", default="/image_raw", help="dùng khi không có --device")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=15, help="fps xin từ camera")
    p.add_argument("--cam-width", type=int, default=None,
                   help="do phan giai xin camera (mac dinh = --width). Vd 1920 + --zoom 2")
    p.add_argument("--cam-height", type=int, default=None)
    p.add_argument("--zoom", type=float, default=1.0,
                   help="cat giua khung camera, hep FOV de bat nguoi xa hon. Chi voi --device")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--quality", type=int, default=60,
                   help="chat luong JPEG TOI DA; tu ha toi mj.Q_MIN khi mang nghen")
    p.add_argument("--stream-width", type=int, default=0,
                   help="thu nho khung truoc khi phat (0 = giu nguyen). Vd quay 1280, phat 960")
    p.add_argument("--h264", default=None, metavar="HOST:PORT",
                   help="day them H.264 MPEG-TS/UDP toi day (vd IP Tailscale tram mat dat "
                        "100.82.93.76:5600; nhieu tram: cach nhau dau phay). "
                        "MJPEG :8080 van chay song song")
    p.add_argument("--h264-kbps", type=int, default=1500,
                   help="bitrate; co --h264-fb-port thi la TRAN, tu ha khi mat goi")
    p.add_argument("--h264-fb-port", type=int, default=0,
                   help="cong UDP nhan bao cao mat goi tu h264_feedback.py (vd 5601). 0 = tat")
    p.add_argument("--h264-kbps-min", type=int, default=400)
    p.add_argument("--gamma", type=float, default=1.0,
                   help="nang vung toi video H.264 (vd 1.4). 1 = tat. Dung kem --hl-ae")
    p.add_argument("--hl-ae", type=int, default=0, metavar="EXP_MAX",
                   help="tu phoi sang UU TIEN VUNG SANG (den khong loa), tran exposure "
                        "don vi 100 us (vd 100 = 10 ms, chong nhoe). 0 = de camera tu dong")
    p.add_argument("--no-mjpeg", action="store_true",
                   help="tat MJPEG :8080/stream (khoi nen JPEG, khoi ton bang thong). "
                        "/status van chay. Dung kem --h264 khi bay qua 4G")
    p.add_argument("--h264-preset", default="superfast")
    p.add_argument("--h264-ir", action="store_true",
                   help="x264 intra-refresh thay khung IDR moi giay: bitrate deu, mat goi 4G lanh dan")
    p.add_argument("--stamp", action="store_true",
                   help="ve gio chup vao goc duoi trai de do tre (jetson/latency_stamp.py o tram)")
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
    p.add_argument("--fire-save", default=None, metavar="DIR",
                   help="luu khung co hop lua + nhan YOLO vao DIR de train lai (xem _FireSaver)")
    p.add_argument("--fire-verify", default=None, metavar="KEY",
                   help="Gemini xac nhan vet lua truoc khi ra /fire/detections: khoa service account "
                        ".json (Vertex) - xem fire_verify.py")
    p.add_argument("--fire-scan", type=float, default=0.0, metavar="S",
                   help="can --fire-verify: Gemini quet nguyen khung moi S s, so voi YOLO, ghi log + luu khung "
                        "lech vao <--fire-save>/scan; lua YOLO SOT -> /fire/detections (Gemini nam quyen). "
                        "Chi khi drone arm hoac /fire/scan true (~$0.0007/lan)")
    p.add_argument("--fire-picture", action="store_true",
                   help="Gemini tinh ca lua trong ANH IN / man hinh (thu to A0). Mac dinh chi lua that. "
                        "Khong sua duoc dong lenh service thi: touch ~/jetson/fire_picture (rm de tat)")
    a = p.parse_args()

    import person_follow_reid as pf
    from fire_tracker import _hud

    _init_count_pub()

    # Cat giua -> goc nhin hep lai; lenh bam (ang_x, fx) tinh tu HFOV nen phai doi theo.
    if a.zoom > 1:
        pf.HFOV_DEG = float(np.degrees(2 * np.arctan(np.tan(np.radians(pf.HFOV_DEG) / 2) / a.zoom)))
        print(f"[follow] zoom {a.zoom:g} -> HFOV {pf.HFOV_DEG:.1f} deg", flush=True)

    global _fire_tracker, _fire_period, _fire_ros, _fire_saver, _fire_verify, _fire_scan
    if a.fire_model:
        from fire_tracker import FireTracker

        _fire_period = 1.0 / a.fire_hz if a.fire_hz > 0 else 0.0
        _fire_tracker = FireTracker(a.fire_model, conf=a.fire_conf,
                                    threads=a.fire_threads,
                                    confirm_hits=1, max_miss=2)
        if a.fire_save:
            _fire_saver = _FireSaver(a.fire_save, a.cam_width or a.width, _fire_tracker.names)
        if a.fire_verify:
            try:  # khoa hong/thieu: chi YOLO, van stream - drone khong duoc mu vi Gemini
                from fire_verify import FireVerify
                _fire_verify = FireVerify(a.fire_verify,
                                          scan_dir=os.path.join(a.fire_save, "scan") if a.fire_save else None)
                _fire_tracker.tag = _fire_verify.tag
                _fire_scan = a.fire_scan
                _fire_verify.on_fire, _fire_verify.picture = _gemini_fire, _PictureFlag(a.fire_picture)
                print(f"[ai] anh lua in: {'BAT' if _fire_verify.picture else 'tat'} (--fire-picture / {PICTURE_FLAG})",
                      flush=True)
                if _fire_scan:
                    print(f"[scan] Gemini quet nguyen khung moi {_fire_scan:g} s khi drone ARM hoac /fire/scan true",
                          flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"[ai] tat Gemini ({type(e).__name__}: {e}) -> chi YOLO", flush=True)
        # KHONG gan mj._tracker: lua van ve hop len video, nhung /status + ROS chi bao nguoi.
        threading.Thread(target=_fire_loop, args=(_fire_tracker,), daemon=True).start()
        lop = ([_fire_tracker.names[i] for i in sorted(_fire_tracker.keep_cls)]
               if _fire_tracker.keep_cls is not None
               else list(_fire_tracker.names.values()))
        print(f"[fire] model {a.fire_model} conf={a.fire_conf} "
              f"threads={a.fire_threads} hz={a.fire_hz:g} lop={lop}", flush=True)

    cap_ref = []                                 # `show` can hoi tuoi khung cua chinh no
    if _fire_tracker is not None and _pub_count is not None:  # _pub_count = rclpy da init duoc
        try:
            _fire_ros = _FireRos(cap_ref)
        except Exception as e:  # noqa: BLE001
            print(f"[ROS] khong tao duoc node lua ({e}) -> chi ve hop len video", flush=True)

    def _make_cap(*_, **__):
        c = (GstCap(a.device, a.width, a.height, a.fps, cam_w=a.cam_width,
                    cam_h=a.cam_height, zoom=a.zoom) if a.device else RosCap(a.topic))
        cap_ref.append(c)
        return c

    cv2.VideoCapture = _make_cap
    if a.hl_ae and a.device:
        HighlightAE(a.device, cap_ref, a.hl_ae)
    elif a.device:                               # lan chay truoc co the de camera o thu cong
        # Tat bu nguoc sang: no keo vung toi len nhung day den/troi bi chay them.
        dev = resolve_dev(a.device)
        if dev:
            v4l2_set(dev, auto_exposure=3, gain=0, backlight_compensation=0)
    mj._mjpeg_on = not a.no_mjpeg
    h264 = (H264Out(a.h264, a.h264_kbps, a.h264_preset, a.h264_fb_port, a.h264_kbps_min,
                    a.gamma, a.h264_ir) if a.h264 else None)
    mj._q = mj._q_max = a.quality                # tran chat luong; mang nghen thi Handler ha xuong

    # Chi phat khung pf DA VE XONG: hop/xuong khop 100% voi hinh, video di theo nhip
    # model (~9 fps voi model 320). Da thu tach hinh ra 15 fps va dan lop phu len khung
    # moi hon (2026-09-15): lop phu tre 160-290 ms, lech ro khi vung tay/di nhanh; bu
    # bang optical flow (9.8 vs 10.1 px) hay ngoai suy van toc (18.2 vs 16.5 px) deu
    # KHONG hon khong bu — o cu ly gan hop doi hinh dang (tay, chan) chu khong truot
    # deu. Muon hinh nhanh hon ma van khop thi phai lam model nhanh hon.
    n_frame = [0]
    fps = [0.0, time.monotonic()]                # [fps EMA, moc khung truoc] - fps VIDEO that
    seg_t = [time.monotonic() + 5]               # --stamp: lan in [doan] ke tiep

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
        mj._persons = n = pf.STATS["persons"]    # cho /status
        if _pub_count is not None:
            pub, Int32 = _pub_count
            pub.publish(Int32(data=n))           # moi khung phat ~15 Hz: 4 byte, khong dang tiec
        now = time.monotonic()
        dt, fps[1] = now - fps[1], now
        if dt > 0:
            fps[0] = 1 / dt if fps[0] == 0 else 0.9 * fps[0] + 0.1 / dt
        hud = f"NGUOI {n} | {fps[0]:.0f} FPS"
        if getattr(cap, "zoom", 1.0) > 1:
            hud += f" | x{cap.zoom:g}"           # nguoi van hanh biet khung dang zoom
        frame = np.ascontiguousarray(frame)
        if _fire_tracker is not None:
            _fire_tracker.draw_cached(frame, hud=hud)
        else:
            _hud(frame, hud)
        h, w = frame.shape[:2]
        if a.stream_width and w > a.stream_width:
            # Model can 1280 de bat nguoi xa; GUI khong can - nho lai cho nhe WiFi.
            frame = cv2.resize(frame, (a.stream_width, round(h * a.stream_width / w)),
                               interpolation=cv2.INTER_AREA)
        t_push = None
        if a.stamp:
            # gio chup = luc gst tra khung (GstCap._t_read); RosCap khong co -> luc phat
            t_cap = getattr(cap, "_t_read", 0.0) or now
            latency_stamp.stamp(frame, time.time() - (time.monotonic() - t_cap))
            t_push, t_ret = time.monotonic(), getattr(cap, "_t_ret", now)
            _seg_add("cho", t_ret - t_cap)       # khung nam doi vong nhan dien lay
            _seg_add("xu_ly", now - t_ret)       # yolo + tracker + ve hop
            _seg_add("ve", t_push - now)         # hop lua + HUD + dau thoi gian
            if t_push >= seg_t[0]:
                seg_t[0] = t_push + 5
                _seg_report()
        if h264 is not None:
            h264.push(frame, t_push)
        if a.no_mjpeg:
            with mj._lock:
                mj._out += 1                     # dong [mjpeg] "nen" = khung dua sang H.264
            return
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, mj._q])
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
    except Exception:  # noqa: BLE001
        # Loi o vong chinh ma de Python thoat binh thuong thi treo o "terminate called
        # without an active exception" (luong onnxruntime/rclpy chua dung): tien trinh
        # con song, systemd khong restart, GUI mat hinh mai. Thoat CUNG de systemd lo.
        import traceback  # os da import o dau file: import lai o day lam os thanh bien cuc bo cua main
        traceback.print_exc()
        sys.stdout.flush()
        os._exit(1)


if __name__ == "__main__":
    main()
