#!/usr/bin/env python3
"""May chu MJPEG cho camera companion. CHAY TREN PI, hoac tren may chay mo phong.

    python3 mjpeg_server.py                          # /dev/video0, cong 8080
    python3 mjpeg_server.py --device /dev/video2 --every 2
    python3 mjpeg_server.py --ros-topic /rgb/image   # NGUON ROS2 (mo phong)
    python3 mjpeg_server.py --ros-topic /rgb/image \
        --detect ~/firedrop_sim/tracking/best_nano_111_640.onnx   # + bam vet lua

Trong Docker phai cho container thay camera va mo cong:

    docker run --device=/dev/video0 -p 8080:8080 ...

Vi sao mot cong HTTP RIENG chu khong nhet chung WebSocket 8765: mot khung hinh
40 KB xep hang truoc mot lenh se lam tre duong lenh ROS2 (nguyen tac 2.1). Video
la thu dep-thi-co, no khong duoc chen vao hang doi lenh. Cong nay chet thi mat
video, het — nut do di duong SiK, khong dinh dang gi toi day.

MOT luong doc camera duy nhat, N client cung xem khung moi nhat. Khong xep hang
khung cu: GUI cham hon camera thi do tre troi dan thanh vai giay: tha khung con
hon tra khung qua date.

--------------------------------------------------------------------------
HAI NGUON, MOT DUONG RA
--------------------------------------------------------------------------
`grab()`     doc /dev/videoN bang OpenCV        — camera that tren Pi
`grab_ros()` doc mot topic sensor_msgs/Image    — camera trong Gazebo

Ca hai chi lam DUNG mot viec: dat JPEG moi nhat vao `_latest` va tang `_seq`.
Tu do tro xuong (Handler, cong 8080, dinh dang multipart) KHONG doi mot dong,
va GUI khong sua mot ky tu nao — no van chi biet `http://<host>:8080/stream`.
Do la ly do ranh gioi dat o HTTP chu khong dat o DDS: doi nguon la doi mot ham,
khong phai doi giao thuc.

Dung voi ~/firedrop_sim (mo phong tha bong chua chay):

    source /opt/ros/humble/setup.bash
    python3 ~/GUI_NATIVE/tools/mjpeg_server.py --ros-topic /rgb/image

`/rgb/image` la camera RGB nadir cua drone, dung khung hinh ma `fire_detector`
doc — nen cai thay tren GUI CHINH LA cai thuat toan dang nhin, khong phai mot
goc camera khac cho dep.

`--detect` cam them YOLO ONNX vao dung cho nay va ve hop bam vet len khung hinh
TRUOC khi nen JPEG (xem fire_tracker.py). GUI khong sua mot dong: no van chi
biet `http://<host>:8080/stream`, chi la trong khung hinh nay co san hop.
⚠ Hop nay la de NGUOI XEM. Quyet dinh tha bong van do `fire_detector` ben
firedrop_sim quyet, va no dung nguong HSV do duoc chu khong dung model nay —
xem "Nguyen tac bat di bat dich" trong README cua firedrop_sim.

⚠ KHONG dung cv_bridge o day. No keo theo ca ROS perception stack va tren nhieu
may cai dat ROS2 binary thi `import cv_bridge` gay ABI mismatch voi OpenCV cua
pip. Doi imgmsg -> numpy chi la mot phep reshape; viet tay 15 dong con hon them
mot phu thuoc nang co the hong theo cach kho chan doan.
"""

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

BOUNDARY = "frame"

_lock = threading.Lock()
_latest = None  # JPEG byte moi nhat
_seq = 0  # tang moi khung — de client biet co khung moi chua
_in = _out = _bytes = 0  # dem cho dong thong ke, xem _stats_loop()
_raw = None  # khung BGR moi nhat cho luong YOLO — xem _yolo_loop()


def _luu(jpg):
    """Dat JPEG moi nhat vao cho chung. CA HAI nguon deu di qua day.

    Gop lai mot ham vi cai dem thong ke phai dem DU: de moi nguon tu tang _seq
    thi them nguon thu ba la lai quen dem, va so lieu sai im lang con te hon
    khong co so lieu.
    """
    global _latest, _seq, _out, _bytes
    with _lock:
        _latest = jpg
        _seq += 1
        _out += 1
        _bytes += len(jpg)


def _stats_loop(period, tracker=None):
    """In nhip that moi `period` giay: nguon dua vao bao nhieu, minh nen ra bao nhieu.

    Hai con so PHAI tach nhau. Gop lam mot thi tut fps khong noi duoc la camera
    chet hay minh nghen: nguon 30 Hz ma ra 8 fps la loi ben nay; nguon 8 Hz thi
    loi o camera/USB, sua ben nay vo ich.
    """
    global _in, _out, _bytes
    while True:
        time.sleep(period)
        with _lock:
            i, o, b = _in, _out, _bytes
            _in = _out = _bytes = 0
        yolo = f", yolo {tracker.ms:.0f} ms" if tracker is not None else ""
        print(f"[mjpeg] nguon {i / period:5.1f} Hz  ->  nen {o / period:5.1f} fps, "
              f"{b / period / 1024:6.0f} KB/s{yolo}", flush=True)


def _yolo_loop(tracker):
    """Chay model tren khung MOI NHAT, mai mai, o nhip rieng cua no.

    Vi sao mot luong rieng: do tren Pi5, yolov8n-640 mat ~410 ms mot khung, con
    camera ra khung moi ~67 ms. Suy dien ngay trong duong nen JPEG thi CA LUONG
    VIDEO tut xuong 2,4 fps — nguoi lai mat hinh de doi cai hop. Tach ra thi
    video giu nguyen nhip, hop cap nhat cham hon va tre toi ~410 ms.

    KHONG xep hang khung: lay xong thi xoa cho. YOLO cham hon camera nen hang
    doi se chi day len mai, va hop ve theo khung tu 5 giay truoc con te hon hop
    cu 0,4 giay.
    """
    global _raw
    while True:
        with _lock:
            img, _raw = _raw, None
        if img is None:
            time.sleep(0.005)
            continue
        try:
            tracker.infer(img)
        except Exception as e:  # noqa: BLE001
            # Model hong hay anh la kich thuoc: keu roi chay tiep. Chet luong nay
            # thi hop dung im mai ma video van chay — khong ai biet vi sao.
            print(f"[track] loi suy dien: {type(e).__name__}: {e}", flush=True)
            time.sleep(1.0)


def _ve_hop(tracker, img):
    """Ve hop bam vet len anh TRUOC khi nen JPEG. Khong co tracker thi tra nguyen.

    `ascontiguousarray` khong phai cho nhanh: anh tu ROS la view cua
    `np.frombuffer` nen CHI DOC, va rgb8 con la lat buoc `[:, :, ::-1]`. Ve
    thang len do thi cv2 nem "Layout of the output array is incompatible" -
    ngay khung dau tien, va chi khi doc ROS chu khong khi doc USB.
    """
    if tracker is None:
        return img
    img = np.ascontiguousarray(img)
    global _raw
    with _lock:
        if _raw is None:  # luong YOLO chua lay khung truoc — dung de no lai
            _raw = img.copy()
    tracker.draw_cached(img)
    return img


def grab(device, width, height, every, quality, tracker=None):
    """Doc camera mai mai. Camera rut ra hay loi thi cho roi mo lai, khong chet."""
    global _latest, _seq
    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    cap = None
    idx = 0
    while True:
        if cap is None or not cap.isOpened():
            if cap is not None:
                cap.release()
            cap = cv2.VideoCapture(device)
            # MJPG: webcam tu nen JPEG, USB2 do nghen hon nhieu so voi YUYV tho.
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            # Khong co dong nay thi driver dem san vai khung, va do tre bat dau
            # troi: cai ban thay la qua khu chu khong phai hien tai.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                print(f"[mjpeg] khong mo duoc {device}, thu lai sau 2s", flush=True)
                time.sleep(2.0)
                continue
            # KHONG dat CAP_PROP_FPS. Do tren Rapoo/Pi5: de mac dinh duoc 16,0 fps
            # voi p50=67ms, max=68ms — deu den muc gan nhu khong lech. Ep xuong 10
            # thi driver tra 7,0 fps va p90 vot len 200ms. Driver tu chay la tot
            # nhat, dung day no.
            print(f"[mjpeg] mo {device} {width}x{height}, gui 1/{every} khung", flush=True)

        ok, frame = cap.read()
        if not ok:
            print("[mjpeg] mat khung, mo lai camera", flush=True)
            cap.release()
            cap = None
            continue

        # Thua nhip theo SO DEM khung, khong theo dong ho. Day la cho da sai mot
        # lan va phai ghi lai: thua nhip bang `time.monotonic()` voi chu ky 100 ms
        # trong khi camera tra khung moi 67 ms thi hai nhip danh nhau sinh phach —
        # khung den som 1 ms bi vut di roi phai doi tron mot chu ky nua. Do duoc:
        # 9,1 fps tut xuong 2,5 fps, p90 250ms vot len 939ms, dinh 4,5 GIAY, du de
        # ben GUI lat sang o xam ba lan trong 25 giay.
        #
        # Dem khung thi khoang cach luon la boi so cua chu ky camera, khong the co
        # phach. `cap.read()` chan theo driver nen vong nay cung khong quay tit.
        global _in
        _in += 1
        idx += 1
        if idx % every:
            continue

        ok, jpg = cv2.imencode(".jpg", _ve_hop(tracker, frame), params)
        if ok:
            _luu(jpg.tobytes())


def _imgmsg_to_bgr(msg):
    """sensor_msgs/Image -> anh BGR cho cv2.imencode. None neu khong hieu encoding.

    Chi nhan bon encoding thuc su gap: rgb8/bgr8 (camera mau), mono8 (hong
    ngoai), 32FC1 (depth, met). Gap thu khac thi TRA VE None chu khong doan —
    doan sai encoding cho ra mot khung hinh xanh le nhin nhu camera hong, va
    ban se di tim loi o day camera thay vi o dong nay.
    """
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    enc = msg.encoding.lower()
    if enc in ("rgb8", "bgr8"):
        img = buf.reshape(msg.height, msg.step)[:, : msg.width * 3].reshape(msg.height, msg.width, 3)
        return img[:, :, ::-1] if enc == "rgb8" else img
    if enc == "mono8":
        img = buf.reshape(msg.height, msg.step)[:, : msg.width]
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if enc in ("32fc1", "32fc"):
        # Depth tinh bang MET. Trai ra 0..10 m thanh thang xam de nhin duoc;
        # day la anh DE NGUOI XEM, khong ai do khoang cach tu no.
        d = buf.view(np.float32).reshape(msg.height, msg.step // 4)[:, : msg.width]
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
        g = np.clip(d / 10.0 * 255.0, 0, 255).astype(np.uint8)
        return cv2.applyColorMap(g, cv2.COLORMAP_TURBO)
    return None


def grab_ros(topic, every, quality, tracker=None):
    """Doc mot topic sensor_msgs/Image mai mai. Cung dau ra voi `grab()`.

    Chay trong thread rieng nhu `grab()`, nhung rclpy.spin() phai o dung thread
    da tao node — nen node duoc tao NGAY TRONG day chu khong o main().

    QoS: BEST_EFFORT + depth 1. Anh camera trong ROS2 gan nhu luon publish o
    BEST_EFFORT; subscriber de RELIABLE (mac dinh) thi KHONG BAO GIO KHOP va
    khong co mot dong loi nao — chi la khong bao gio co khung hinh. Day la cai
    bay pho bien nhat khi doc camera bang rclpy.
    """
    import rclpy
    from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
    from sensor_msgs.msg import Image

    params = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    state = {"idx": 0, "warned": False}

    def on_image(msg):
        global _in
        _in += 1
        state["idx"] += 1
        if state["idx"] % every:
            return
        img = _imgmsg_to_bgr(msg)
        if img is None:
            if not state["warned"]:
                print(f"[mjpeg] khong hieu encoding '{msg.encoding}' - bo qua", flush=True)
                state["warned"] = True
            return
        ok, jpg = cv2.imencode(".jpg", _ve_hop(tracker, img), params)
        if ok:
            _luu(jpg.tobytes())  # _luu tu khoa lay — `_lock` KHONG tai nhap duoc

    rclpy.init(args=None)
    node = rclpy.create_node("mjpeg_server")
    qos = QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                     history=QoSHistoryPolicy.KEEP_LAST)
    node.create_subscription(Image, topic, on_image, qos)
    print(f"[mjpeg] doc ROS2 topic {topic}, gui 1/{every} khung", flush=True)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.0"

    def do_GET(self):
        if self.path not in ("/stream", "/"):
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()

        last = -1
        try:
            while True:
                with _lock:
                    jpg, seq = _latest, _seq
                if jpg is None or seq == last:
                    time.sleep(0.005)  # chua co khung moi — dung gui lai khung cu
                    continue
                last = seq
                self.wfile.write(
                    f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(jpg)}\r\n\r\n".encode()
                )
                self.wfile.write(jpg)
                self.wfile.write(b"\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass  # GUI dong tab hay rot WiFi — chuyen binh thuong, khong phai loi

    def log_message(self, fmt, *args):
        pass  # mot ket noi keo dai ca chuyen bay, khong co gi de log moi dong


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--device", default="/dev/video0")
    p.add_argument("--port", type=int, default=8080)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--every", type=int, default=1,
                   help="gui 1 trong N khung (1 = moi khung). Rapoo/Pi5 native 16 fps, "
                        "nen --every 2 ra 8 fps. Thua nhip theo dem, khong theo dong ho.")
    p.add_argument("--quality", type=int, default=70, help="chat luong JPEG 1-100")
    p.add_argument("--ros-topic", default=None,
                   help="doc sensor_msgs/Image tu topic nay thay vi /dev/videoN. "
                        "Vd: --ros-topic /rgb/image cho ~/firedrop_sim. "
                        "Phai `source /opt/ros/humble/setup.bash` truoc.")
    p.add_argument("--detect", default=None, metavar="MODEL.onnx",
                   help="bam vet lua/khoi bang YOLO ONNX va ve hop len khung hinh. "
                        "Vd: --detect ~/firedrop_sim/tracking/best_nano_111_640.onnx")
    p.add_argument("--conf", type=float, default=0.35,
                   help="nguong diem cua --detect (mac dinh 0.35)")
    p.add_argument("--stats", type=float, default=5.0,
                   help="in nhip khung moi N giay (0 = tat)")
    a = p.parse_args()

    # Nap model TRUOC khi mo cong: file hong hay thieu onnxruntime thi phai chet
    # ngay o day. Nap trong thread doc camera thi cong 8080 van mo, GUI van thay
    # video - chi la khong bao gio co hop nao, va khong co gi noi vi sao.
    tracker = None
    if a.detect:
        from fire_tracker import FireTracker

        # confirm_hits=1, max_miss=2: o nhip ~2,4 fps cua luong YOLO, day la
        # "hien ngay khi thay, xoa sau ~0,8 giay khong thay". Giu mac dinh (2/5)
        # thi phai 0,8 giay hop moi hien va 2 giay sau khi vat di hop van con.
        tracker = FireTracker(a.detect, conf=a.conf, threads=2,
                              confirm_hits=1, max_miss=2)
        print(f"[mjpeg] bam vet bang {a.detect} - lop {list(tracker.names.values())}",
              flush=True)

    if a.ros_topic:
        target, args = grab_ros, (a.ros_topic, max(1, a.every), a.quality, tracker)
    else:
        target, args = grab, (a.device, a.width, a.height, max(1, a.every), a.quality,
                              tracker)
    threading.Thread(target=target, args=args, daemon=True).start()

    if tracker is not None:
        threading.Thread(target=_yolo_loop, args=(tracker,), daemon=True).start()
    if a.stats > 0:
        threading.Thread(target=_stats_loop, args=(a.stats, tracker), daemon=True).start()

    srv = ThreadingHTTPServer(("0.0.0.0", a.port), Handler)
    print(f"[mjpeg] http://0.0.0.0:{a.port}/stream", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
