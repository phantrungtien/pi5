#!/usr/bin/env python3
"""
person_follow_reid.py  --  bản PI 5 (ONNX, không torch/ultralytics)
---------------------------------------------------------------
Bám 1 người bằng MỘT model duy nhất:
  yolo11n-pose.onnx  -> box + 17 keypoint COCO trong 1 lần chạy
  IoUTracker (track.py)      -> ID giữa các frame  (thay BoT-SORT)
  osnet_x0_25.onnx           -> ReID gallery, khóa/tìm lại đúng người (thay torchreid)

Bản gốc chạy 2 nhánh YOLO (detect + pose) rồi ghép bằng IoU; model pose đã cho
cả box lẫn keypoint nên nhánh detect bị bỏ -> nhẹ gấp đôi, không cần ghép.

Lấy tâm người (ưu tiên giảm dần, mượt nhất cho điều khiển):
  1) trung bình 2 vai (5,6) + 2 hông (11,12)   [torso - ổn định nhất]
  2) trung điểm 2 vai                           [khi thiếu hông]
  3) tâm bbox                                    [khi không có pose]

    python3 person_follow_reid.py --source 0
    python3 person_follow_reid.py --selftest      # không cần camera
"""

import argparse
import threading
import time
from collections import deque

import cv2
import numpy as np
import onnxruntime as ort

from track import Detector, IoUTracker

# ===== ROS2: publish lệnh sang node điều khiển (follow_control.py) =====
# 2 topic: /follow/command (String)  +  /follow/target (Vector3: ang_x, ang_y, z)
# QoS mặc định (RELIABLE, depth 10) -> KHỚP với subscriber bên follow_control.
ROS_OK = False
STATS = {"persons": 0}  # so nguoi khung gan nhat — run_gui doc de ve HUD + /status
_node = _pub_cmd = _pub_tgt = None
try:
    import rclpy
    from std_msgs.msg import String
    from geometry_msgs.msg import Vector3
    ROS_OK = True
except Exception as _e:
    print(f"[ROS] rclpy không sẵn ({_e}) -> chỉ in, KHÔNG publish. "
          f"Hãy chạy trong terminal đã 'source /opt/ros/<distro>/setup.bash'.")


def init_ros():
    """Khởi tạo node + 2 publisher. Gọi 1 lần trong main()."""
    global _node, _pub_cmd, _pub_tgt
    if not ROS_OK:
        return
    rclpy.init()
    _node = rclpy.create_node("vision_follow")
    _pub_cmd = _node.create_publisher(String, "/follow/command", 10)
    _pub_tgt = _node.create_publisher(Vector3, "/follow/target", 10)
    print("[ROS] vision_follow node OK -> publishing /follow/command, /follow/target")


def shutdown_ros():
    if ROS_OK and _node is not None:
        _node.destroy_node()
        rclpy.shutdown()


# ===== CẤU HÌNH =====
# imgsz 416 chứ không 640: đo trên Pi 5 là 97 ms vs 293 ms (3.0x) cho cùng model.
# `Detector` đọc imgsz từ chính file onnx nên đổi model = đổi đúng dòng này.
# Còn bản 320 (53 ms) trong thư mục, nhanh gấp 5.5x nhưng bỏ sót người ở xa.
POSE_ONNX = "yolo11n-pose-416.onnx"   # box + 17 keypoint, 1 lần chạy
REID_ONNX = "osnet_x0_25.onnx"    # ReID 512-D
CONF      = 0.4
CAM_INDEX = 0                     # `v4l2-ctl --list-devices` để kiểm tra
HFOV_DEG  = 70.0                  # góc nhìn NGANG của camera (tra spec/calib). Webcam ~60-70

# ===== ƯỚC LƯỢNG KHOẢNG CÁCH (mono + chiều cao người ~1.70m) =====
ADULT_HEIGHT_M = 1.70   # giả định người trưởng thành
# Các đoạn THẲNG ĐỨNG trên cơ thể + tỷ lệ chiều dài thật so với chiều cao H
# (nhân trắc học). Dùng đoạn dọc vì khi người xoay trái/phải (FRONT/BACK/side)
# chiều dài chiếu theo trục dọc ảnh gần như không đổi -> sai số thấp nhất.
# Ưu tiên đoạn DÀI nhất còn nhìn thấy (đoạn càng dài, sai số pixel càng nhỏ).
#   (tên, keypoint trên, keypoint dưới, tỷ_lệ/H)
DIST_SEGMENTS = [
    ("full",  [0],      [15, 16], 0.87),   # mũi -> cổ chân  (~1.48m)
    ("legs+", [5, 6],   [15, 16], 0.78),   # vai  -> cổ chân (~1.33m)
    ("torso", [5, 6],   [11, 12], 0.29),   # vai  -> hông    (~0.49m, khi chân bị che)
    ("thigh", [11, 12], [13, 14], 0.245),  # hông -> gối     (~0.42m)
    ("head",  [1, 2, 3, 4], [5, 6], 0.12), # mắt/tai -> vai  (~0.20m, TẦM GẦN: chỉ thấy đầu+vai)
]
MIN_SEG_PX = 20          # đoạn < 20px -> bỏ (sai số lượng tử hóa quá lớn)
DIST_RANGE = (0.3, 15.0) # mét, ngoài khoảng này coi là outlier

# ===== BÙ NGHIÊNG CHO CAMERA GẮN CỨNG (strapdown, ArduPilot) =====
# Camera gắn cứng -> khi drone nghiêng, trục quang không còn ngang:
#   - roll : ảnh xoay -> đo đoạn theo ĐỘ DÀI EUCLID (bất biến xoay) thay vì Δy.
#   - pitch: đoạn dọc bị co ~cos(pitch) -> chia lại để khôi phục.
# Attitude lấy từ MAVLink ATTITUDE (xem class AttitudeSource).
MAV_CONNECT = None       # vd "udp:127.0.0.1:14550" (SITL/MAVProxy) hoặc "/dev/serial0"
MAV_BAUD    = 57600
PITCH_CLAMP_COS = 0.34   # ~70deg: chặn chia cho số quá nhỏ khi pitch lớn

# ===== LỆNH ĐIỀU KHIỂN CẤP CAO (vision chỉ xuất command, control tự suy setpoint) =====
STANDOFF_MIN_M = 1.5     # Z < ngưỡng này -> quá gần -> BACK_OFF
# FOLLOW : thấy thân, bám bình thường (control giữ standoff theo z + yaw theo ang_x)
# BACK_OFF: quá gần / chỉ thấy đầu -> control lùi ra, tạm giữ yaw
# SEARCH : từng khóa mục tiêu, đang mất -> control hold + quét, KHÔNG tiến
# IDLE   : chưa khóa mục tiêu
CMD_COLORS = {"FOLLOW": (0, 255, 0), "BACK_OFF": (0, 165, 255),
              "SEARCH": (0, 0, 255), "IDLE": (180, 180, 180)}

# COCO-17: 0 nose,1 l_eye,2 r_eye,5 l_sho,6 r_sho,11 l_hip,12 r_hip ...
SKELETON = [(5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12),
            (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
            (0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6)]
FACE_IDX, SHO_IDX, HIP_IDX = [0, 1, 2], [5, 6], [11, 12]
KNEE_IDX, ANKLE_IDX = [13, 14], [15, 16]

# ===== LỚP PHÂN LOẠI TẦM NHÌN (drone đang thấy CẢ NGƯỜI / THÂN / ĐẦU) =====
# Mục đích: tâm bám phải là MỘT ĐIỂM CỐ ĐỊNH TRÊN CƠ THỂ (tâm thân) ở mọi tầm.
# Trước đây khi chỉ thấy đầu thì rơi về tâm bbox = giữa mặt -> lúc lùi ra thấy
# chân, tâm tụt xuống thân đột ngột -> yaw/pitch giật. Nay tầm gần thì SUY RA
# tâm thân từ đầu+vai theo nhân trắc học, nên tâm liên tục qua mọi tầm nhìn.
#   FULL : thấy hông + gối/cổ chân   -> tâm thân đo trực tiếp
#   BODY : thấy vai + hông           -> tâm thân đo trực tiếp
#   HEAD : chỉ đầu/vai (quá gần)     -> tâm thân NGOẠI SUY
#   NONE : không có keypoint         -> tâm bbox
# BẤT BIẾN: tâm bám LUÔN nằm trong bbox người. Không bao giờ ngắm ra chỗ camera
# không nhìn thấy -- bản trước ngoại suy tâm thân theo nhân trắc học nên khi chỉ
# thấy đầu+vai, tâm rơi ra NGOÀI khung hình (đo thật: y=990 trong khung cao 720,
# 100% số frame) -> drone chúc xuống chỗ trống và mất mục tiêu.
BBOX_INSET = 0.05        # kẹp tâm vào trong bbox, chừa mép 5%

# ===== CHỐNG GIẬT TÂM =====
# Không dùng NGƯỠNG CỨNG để chọn tâm. Keypoint hông nhấp nháy quanh ngưỡng là
# tâm nhảy giữa "mặt" và "thân". Thay bằng 2 lớp liên tục:
#  1) HÒA TRỘN: tâm = w*(đo từ vai+hông) + (1-w)*(ngoại suy từ vai), w chạy
#     mượt theo độ tin cậy hông -> hông hiện dần thì tâm TRƯỢT chứ không nhảy.
#  2) EMA theo thời gian: lọc nốt nhiễu từng frame.
HIP_BLEND  = (0.25, 0.75)  # conf hông: <=0.25 chỉ dùng phần thấy được, >=0.75 dùng tâm thân đo
MIN_SHO_W  = 0.30          # vai mờ hơn mức này -> bỏ pose, quay về tâm bbox
CENTER_OFFSET_EMA = 0.9    # lọc ĐỘ LỆCH neo->tâm (KHÔNG lọc cả tâm, xem CenterFilter)
CENTER_JUMP_RESET = 0.25   # tâm nhảy > 25% bề ngang khung -> coi là đổi người, xóa lọc
VIEW_STICKY_N = 3          # view phải lặp N frame mới được đổi (hết nhấp nháy nhãn/lệnh)

# ===== NGƯỠNG THỜI GIAN: KHAI BÁO BẰNG GIÂY, KHÔNG BẰNG FRAME =====
# Mọi ngưỡng dưới đây từng đếm bằng FRAME và được chỉnh trên laptop 22 fps. Pi 5
# chạy 3.6 fps nên CÙNG con số đó dài ra 6 LẦN tính theo giây: max_lost=90 frame
# là 4.1 s trên laptop nhưng 25 s trên Pi — drone SEARCH suốt 25 giây mới chịu nhả
# mục tiêu. Cùng lỗi ở max_age (0.7 s -> 4.2 s) và median Z (0.3 s -> 1.9 s).
# Máy nhanh hay chậm thì THỜI GIAN mới là thứ phải giữ nguyên, không phải số frame,
# nên khai báo bằng giây rồi quy về frame theo nhịp ĐO ĐƯỢC lúc chạy (_tune_for_fps).
LOST_S        = 4.0    # mất mục tiêu quá ngần này -> bỏ khoá hẳn
TRACK_AGE_S   = 0.7    # track không khớp detection quá ngần này -> xoá
NEW_TRACK_S   = 0.10   # detection phải sống ngần này mới được coi là track thật
VIEW_STICKY_S = 0.15   # view phải lặp ngần này mới được đổi
DIST_MED_S    = 0.35   # bề rộng cửa sổ median lọc Z
CENTER_TAU_S  = 0.43   # hằng số thời gian EMA độ lệch tâm (đúng bằng EMA 0.9 @ 22 fps)
FPS_BOOT      = 5.0    # nhịp giả định cho frame đầu, trước khi đo được


def _tune_for_fps(fps, ids_tracker, tracker, view_sticky, center_filter, dist_hist):
    """Quy các ngưỡng theo GIÂY ở trên về số frame tại nhịp đang chạy.

    Gọi MỖI FRAME chứ không một lần lúc khởi động: nhịp thay đổi thật trong lúc bay
    (thêm người trong khung -> ReID chạy nhiều crop hơn -> infer chậm đi), và ngưỡng
    tính theo nhịp cũ thì sai đúng vào lúc cảnh đông người, tức lúc cần nhất.
    Trả về dist_hist vì deque KHÔNG đổi được maxlen tại chỗ.
    """
    n = lambda sec: max(1, round(sec * fps))
    ids_tracker.max_age = n(TRACK_AGE_S)
    ids_tracker.min_hits = n(NEW_TRACK_S)
    tracker.max_lost = n(LOST_S)
    # Sàn 2: n=1 nghĩa là đổi view ngay lần đầu, tức Sticky không còn chống nhấp
    # nháy nữa. Bộ lọc mà mất tác dụng thì thà chịu thêm 1 frame trễ.
    view_sticky.n = max(2, n(VIEW_STICKY_S))
    # EMA rời rạc -> liên tục: a = e^(-dt/tau). Giữ nguyên hằng số thời gian thật
    # thay vì giữ nguyên hệ số a, vì a=0.9 ở 3.6 fps là tau 2.7 s (trễ thấy rõ).
    center_filter.a = float(np.exp(-1.0 / max(fps * CENTER_TAU_S, 1e-6)))
    # Sàn 3: median lọc OUTLIER, cần ít nhất 3 mẫu mới bỏ được 1 giá trị bậy.
    # n(0.35 s) ở 3.6 fps ra 1 -> hết lọc, đúng lúc keypoint nhiễu nhất.
    m = max(3, n(DIST_MED_S))
    return dist_hist if dist_hist.maxlen == m else deque(dist_hist, maxlen=m)


class Sticky:
    """Giữ trạng thái, chỉ đổi khi giá trị mới lặp lại n frame liên tiếp."""

    def __init__(self, init, n=3):
        self.v, self.n, self._cand, self._c = init, n, init, 0

    def __call__(self, v):
        if v == self.v:
            self._c = 0
        else:
            self._c = self._c + 1 if v == self._cand else 1
            self._cand = v
            if self._c >= self.n:
                self.v, self._c = v, 0
        return self.v


def center_anchor(kp_xy, kp_conf, bbox):
    """Điểm NEO tin cậy nhất trên người -> (điểm, loại).

    Đo trên 150 frame thật ở tầm gần: nhiễu tần số cao của mặt ~1.1px, vai ~8.9px.
    Nên luôn ưu tiên mặt; vai chỉ dùng khi người quay lưng; bí quá mới lấy bbox.
    """
    f, wf = _wmean(kp_xy, kp_conf, FACE_IDX)
    if f is not None and wf >= MIN_SHO_W:
        return f, "face"
    s, ws = _wmean(kp_xy, kp_conf, SHO_IDX)
    if s is not None and ws >= MIN_SHO_W:
        return s, "sho"
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2], float), "bbox"


class CenterFilter:
    """Ổn định tâm bám mà KHÔNG gây trễ.

    Ý tưởng: tâm = NEO + ĐỘ LỆCH.  Neo (mặt) bám người tức thì; độ lệch neo->tâm
    biến thiên chậm nên EMA mạnh được. Nhiễu keypoint nằm hết trong độ lệch -> bị
    dập, còn chuyển động thật đi qua neo -> không trễ.
    Đo trên 150 frame thật + chuyển động ramp đã biết trước:
        thô            : nhiễu 20.8px, trễ  0px
        EMA cả tâm 0.9 : nhiễu  1.5px, trễ 35px   <- mượt nhưng vô dụng cho điều khiển
        neo + EMA lệch : nhiễu  1.5px, trễ  0px   <- dùng cái này
    """

    def __init__(self, a=CENTER_OFFSET_EMA, jump=None):
        self.a, self.jump = a, jump
        self.reset()

    def reset(self):
        self.off, self.out, self.kind = None, None, None

    def __call__(self, raw, anchor, kind):
        raw, anchor = np.asarray(raw, float), np.asarray(anchor, float)
        if self.off is None or kind != self.kind or (
                self.jump and np.linalg.norm(raw - self.out) > self.jump):
            # đổi loại neo -> giữ liên tục đầu ra bằng cách quy độ lệch về neo mới
            self.off = (self.out - anchor) if self.out is not None and kind != self.kind \
                else (raw - anchor)
            self.kind = kind
        else:
            self.off = self.a * self.off + (1 - self.a) * (raw - anchor)
        self.out = anchor + self.off
        return float(self.out[0]), float(self.out[1])


def _wmean(kp_xy, kp_conf, idx, floor=0.2):
    """(điểm trung bình có trọng số theo conf, độ tin cậy 0..1) hoặc (None, 0).

    Trọng số theo conf thay vì lọc cứng -> khi 1 keypoint mờ dần, điểm trung bình
    TRƯỢT dần thay vì nhảy lúc nó vượt/tụt qua ngưỡng.
    """
    c = np.clip(np.asarray(kp_conf)[idx] - floor, 0, None)
    if c.sum() <= 1e-6:
        return None, 0.0
    p = (np.asarray(kp_xy)[idx] * c[:, None]).sum(0) / c.sum()
    return p, float(np.asarray(kp_conf)[idx].max())


def classify_view(kp_conf, th=0.5):
    """Drone đang thấy phần nào của người -> FULL / BODY / HEAD / NONE."""
    if kp_conf is None:
        return "NONE"
    seen = lambda idx: sum(1 for i in idx if kp_conf[i] > th)
    if seen(HIP_IDX) and (seen(KNEE_IDX) or seen(ANKLE_IDX)):
        return "FULL"
    if seen(SHO_IDX) and seen(HIP_IDX):
        return "BODY"
    if seen(SHO_IDX) or seen(FACE_IDX):
        return "HEAD"
    return "NONE"


def person_center(kp_xy, kp_conf, bbox, view=None, th=0.5):
    """Tâm THÂN người -> (cx, cy, nguồn). Hàm LIÊN TỤC theo độ tin cậy keypoint.

    ĐO      : trung điểm (vai, hông) = tâm thân thật        -> "torso"
    THẤY GÌ : trọng tâm các điểm thân trên NHÌN THẤY được    -> "visible"
              (mặt + vai, trọng số theo conf) -> tầm gần chỉ thấy đầu thì tâm
              nằm ở đầu/cổ, KHÔNG đoán mò xuống chỗ ngoài khung hình.
    Hai ước lượng HÒA TRỘN theo conf hông (HIP_BLEND) nên khi hông hiện dần/mất
    dần, tâm trượt mượt giữa 2 nguồn -> KHÔNG nhảy face <-> body.
    Cuối cùng luôn KẸP vào trong bbox người.
    """
    if kp_conf is None:
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2, "bbox"
    sho, w_sho = _wmean(kp_xy, kp_conf, SHO_IDX)
    hip, w_hip = _wmean(kp_xy, kp_conf, HIP_IDX)
    face, w_face = _wmean(kp_xy, kp_conf, FACE_IDX)

    if sho is None or w_sho < MIN_SHO_W:
        if face is not None and w_face >= MIN_SHO_W:
            return clamp_to_box(face, bbox) + ("face",)
        x1, y1, x2, y2 = bbox
        return (x1 + x2) / 2, (y1 + y2) / 2, "bbox"

    vis, _ = _wmean(kp_xy, kp_conf, FACE_IDX + SHO_IDX)       # chỉ những gì THẤY được

    lo, hi = HIP_BLEND
    w = 0.0 if hip is None else float(np.clip((w_hip - lo) / (hi - lo), 0, 1))
    c = w * (sho + hip) / 2 + (1 - w) * vis if hip is not None else vis
    src = "torso" if w > 0.95 else ("visible" if w < 0.05 else f"mix{w:.2f}")
    return clamp_to_box(c, bbox) + (src,)


def clamp_to_box(c, bbox):
    """Kẹp tâm vào trong bbox (chừa mép BBOX_INSET). Bất biến: không bao giờ ngắm
    ra ngoài phần người camera đang thấy."""
    x1, y1, x2, y2 = bbox
    mx, my = BBOX_INSET * (x2 - x1), BBOX_INSET * (y2 - y1)
    return (float(np.clip(c[0], x1 + mx, x2 - mx)), float(np.clip(c[1], y1 + my, y2 - my)))


def facing(kp_conf, th=0.5):
    return "FRONT" if float(np.mean(kp_conf[FACE_IDX])) > th else "BACK"


def estimate_distance(kp_xy, kp_conf, bbox, fy, frame_h, pitch_rad=0.0, th=0.5):
    """Khoảng cách camera->người (m), pinhole: Z = fy * L_thật / h_pixel.

    Tự chọn đoạn cơ thể tốt nhất theo thứ tự DIST_SEGMENTS:
      full -> legs+ -> torso -> thigh -> bbox (fallback).
    CAMERA GẮN CỨNG (strapdown):
      - Đo đoạn bằng ĐỘ DÀI EUCLID giữa 2 đầu (||p_bot - p_top||), KHÔNG chỉ |dy|
        -> bất biến với ROLL của drone (ảnh xoay không đổi chiều dài đoạn).
      - Bù PITCH: camera chúc xuống θ làm đoạn dọc co ~cos(θ); chia lại để khôi phục
        chiều dài tương đương khi camera ngang.  pitch_rad lấy từ read_attitude().
    Trả về (Z mét, tên đoạn) hoặc (None, None).
    """
    cosp = max(np.cos(pitch_rad), PITCH_CLAMP_COS)  # clamp tránh chia ~0 khi pitch quá lớn
    if kp_xy is not None and kp_conf is not None:
        for name, top_i, bot_i, ratio in DIST_SEGMENTS:
            top = [i for i in top_i if kp_conf[i] > th]
            bot = [i for i in bot_i if kp_conf[i] > th]
            if not top or not bot:
                continue
            p_top = kp_xy[top].mean(axis=0)
            p_bot = kp_xy[bot].mean(axis=0)
            seg = float(np.hypot(p_bot[0] - p_top[0], p_bot[1] - p_top[1]))  # roll-invariant
            if seg < MIN_SEG_PX:         # quá ngắn -> bỏ
                continue
            h_eff = seg / cosp           # khôi phục về "camera ngang" (bù pitch)
            return fy * ADULT_HEIGHT_M * ratio / h_eff, name
    # Fallback: chiều cao bbox ~ chiều cao người, CHỈ khi:
    #  - người không bị cắt bởi mép trên/dưới khung hình, VÀ
    #  - bbox có dáng "toàn thân đứng" (cao/rộng >= 1.8). Bbox đầu+vai khi đứng
    #    sát camera gần như vuông -> nếu coi là 1.70m sẽ thổi Z lên gấp nhiều lần.
    # LƯU Ý strapdown: bbox AABB phình to khi ảnh roll -> kém tin; chỉ dùng khi bí.
    x1, y1, x2, y2 = bbox
    bh, bw = (y2 - y1), max(x2 - x1, 1e-6)
    if y1 > 5 and y2 < frame_h - 5 and bh > 2 * MIN_SEG_PX and bh / bw >= 1.8:
        return fy * ADULT_HEIGHT_M / (bh / cosp), "bbox"
    return None, None


class AttitudeSource:
    """Đọc roll/pitch (RAD) của drone từ MAVLink ATTITUDE (ArduPilot).
    Quy ước ArduPilot/MAVLink: pitch>0 = MŨI NGẨNG LÊN, roll>0 = NGHIÊNG PHẢI.
    Chạy 1 thread nền đọc liên tục; nếu không kết nối được -> trả (0,0)
    (coi camera ngang -> hành vi y như chưa bù)."""

    def __init__(self, connect=None, baud=57600):
        self.roll = 0.0
        self.pitch = 0.0
        self._lock = threading.Lock()
        self.ok = False
        if not connect:
            return
        try:
            from pymavlink import mavutil
            self._m = mavutil.mavlink_connection(connect, baud=baud)
            self._m.wait_heartbeat(timeout=5)
            # Xin luồng ATTITUDE ~30Hz (ArduPilot dùng EXTRA1 cho attitude)
            self._m.mav.request_data_stream_send(
                self._m.target_system, self._m.target_component,
                mavutil.mavlink.MAV_DATA_STREAM_EXTRA1, 30, 1)
            threading.Thread(target=self._loop, daemon=True).start()
            self.ok = True
            print(f"[ATT] MAVLink OK: {connect}")
        except Exception as e:
            print(f"[ATT] không kết nối MAVLink ({e}) -> pitch=roll=0")

    def _loop(self):
        while True:
            msg = self._m.recv_match(type="ATTITUDE", blocking=True)
            if msg is not None:
                with self._lock:
                    self.roll, self.pitch = float(msg.roll), float(msg.pitch)

    def get(self):
        with self._lock:
            return self.roll, self.pitch


_ATT = None   # singleton, khởi tạo lười ở lần gọi đầu


def read_attitude():
    """(roll, pitch) RAD để bù camera GẮN CỨNG (strapdown) trên ArduPilot.
    - Khoảng cách: chia cos(pitch) -> KHÔNG phụ thuộc dấu pitch.
    - ang_y = ang_ảnh - pitch  (đúng với quy ước pitch NGẨNG-DƯƠNG của ArduPilot).
    Mặc định (0,0) khi MAV_CONNECT=None -> hành vi như camera ngang.
    KIỂM DẤU 1 lần: nghiêng/chúi drone bằng tay, xem ang_x/ang_y/Z có GIỮ NGUYÊN
    trên 1 vật cố định không; nếu lệch ngược -> đổi dấu roll hoặc pitch tại đây."""
    global _ATT
    if _ATT is None:
        _ATT = AttitudeSource(MAV_CONNECT, MAV_BAUD)
    return _ATT.get()


def decide_command(has_target, z, view, lost_target):
    """Quy trạng thái nhìn thấy mục tiêu -> 1 LỆNH cấp cao (không tính setpoint)."""
    if has_target:
        if view == "HEAD":                        # chỉ thấy đầu+vai = quá gần để khung thân
            return "BACK_OFF"
        if z is not None and z < STANDOFF_MIN_M:  # quá gần theo khoảng cách
            return "BACK_OFF"
        return "FOLLOW"
    if lost_target:                               # từng khóa, đang mất -> quét tìm
        return "SEARCH"
    return "IDLE"


def emit_command(cmd, ang_x=None, ang_y=None, z=None):
    """Gửi LỆNH sang node điều khiển. GỌI MỖI FRAME.
    - Publish ra /follow/command + /follow/target mỗi frame (control cần stream liên tục).
    - In ra console khi lệnh ĐỔI cho đỡ rối."""
    # publish (mỗi frame)
    if ROS_OK and _pub_cmd is not None:
        _pub_cmd.publish(String(data=cmd))
        v = Vector3()
        v.x = float(ang_x) if ang_x is not None else 0.0
        v.y = float(ang_y) if ang_y is not None else 0.0
        v.z = float(z) if z is not None else -1.0   # z<=0 -> control coi như "không có z"
        _pub_tgt.publish(v)
    # in khi đổi lệnh
    if getattr(emit_command, "_last", None) != cmd:
        ax = f"{ang_x:+.1f}" if ang_x is not None else "--"
        ay = f"{ang_y:+.1f}" if ang_y is not None else "--"
        zz = f"{z:.2f}" if z is not None else "--"
        print(f"[CMD] {cmd:8s} ang_x={ax} ang_y={ay} z={zz}")
        emit_command._last = cmd


def draw_pose(frame, kp_xy, kp_conf, color, th=0.5):
    # th 0.5 chu khong 0.3: khop ngoai khung (hong khi nguoi dung sat camera) model
    # van doan ra o ~0.3, ve len la thanh doan xuong noi toi mep anh.
    for a, b in SKELETON:
        if kp_conf[a] > th and kp_conf[b] > th:
            cv2.line(frame, tuple(kp_xy[a].astype(int)), tuple(kp_xy[b].astype(int)), color, 2)
    for i, p in enumerate(kp_xy):
        if kp_conf[i] > th:
            cv2.circle(frame, tuple(p.astype(int)), 3, color, -1)


def draw_crosshair(frame, size=14, color=(255, 255, 0)):
    """Dấu + tại tâm khung hình để so độ lệch mục tiêu."""
    h, w = frame.shape[:2]
    cx, cy = w // 2, h // 2
    cv2.line(frame, (cx - size, cy), (cx + size, cy), color, 2)
    cv2.line(frame, (cx, cy - size), (cx, cy + size), color, 2)
    cv2.circle(frame, (cx, cy), 2, color, -1)


class TargetTracker:
    """Quản lý 1 mục tiêu duy nhất qua ID + ReID gallery."""

    def __init__(self, gallery_size=15, reacquire_sim=0.55,
                 diversity_sim=0.90, max_lost=90):
        self.target_id = None
        self.gallery = deque(maxlen=gallery_size)
        self.lost_frames = 0
        self.reacquire_sim = reacquire_sim
        self.diversity_sim = diversity_sim
        self.max_lost = max_lost

    def _best_sim(self, feat):
        if not self.gallery:
            return 0.0
        return max(float(np.dot(feat, g)) for g in self.gallery)

    def _add_to_gallery(self, feat):
        if self._best_sim(feat) < self.diversity_sim:
            self.gallery.append(feat)

    def update(self, ids, areas, feats):
        if len(ids) == 0:
            if self.target_id is not None:
                self.lost_frames += 1
                if self.lost_frames > self.max_lost:
                    self._reset()
            return None
        if self.target_id is None:
            idx = int(np.argmax(areas))
            self.target_id = ids[idx]
            self.gallery.clear()
            self._add_to_gallery(feats[idx])
            self.lost_frames = 0
            return idx
        if self.target_id in ids:
            idx = ids.index(self.target_id)
            self._add_to_gallery(feats[idx])
            self.lost_frames = 0
            return idx
        self.lost_frames += 1
        best_idx, best_sim = -1, 0.0
        for i, f in enumerate(feats):
            s = self._best_sim(f)
            if s > best_sim:
                best_sim, best_idx = s, i
        if best_sim >= self.reacquire_sim:
            self.target_id = ids[best_idx]
            self._add_to_gallery(feats[best_idx])
            self.lost_frames = 0
            return best_idx
        if self.lost_frames > self.max_lost:
            self._reset()
        return None

    def _reset(self):
        self.target_id = None
        self.gallery.clear()
        self.lost_frames = 0


class ReID:
    """OSNet x0.25 chạy bằng onnxruntime (thay torchreid.FeatureExtractor + torch).
    Tiền xử lý phải giống hệt torchreid: RGB, resize 256x128, /255, chuẩn hóa ImageNet."""

    MEAN = np.array([0.485, 0.456, 0.406], np.float32).reshape(3, 1, 1)
    STD = np.array([0.229, 0.224, 0.225], np.float32).reshape(3, 1, 1)

    def __init__(self, path, threads=4):
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.iname = self.sess.get_inputs()[0].name

    def __call__(self, frame, boxes):
        """-> list dài bằng boxes, mỗi phần tử là vector 512-D đã chuẩn hóa, hoặc None."""
        crops, valid = [], []
        h, w = frame.shape[:2]
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            x1, y1 = max(0, int(x1)), max(0, int(y1))
            x2, y2 = min(w, int(x2)), min(h, int(y2))
            if x2 - x1 < 8 or y2 - y1 < 16:
                continue
            c = cv2.resize(frame[y1:y2, x1:x2], (128, 256), interpolation=cv2.INTER_LINEAR)
            c = c[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0   # BGR->RGB, CHW
            crops.append((c - self.MEAN) / self.STD)
            valid.append(i)
        feats = [None] * len(boxes)
        if crops:
            emb = self.sess.run(None, {self.iname: np.stack(crops)})[0]
            emb /= (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
            for j, i in enumerate(valid):
                feats[i] = emb[j]
        return feats


def main(a):
    if getattr(a, "detect_only", False):
        return detect_only(a)

    detector = Detector(a.model, a.threads)
    reid = ReID(a.reid, a.threads)
    ids_tracker = IoUTracker(iou_thr=0.3, max_age=15, min_hits=2)   # thay BoT-SORT
    tracker = TargetTracker()
    dist_hist = deque(maxlen=1)   # median Z; maxlen thật do _tune_for_fps đặt
    prev_target_id = None
    view_sticky = Sticky("NONE", VIEW_STICKY_N)   # chống nhấp nháy nhãn/lệnh
    center_filter, ema_id = CenterFilter(), None  # ổn định tâm bám (neo + lọc độ lệch)
    loop_dt = 1.0 / FPS_BOOT      # chu kỳ vòng lặp đã lọc -> quy ngưỡng giây sang frame
    init_ros()                    # ROS2: bật publisher /follow/command, /follow/target

    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2) if isinstance(src, int) else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được camera {a.source!r}")
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)
        for _ in range(5):
            cap.read()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        H, W = frame.shape[:2]
        dist_hist = _tune_for_fps(1.0 / loop_dt, ids_tracker, tracker,
                                  view_sticky, center_filter, dist_hist)

        # --- 1 model: box + 17 keypoint, rồi gán ID bằng IoU ---
        dets = detector(frame, a.conf)
        live = ids_tracker.update(dets)
        STATS["persons"] = len(live)
        t_inf = (time.perf_counter() - t0) * 1000

        # --- ReID + khóa/bám/tìm lại ---
        boxes = [t.box for t in live]
        feats = reid(frame, boxes) if boxes else []
        keep = [i for i, f in enumerate(feats) if f is not None]
        tracks_k = [live[i] for i in keep]
        ids_k = [live[i].id for i in keep]
        areas_k = [float((b[2] - b[0]) * (b[3] - b[1])) for b in (boxes[i] for i in keep)]
        feats_k = [feats[i] for i in keep]
        tgt_idx = tracker.update(ids_k, areas_k, feats_k)

        # --- Vẽ mọi người (box + id) ---
        for i, t in enumerate(tracks_k):
            x1, y1, x2, y2 = map(int, t.box)
            is_tgt = (i == tgt_idx)
            color = (0, 255, 0) if is_tgt else (160, 160, 160)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2 if is_tgt else 1)
            if not is_tgt:
                cv2.putText(frame, f"id{t.id}", (x1, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # --- Lệnh mặc định mỗi frame (sẽ được ghi đè bên dưới nếu có thông tin) ---
        cmd, cmd_ax, cmd_ay, cmd_z = "IDLE", None, None, None

        # --- Tâm + độ lệch + khoảng cách của mục tiêu ---
        if tgt_idx is not None:
            tgt = tracks_k[tgt_idx]
            tb = tgt.box
            cx, cy, src_c, ori, view = (tb[0]+tb[2])/2, (tb[1]+tb[3])/2, "bbox", "?", "NONE"
            kp_xy_t, kp_conf_t = None, None
            kp = tgt.kpts                          # keypoint đi kèm chính detection này
            if kp is not None:
                kp_xy_t, kp_conf_t = kp[:, :2], kp[:, 2]
                view = view_sticky(classify_view(kp_conf_t))   # FULL/BODY/HEAD/NONE, đã chống nhấp nháy
                draw_pose(frame, kp_xy_t, kp_conf_t, (0, 255, 0))
                cx, cy, src_c = person_center(kp_xy_t, kp_conf_t, tb)
                ori = facing(kp_conf_t)

            # --- Ổn định tâm: neo vào điểm tin cậy nhất, chỉ lọc độ lệch ---
            cx_raw, cy_raw = cx, cy
            if tracker.target_id != ema_id:            # đổi người -> bỏ lịch sử lọc
                center_filter.reset()
                ema_id = tracker.target_id
            center_filter.jump = CENTER_JUMP_RESET * W
            anchor, akind = center_anchor(kp_xy_t, kp_conf_t, tb) if kp is not None \
                else (np.array([cx, cy]), "bbox")
            cx, cy = center_filter((cx, cy), anchor, akind)

            cv2.line(frame, (W // 2, H // 2), (int(cx), int(cy)), (0, 0, 255), 1)
            cv2.circle(frame, (int(cx_raw), int(cy_raw)), 2, (200, 200, 200), -1)  # tâm thô (trước EMA)
            # đặc = tâm thân ĐO trực tiếp, rỗng = có phần NGOẠI SUY từ đầu/vai
            cv2.circle(frame, (int(cx), int(cy)), 6, (0, 0, 255), -1 if src_c == "torso" else 2)
            # px -> ĐỘ (pinhole): fx = (W/2)/tan(HFOV/2); pixel vuông -> fy = fx
            fx = (W / 2) / np.tan(np.radians(HFOV_DEG) / 2)
            # Camera GẮN CỨNG: lấy tư thế drone để khử roll + bù pitch
            roll, pitch = read_attitude()
            dx, dy = cx - W / 2, cy - H / 2
            # de-roll: xoay offset về hệ "camera ngang" theo roll của drone
            cr, sr = np.cos(roll), np.sin(roll)
            dx_l = cr * dx + sr * dy
            dy_l = -sr * dx + cr * dy
            ang_x = np.degrees(np.arctan2(dx_l, fx))                 # + : người lệch PHẢI
            ang_y = np.degrees(np.arctan2(dy_l, fx)) - np.degrees(pitch)  # + : DƯỚI đường chân trời thật
            # (đã trừ pitch -> ang_y là độ cao THẬT của người, không lẫn việc drone chúi)

            # --- Khoảng cách (mono, theo chiều cao người 1.70m, bù pitch) ---
            if tracker.target_id != prev_target_id:          # đổi mục tiêu -> xóa lịch sử
                dist_hist.clear()
                prev_target_id = tracker.target_id
            z_raw, z_src = estimate_distance(kp_xy_t, kp_conf_t, tb, fx, H, pitch_rad=pitch)
            if z_raw is not None and DIST_RANGE[0] < z_raw < DIST_RANGE[1]:
                dist_hist.append(z_raw)
            z = float(np.median(dist_hist)) if dist_hist else None
            z_txt = f"Z={z:.2f}m ({z_src})" if z is not None else "Z=?"

            # --- Lệnh cấp cao cho mục tiêu hiện tại ---
            cmd = decide_command(True, z, view, False)
            cmd_ax, cmd_ay, cmd_z = ang_x, ang_y, z

            cv2.putText(frame, f"TARGET id{tgt.id} [{ori}] view={view} {z_txt}",
                        (int(tb[0]), int(tb[1]) - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
            cv2.putText(frame, f"view={view} center={src_c} neo={akind} ang_x={ang_x:+.1f}deg ang_y={ang_y:+.1f}deg",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif tracker.target_id is not None:
            cmd = decide_command(False, None, None, True)    # đang mất -> SEARCH
            cv2.putText(frame, f"SEARCHING... lost={tracker.lost_frames}",
                        (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # --- Xuất LỆNH sang code điều khiển (mỗi frame) ---
        emit_command(cmd, cmd_ax, cmd_ay, cmd_z)

        loop_dt = 0.9 * loop_dt + 0.1 * (time.perf_counter() - t0)   # EMA, chống giật 1 frame
        fps = 1.0 / loop_dt
        cv2.putText(frame, f"FPS {fps:.1f}  infer {t_inf:.0f}ms  det {len(dets)}  "
                           f"live {len(live)}  views {len(tracker.gallery)}  "
                           f"lost{tracker.max_lost}/age{ids_tracker.max_age}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        cv2.putText(frame, f"CMD: {cmd}", (10, H - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CMD_COLORS.get(cmd, (255, 255, 255)), 2)
        draw_crosshair(frame)
        cv2.imshow("person-follow pi5 (q=quit, r=reset)", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            tracker._reset()
            dist_hist.clear()
            center_filter.reset()

    cap.release()
    cv2.destroyAllWindows()
    shutdown_ros()



# Chan drone (den quan bang do) nam co dinh trong khung camera nhin xuong; detect COCO nhan
# thanh PERSON 0,4-0,6 (bay that 27/10/2026, bam ca 2 phut). Chuan hoa [x1,y1,x2,y2] tren khung
# GOC zoom 1, do tu video tablet. Doi cho gan camera / chan thi do lai.
LEG_ZONES = np.array([[0.00, 0.25, 0.18, 1.00],    # chan trai
                      [0.75, 0.35, 0.90, 1.00]])   # chan phai


def on_legs(dets, shape, zoom=1.0):
    """Hop nguoi >= 60% nam trong vung chan VA cham mep duoi khung (chan luon cham mep duoi).

    ponytail: nguoi that dung ngay duoi drone, trong dai mep, cham mep duoi cung bi bo;
    can hon thi loc theo hop dung yen khi drone dang di chuyen.
    """
    h, w = shape[:2]
    z = np.clip((LEG_ZONES - 0.5) * zoom + 0.5, 0, 1) * [w, h, w, h]   # zoom = cat GIUA 1/zoom
    b = dets[:, None, :4]
    iw = (np.minimum(b[..., 2], z[:, 2]) - np.maximum(b[..., 0], z[:, 0])).clip(0)
    ih = (np.minimum(b[..., 3], z[:, 3]) - np.maximum(b[..., 1], z[:, 1])).clip(0)
    area = ((b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])).clip(1)
    return ((iw * ih / area).max(1) >= 0.6) & (dets[:, 3] >= 0.95 * h)


def detect_only(a):
    """Detect and draw every person; no ReID, target lock, or ROS commands."""
    detector = Detector(a.model, a.threads)
    # Chi ve/dem: mượt + giu hop 3 khung (0,1 s o 30 fps) + phai thay 3 khung moi hien
    # -> het rung, het chop tat, het hop rac 1 khung. Tre them ~1 khung, ve thi khong sao.
    ids_tracker = IoUTracker(iou_thr=0.3, max_age=15, min_hits=3, smooth=0.7, hold=3)
    src = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2) if isinstance(src, int) else cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Không mở được camera {a.source!r}")
    if isinstance(src, int):
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, a.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, a.height)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        dets = detector(frame, a.conf, classes=[0])  # model detect COCO co 80 lop, 0 = person
        if dets.shape[1] > 6 and len(dets):  # reshape(0, -1, 3) loi khi khong co ai
            # Pose: bo hop co < 4 khop ro - model hay ra hop chi om CANH TAY nguoi ngoi
            # sat camera (3 khop). 90 khung that: dem dung 96% vs 19%, khong mat nguoi xa.
            dets = dets[(dets[:, 6:].reshape(len(dets), -1, 3)[:, :, 2] > 0.5).sum(1) >= 4]
        if len(dets):
            dets = dets[~on_legs(dets, frame.shape, getattr(cap, "zoom", 1.0))]   # GstCap co .zoom
        live = ids_tracker.update(dets)
        STATS["persons"] = len(live)
        for t in live:
            x1, y1, x2, y2 = map(int, t.box)
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"PERSON {t.conf:.2f}",
                        (x1, max(16, y1 - 7)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            kp = t.kpts
            if kp is not None:  # model detect thuong khong co khop -> chi hop
                draw_pose(frame, kp[:, :2], kp[:, 2], color)
        cv2.putText(frame, f"PERSONS {len(live)}  DET {len(dets)}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (255, 255, 0), 2)
        cv2.imshow("person-detect-only", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()

def selftest():
    kp = np.zeros((17, 2), np.float32)
    cf = np.zeros(17, np.float32)
    box = (0, 0, 100, 400)
    assert classify_view(cf) == "NONE" and person_center(kp, cf, box)[2] == "bbox"

    # chỉ vai (quay lưng, quá gần) -> tâm NẰM TRÊN VAI, không đoán xuống dưới
    kp[[5, 6]] = [[40, 100], [60, 100]]; cf[[5, 6]] = 0.9
    assert classify_view(cf) == "HEAD"
    assert person_center(kp, cf, box)[:3] == (50.0, 100.0, "visible")

    # thêm mặt -> tâm là trọng tâm mặt+vai, tức nằm GIỮA mặt và vai
    kp[FACE_IDX] = [[50, 70], [45, 65], [55, 65]]; cf[FACE_IDX] = 0.9
    cx, cy, src = person_center(kp, cf, box)   # mặt tb y=66.67 (3 điểm), vai y=100 (2 điểm)
    assert src == "visible" and 66.67 < cy < 100, (cy, src)
    assert abs(cy - 80.0) < 0.1, cy

    # BẤT BIẾN: dù keypoint có nằm ngoài khung/bbox, tâm vẫn bị kẹp vào trong bbox
    kp_o = kp.copy(); kp_o[[5, 6]] = [[40, 5000], [60, 5000]]   # vai bị model đoán bậy ra ngoài
    cxo, cyo = person_center(kp_o, cf, box)[:2]
    assert box[1] <= cyo <= box[3] and box[0] <= cxo <= box[2], (cxo, cyo)

    kp[[11, 12]] = [[42, 200], [58, 200]]; cf[[11, 12]] = 0.9
    assert classify_view(cf) == "BODY"                      # vai + hông, chưa thấy chân
    assert person_center(kp, cf, box)[:3] == (50.0, 150.0, "torso")
    kp_b, cf_b = kp.copy(), cf.copy()          # trạng thái BODY, dùng cho test khoảng cách
    kp[[15, 16]] = [[45, 380], [55, 380]]; cf[[15, 16]] = 0.9
    assert classify_view(cf) == "FULL"                      # có chân -> cả người
    assert person_center(kp, cf, box)[:3] == (50.0, 150.0, "torso")   # tâm KHÔNG nhảy

    # LIÊN TỤC: hông hiện dần (conf 0 -> 1) thì tâm phải TRƯỢT, không nhảy bậc.
    # Đây chính là lỗi giật face<->body: trước đây vượt ngưỡng 0.5 là nhảy ~10px/frame.
    kp_c, cf_c = kp.copy(), cf.copy()
    kp_c[[11, 12]] = [[42, 200], [58, 200]]
    prev, steps = None, []
    for hc in np.arange(0.0, 1.0001, 0.02):
        cf_c[[11, 12]] = hc
        cy_i = person_center(kp_c, cf_c, box)[1]
        if prev is not None:
            steps.append(abs(cy_i - prev))
        prev = cy_i
    assert max(steps) < 3.0, f"tâm nhảy {max(steps):.2f}px giữa 2 bước conf"
    assert abs(prev - 150.0) < 1e-6                          # hông rõ hẳn -> đúng tâm đo

    # conf hông NHẤP NHÁY quanh vùng hòa trộn (đúng cảnh thật) -> sau lọc phải im
    fl = CenterFilter(a=0.9)
    rng2 = np.random.default_rng(1)
    raw2, outs2 = [], []
    for _ in range(200):
        cf_c[[11, 12]] = rng2.uniform(0.25, 0.75)     # trường hợp xấu nhất: nhảy loạn cả dải
        c2 = person_center(kp_c, cf_c, box)[:2]
        a2, k2 = center_anchor(kp_c, cf_c, box)
        raw2.append(c2); outs2.append(fl(c2, a2, k2))
    s_raw = np.std(np.array(raw2[50:])[:, 1])
    s_out = np.std(np.array(outs2[50:])[:, 1])
    assert s_out < s_raw / 4, f"lọc chỉ dập được {s_raw / s_out:.1f}x ({s_raw:.1f} -> {s_out:.1f}px)"

    # CenterFilter: neo chạy thì bám ngay (trễ 0), nhiễu quanh neo thì bị dập
    cf_f = CenterFilter(a=0.9)
    for i in range(30):                                      # neo trượt 4px/frame, lệch cố định
        out = cf_f((4.0 * i + 10, 5), (4.0 * i, 0), "face")
    assert abs(out[0] - (4 * 29 + 10)) < 1e-6 and abs(out[1] - 5) < 1e-6, out   # bám sát, KHÔNG trễ
    cf_f.reset()
    rng = np.random.default_rng(0)
    outs = [cf_f((10 + rng.normal(0, 10), 5 + rng.normal(0, 10)), (0, 0), "face") for _ in range(200)]
    jit = np.std(np.array(outs[50:]), axis=0)
    assert (jit < 4).all(), jit                              # nhiễu vào 10px -> ra < 4px
    prev = cf_f.out.copy()
    now = cf_f((100, 100), (90, 90), "sho")                  # đổi loại neo -> đầu ra liên tục
    assert np.linalg.norm(np.array(now) - prev) < 1e-6, (prev, now)

    st = Sticky("A", 3)
    assert [st("B"), st("B"), st("B")] == ["A", "A", "B"]    # phải lặp 3 frame mới đổi
    assert [st("A"), st("B"), st("A")] == ["B", "B", "B"]    # nhấp nháy -> giữ nguyên

    assert facing(np.zeros(17, np.float32)) == "BACK"        # chưa thấy mặt
    assert facing(cf) == "FRONT"

    # người cao 1.70m, vai->hông = 0.29H = 100px, fx=800 -> Z = 800*0.493/100 ~ 3.94m
    z, name = estimate_distance(kp_b, cf_b, box, 800, 480)
    assert name == "torso" and abs(z - 3.944) < 0.01, (z, name)
    # pitch 45deg: đoạn dọc co lại -> khôi phục -> Z nhỏ hơn theo cos
    z45, _ = estimate_distance(kp_b, cf_b, box, 800, 480, pitch_rad=np.pi / 4)
    assert abs(z45 - z * np.cos(np.pi / 4)) < 0.01, (z, z45)
    # roll 90deg (người nằm ngang trong ảnh): euclid bất biến -> Z không đổi
    kp_r2 = kp_b.copy(); kp_r2[[5, 6, 11, 12]] = [[100, 40], [100, 60], [200, 42], [200, 58]]
    assert abs(estimate_distance(kp_r2, cf_b, box, 800, 480)[0] - z) < 0.01

    assert decide_command(True, 5.0, "FULL", False) == "FOLLOW"
    assert decide_command(True, 1.0, "FULL", False) == "BACK_OFF"    # dưới standoff
    assert decide_command(True, 9.0, "HEAD", False) == "BACK_OFF"    # chỉ thấy đầu
    assert decide_command(False, None, None, True) == "SEARCH"
    assert decide_command(False, None, None, False) == "IDLE"

    t = TargetTracker(reacquire_sim=0.55)
    f1 = np.array([1, 0, 0], np.float32); f2 = np.array([0, 1, 0], np.float32)
    assert t.update([7, 9], [100, 500], [f1, f2]) == 1 and t.target_id == 9   # khóa người to nhất
    assert t.update([9, 7], [500, 100], [f2, f1]) == 0                        # bám đúng id
    assert t.update([3], [500], [f2]) == 0 and t.target_id == 3               # id đổi -> ReID nhận lại
    assert t.update([4], [500], [f1]) is None                                 # khác người -> không nhận
    # --- ngưỡng theo giây: cùng THỜI GIAN thật ở hai nhịp rất khác nhau ---
    class _Trk:
        pass

    for f in (22.0, 3.6):
        it, tt = _Trk(), TargetTracker()
        st, cf, dh = Sticky("NONE", 3), CenterFilter(), deque(maxlen=1)
        dh = _tune_for_fps(f, it, tt, st, cf, dh)
        assert abs(tt.max_lost / f - LOST_S) < 0.5, (f, tt.max_lost)
        assert abs(it.max_age / f - TRACK_AGE_S) < 0.5, (f, it.max_age)
        assert dh.maxlen >= 3 and st.n >= 2, (f, dh.maxlen, st.n)   # sàn của bộ lọc
    # ở 3.6 fps ngưỡng phải NGẮN HƠN NHIỀU tính bằng frame so với 22 fps
    it_s, tt_s = _Trk(), TargetTracker()
    _tune_for_fps(3.6, it_s, tt_s, Sticky("NONE", 3), CenterFilter(), deque(maxlen=1))
    assert tt_s.max_lost < 20, tt_s.max_lost          # 90 frame cũ = 25 s trên Pi
    # EMA: giữ nguyên HẰNG SỐ THỜI GIAN, không giữ nguyên hệ số a
    cf22, cf36 = CenterFilter(), CenterFilter()
    _tune_for_fps(22.0, _Trk(), TargetTracker(), Sticky("NONE", 3), cf22, deque(maxlen=1))
    _tune_for_fps(3.6, _Trk(), TargetTracker(), Sticky("NONE", 3), cf36, deque(maxlen=1))
    assert abs(cf22.a - CENTER_OFFSET_EMA) < 0.02, cf22.a   # 22 fps = mốc gốc, phải ra 0.9
    assert cf36.a < 0.6, cf36.a                             # nhịp chậm -> lọc phải nhẹ đi

    # --- chan drone: hop that do tu video bay 27/10 (khung 1280x720) ---
    d = np.array([[970, 271, 1141, 720, .58, 0],            # chan phai, PERSON 0.58
                  [8, 202, 228, 718, .41, 0],               # chan trai, PERSON 0.41
                  [600, 300, 700, 600, .7, 0],              # nguoi giua khung
                  [20, 300, 200, 650, .7, 0]], float)       # trong dai chan nhung KHONG cham mep duoi
    assert on_legs(d, (720, 1280)).tolist() == [True, True, False, False]
    assert not on_legs(d[:2], (720, 1280), zoom=2.0).any()  # zoom 2: chan ra ngoai khung cat

    print("selftest ok")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=POSE_ONNX)
    p.add_argument("--reid", default=REID_ONNX)
    p.add_argument("--source", default=str(CAM_INDEX))
    p.add_argument("--conf", type=float, default=CONF)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--selftest", action="store_true")
    p.add_argument("--detect-only", action="store_true",
                   help="detect va ve tat ca nguoi, khong ReID/FOLLOW/ROS2")
    a = p.parse_args()
    selftest() if a.selftest else main(a)
