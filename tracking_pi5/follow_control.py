#!/usr/bin/env python3
"""
follow_control.py
---------------------------------------------------------------
Bám đuổi người trên ArduPilot (MAVROS / GUIDED), nhận LỆNH từ node vision
(person_follow_reid.py) qua 2 topic:
    /follow/command  std_msgs/String     : FOLLOW | BACK_OFF | SEARCH | IDLE
    /follow/target   geometry_msgs/Vector3: x=ang_x(deg)  y=ang_y(deg)  z=khoảng cách(m)

Điều khiển kiểu YAW-AND-GO (an toàn, ít phụ thuộc hệ trục):
    - yaw : xoay để đưa người về giữa khung (ang_x -> 0)
    - tiến/lùi : giữ khoảng cách STANDOFF theo z (chỉ tiến khi đã gần thẳng hướng)
    - GIỮ ĐỘ CAO cố định (vz=0) — KHÔNG dùng ang_y để chỉnh cao (camera gắn cứng,
      ang_y dễ lẫn pitch drone).

An toàn (theo luật offboard/guided):
    - Luôn stream setpoint 20Hz; mất vision > VISION_TIMEOUT -> HOVER;
      mất lâu > LOST_RTL_SEC -> RTL.
    - Chỉ điều khiển khi mode == GUIDED & đã arm; pilot đổi mode -> NHẢ quyền (ngừng gửi).
    - Chặn trần độ cao MAX_ALT. Ctrl+C -> RTL.

CẢNH BÁO: test theo thứ tự SITL -> HITL -> bay thật ngoài trời, luôn có safety pilot
cầm RC sẵn sàng giành quyền. Lần đầu nên THÁO CÁNH QUẠT.

Author của khung mission gốc: phantrungtien (github.com/phantrungtien)
"""

import math
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy
from std_msgs.msg import String
from geometry_msgs.msg import Vector3, PoseStamped
from mavros_msgs.msg import State, PositionTarget
from mavros_msgs.srv import CommandBool, SetMode, CommandTOL

# ===== THAM SỐ BÁM ĐUỔI =====
STANDOFF_M   = 3.0     # khoảng cách muốn giữ với người (m)
DIST_TOL     = 0.4     # vùng chết quanh standoff (m) -> đỡ rung tiến/lùi
KP_DIST      = 0.4     # (m/s) / (m) — hệ số tiến/lùi theo sai số khoảng cách
VMAX_FWD     = 1.2     # trần tốc độ tiến/lùi (m/s)
KP_YAW       = 1.2     # (rad/s) / (rad) — hệ số xoay theo ang_x
YAW_MAX      = 0.8     # trần tốc độ xoay (rad/s)
FACE_TOL_DEG = 25.0    # chỉ tiến khi |ang_x| < ngưỡng này (xoay cho thẳng hướng trước)
BACK_SPEED   = 0.6     # tốc độ lùi cố định khi BACK_OFF (m/s) — KHÔNG tin z lúc này
SEARCH_YAW   = 0.4     # tốc độ quét yaw khi SEARCH (rad/s)
YAW_DIR      = +1.0    # dấu yaw: mặc định +1 = xoay VỀ PHÍA người (phải->quay phải).
                       #   nếu trong SITL thấy xoay NGƯỢC (ra xa người) thì đổi thành -1.0

# ===== AN TOÀN =====
DO_TAKEOFF     = True   # tự cất cánh trước khi bám; False = coi như đã bay sẵn ở GUIDED
TAKEOFF_ALT    = 5.0    # m
MAX_ALT        = 30.0   # trần độ cao (m) — vượt thì ngừng (an toàn)
VISION_TIMEOUT = 0.5    # s không có lệnh vision -> HOVER
LOST_RTL_SEC   = 5.0    # s mất vision -> RTL
CTRL_HZ        = 20.0   # tần số stream setpoint


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def compute_setpoint(cmd, ang_x_deg, ang_y_deg, z):
    """Quy LỆNH + số đo -> (vx, vy, vz, yaw_rate) trong BODY-FLU.
    vx: + tiến (mũi), vy: + sang trái, vz: + lên, yaw_rate: + quay TRÁI (CCW).
    Hàm thuần (không ROS) để dễ unit-test."""
    vx = vy = vz = yaw = 0.0

    if cmd == "FOLLOW":
        ax = math.radians(ang_x_deg)
        # người lệch PHẢI (ang_x>0) -> xoay PHẢI (CW = yaw âm trong FLU)
        yaw = clamp(-KP_YAW * ax * YAW_DIR, -YAW_MAX, YAW_MAX)
        # chỉ tiến/lùi khi đã gần thẳng hướng và có z tin được
        if z is not None and abs(ang_x_deg) < FACE_TOL_DEG:
            err = z - STANDOFF_M               # >0: xa quá -> tiến; <0: gần quá -> lùi
            if abs(err) > DIST_TOL:
                vx = clamp(KP_DIST * err, -VMAX_FWD, VMAX_FWD)
                vx *= math.cos(ax)             # chưa thẳng hướng thì tiến chậm lại

    elif cmd == "BACK_OFF":
        ax = math.radians(ang_x_deg)
        yaw = clamp(-KP_YAW * ax * YAW_DIR, -YAW_MAX, YAW_MAX)
        vx = -BACK_SPEED                       # quá gần / chỉ thấy đầu -> lùi đều, bỏ qua z

    elif cmd == "SEARCH":
        yaw = SEARCH_YAW * YAW_DIR             # quét chậm tìm lại, KHÔNG tiến

    # IDLE / timeout / không rõ -> giữ nguyên 0 (hover)
    return vx, vy, vz, yaw


class FollowControl(Node):
    def __init__(self):
        super().__init__('follow_control')

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # --- MAVROS subs ---
        self.state_sub = self.create_subscription(State, '/mavros/state', self._state_cb, qos)
        self.pose_sub = self.create_subscription(
            PoseStamped, '/mavros/local_position/pose', self._pose_cb, qos)

        # --- Vision subs (lệnh + số đo) ---
        self.cmd_sub = self.create_subscription(String, '/follow/command', self._cmd_cb, 10)
        self.tgt_sub = self.create_subscription(Vector3, '/follow/target', self._tgt_cb, 10)

        # --- Setpoint pub (velocity + yaw_rate, body frame) ---
        self.sp_pub = self.create_publisher(PositionTarget, '/mavros/setpoint_raw/local', 10)

        # --- Service clients ---
        self.arming_client = self.create_client(CommandBool, '/mavros/cmd/arming')
        self.set_mode_client = self.create_client(SetMode, '/mavros/set_mode')
        self.takeoff_client = self.create_client(CommandTOL, '/mavros/cmd/takeoff')

        # --- State ---
        self.state = State()
        self.pose = PoseStamped()
        self.cmd = "IDLE"
        self.ang_x = 0.0
        self.ang_y = 0.0
        self.z = None
        self.last_rx = 0.0          # thời điểm nhận lệnh vision gần nhất
        self.active = False         # True sau khi cất cánh -> bắt đầu stream điều khiển
        self.rtl_done = False
        self._warned_mode = False

        print('Chờ services MAVROS...')
        self.arming_client.wait_for_service(timeout_sec=10.0)
        self.set_mode_client.wait_for_service(timeout_sec=10.0)
        self.takeoff_client.wait_for_service(timeout_sec=10.0)
        print('✓ Services sẵn sàng\n')

        # Stream setpoint 20Hz — luôn chạy, chỉ publish khi active
        self.timer = self.create_timer(1.0 / CTRL_HZ, self._control_loop)

    # ---------- callbacks ----------
    def _state_cb(self, msg):
        self.state = msg

    def _pose_cb(self, msg):
        self.pose = msg

    def _cmd_cb(self, msg):
        self.cmd = msg.data.strip().upper()
        self.last_rx = time.time()

    def _tgt_cb(self, msg):
        self.ang_x = msg.x
        self.ang_y = msg.y
        self.z = msg.z if msg.z > 0 else None
        self.last_rx = time.time()

    # ---------- MAVROS helpers (giống mission gốc) ----------
    def set_mode(self, mode):
        print(f'Đặt mode {mode}...')
        req = SetMode.Request()
        req.custom_mode = mode
        fut = self.set_mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        ok = bool(fut.result() and fut.result().mode_sent)
        print('✓' if ok else '✗', f'mode {mode}\n')
        return ok

    def arm(self):
        print('Arm...')
        req = CommandBool.Request()
        req.value = True
        fut = self.arming_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        ok = bool(fut.result() and fut.result().success)
        print('✓ Armed\n' if ok else '✗ Arm fail\n')
        return ok

    def takeoff(self, alt):
        print(f'Cất cánh {alt}m...')
        req = CommandTOL.Request()
        req.altitude = alt
        fut = self.takeoff_client.call_async(req)
        rclpy.spin_until_future_complete(self, fut, timeout_sec=5.0)
        ok = bool(fut.result() and fut.result().success)
        print('✓ Takeoff sent\n' if ok else '✗ Takeoff fail\n')
        return ok

    def wait_for_altitude(self, target_alt, tol=0.5, timeout=30.0):
        print(f'Chờ đạt {target_alt}m...')
        t0 = time.time()
        while rclpy.ok() and time.time() - t0 < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            alt = self.pose.pose.position.z
            print(f'  alt {alt:.2f}/{target_alt}m', end='\r')
            if abs(alt - target_alt) < tol:
                print(f'\n✓ Đạt {alt:.2f}m\n')
                return True
            time.sleep(0.1)
        print('\n✗ Quá thời gian chờ độ cao\n')
        return False

    # ---------- vòng điều khiển 20Hz ----------
    def _publish_velocity(self, vx, vy, vz, yaw_rate):
        sp = PositionTarget()
        sp.header.stamp = self.get_clock().now().to_msg()
        sp.coordinate_frame = PositionTarget.FRAME_BODY_NED   # body (MAVROS: FLU)
        sp.type_mask = (
            PositionTarget.IGNORE_PX | PositionTarget.IGNORE_PY | PositionTarget.IGNORE_PZ |
            PositionTarget.IGNORE_AFX | PositionTarget.IGNORE_AFY | PositionTarget.IGNORE_AFZ |
            PositionTarget.IGNORE_YAW                          # dùng velocity + yaw_rate
        )
        sp.velocity.x = float(vx)   # + tiến (mũi)
        sp.velocity.y = float(vy)   # + sang trái
        sp.velocity.z = float(vz)   # + lên
        sp.yaw_rate = float(yaw_rate)  # + quay trái (CCW)
        self.sp_pub.publish(sp)

    def _control_loop(self):
        if not self.active:
            return

        # Pilot giành quyền (đổi khỏi GUIDED) -> NHẢ, ngừng gửi
        if self.state.mode != 'GUIDED':
            if not self._warned_mode:
                self.get_logger().warn(f'Mode={self.state.mode} != GUIDED -> nhả quyền điều khiển')
                self._warned_mode = True
            return
        self._warned_mode = False

        # Trần độ cao
        alt = self.pose.pose.position.z
        if alt > MAX_ALT:
            self.get_logger().warn(f'alt {alt:.1f} > MAX_ALT -> hover')
            self._publish_velocity(0.0, 0.0, 0.0, 0.0)
            return

        # Mất vision -> hover; mất lâu -> RTL
        dt = time.time() - self.last_rx
        cmd = self.cmd
        if dt > VISION_TIMEOUT:
            cmd = "IDLE"                         # hover an toàn
        if dt > LOST_RTL_SEC and not self.rtl_done:
            self.get_logger().error(f'Mất vision {dt:.1f}s -> RTL')
            self.set_mode('RTL')
            self.rtl_done = True
            self.active = False
            return

        vx, vy, vz, yaw = compute_setpoint(cmd, self.ang_x, self.ang_y, self.z)
        self._publish_velocity(vx, vy, vz, yaw)

    # ---------- khởi động ----------
    def startup(self):
        print('=' * 50)
        print('FOLLOW: chờ FCU -> GUIDED -> arm -> takeoff -> BÁM')
        print('=' * 50, '\n')

        print('Chờ kết nối FCU...')
        while not self.state.connected and rclpy.ok():
            rclpy.spin_once(self, timeout_sec=0.1)
            time.sleep(0.1)
        print('✓ FCU connected\n')

        if DO_TAKEOFF:
            if not self.set_mode('GUIDED'):
                return False
            time.sleep(1)
            if not self.arm():
                return False
            time.sleep(2)
            if not self.takeoff(TAKEOFF_ALT):
                return False
            self.wait_for_altitude(TAKEOFF_ALT)
            time.sleep(2)
        else:
            print('DO_TAKEOFF=False -> giả định đã bay ở GUIDED\n')

        self.last_rx = time.time()   # tránh timeout ngay khi vừa kích hoạt
        self.active = True
        print('✓ BẮT ĐẦU BÁM ĐUỔI (Ctrl+C để RTL & thoát)\n')
        return True

    def safe_shutdown(self):
        """RTL khi thoát nếu còn đang bay/điều khiển."""
        try:
            if self.state.armed:
                print('\nRTL...')
                self.set_mode('RTL')
        except Exception:
            pass


def main():
    rclpy.init()
    node = FollowControl()
    try:
        if node.startup():
            rclpy.spin(node)
    except KeyboardInterrupt:
        print('\n\nNgắt bởi người dùng')
    finally:
        node.safe_shutdown()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()