#!/usr/bin/env python3
"""python3 test_drop_reticle.py - vach ngam tha bong (_FireRos.draw_drop) tren khung gia 1280x720.
Can fire_drone: source ~/Firefighting-drone/install/setup.bash (hoac ~/fire_drone_ws) truoc."""
import sys, pathlib, time, types
sys.path[:0] = [str(pathlib.Path(__file__).parent / "tracking_pi5")]
import cv2, numpy as np
from ament_index_python.packages import get_package_share_directory
import run_gui
from fire_drone.fire_locator import ground_pixel, impact_point
from fire_drone.marker_detector import calib_for, load_calib, quat_to_rot

cal = load_calib(get_package_share_directory("fire_drone") + "/config/drone_cam_3840x2160.yaml")
cam_off, P = np.array([0.20, 0, -0.02]), types.SimpleNamespace


def ring(z, rel, vel=(0, 0, 0), pitch=0.0, zoom=1.0):
    """Tam vong xanh (diem anh khung 1280x720) hoac None. pitch > 0 = chui mui (quay quanh y than FLU)."""
    now = time.monotonic()
    pose = P(pose=P(position=P(x=0.0, y=0.0, z=z),
                    orientation=P(x=0.0, y=np.sin(pitch / 2), z=0.0, w=np.cos(pitch / 2))))
    s = P(calib=cal, _ksize=None, cam_off=cam_off, drop_off=cam_off - [0.17, 0, 0], DROP_DELAY=0.1,
          _fx=(ground_pixel, impact_point, calib_for, quat_to_rot), pose=(now, pose),
          vel=(now, P(twist=P(linear=P(x=vel[0], y=vel[1], z=vel[2])))), rel_alt=(now, P(data=rel)))
    cw, ch = round(3840 / zoom), round(2160 / zoom)
    cap = P(_box=lambda: ((3840 - cw) // 2, (2160 - ch) // 2, cw, ch), cam_w=3840, cam_h=2160, zoom_t=0.0)
    f = np.full((720, 1280, 3), 128, np.uint8)
    run_gui._FireRos.draw_drop(s, f, cap)
    m = (f[:, :, 0] == 255) & (f[:, :, 1] == 255) & (f[:, :, 2] == 0)
    n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
    rings = [st[i] for i in range(1, n) if abs(st[i][2] - st[i][3]) <= 2]   # vong: hop vuong, chu "THA" thi khong
    if not rings:
        return None
    x, y, w, h, _ = max(rings, key=lambda r: r[2])
    return x + w // 2, y + h // 2


cx, cy = cal[0][0, 2] / 3, cal[0][1, 2] / 3                   # tam camera ('+') o khung 1280x720
c = ring(5.0, 5.0)
assert abs(c[0] - cx) < 4 and 20 < c[1] - cy < 40, c          # treo 5 m: tha 0.17 m sau ong kinh -> duoi '+' ~28 px
assert ring(2.0, 2.0)[1] - cy > 2 * (c[1] - cy)               # thap hon -> xa '+' hon
assert ring(5.0, 5.0, vel=(1.0, 0, 0))[1] < cy - 100          # troi 1 m/s ve truoc: bong roi truoc ~1.1 m -> len phia mui
assert 130 < c[1] - ring(5.0, 5.0, pitch=np.radians(10))[1] < 160   # chui mui 10 do: f*tan10 ~145 px len phia mui
assert abs((ring(5.0, 5.0, zoom=2.0)[1] - 360) - 2 * (c[1] - 360)) < 6   # zoom x2 quanh tam khung: lech gap doi
assert ring(0.3, 0.3) is None                                 # duoi dat: khong ve vong
assert abs(ring(6.2, 5.0)[1] - c[1]) <= 2                     # z local troi +1.2 m: rel_alt quyet do cao
print("test_drop_reticle OK")
