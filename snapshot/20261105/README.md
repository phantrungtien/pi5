# Chụp drone thật 05/11/2026 19:40 (bàn thử, không cánh)

- `fc/params.parm`: toàn bộ 1032 tham số FC (Pixhawk 6C, **ArduPilot 4.7.0**, git 1511f271). Định dạng `TEN,GIA_TRI`.
- `fc/mission.waypoints`: đường bay đang nạp (8 mục, QGC WPL 110). `fence`/`rally` trống, FENCE_ENABLE=0.
- `ros2/*.yaml`: `ros2 param dump` từng node đang chạy trong drone-ros.service. `ros2/topics.txt`: topic kèm kiểu.
- Chụp lại: `python3 flight_logs/fc_snapshot.py <thu_muc>` trên Jetson (cổng USB thứ 2, không đụng MAVROS), và `ros2 param dump`.

## Khác với SITL laptop (ArduCopter **4.6.3**, `fire_drone_ws/.../config/firehawk_x4.parm`)

4.7 đổi tên và đơn vị nhiều tham số. Đừng chép thẳng file này vào SITL 4.6.

| | SITL 4.6 | FC thật 4.7 |
|---|---|---|
| Tốc độ ngang WP | `WPNAV_SPEED` 500 cm/s | `WP_SPD` **1 m/s** |
| Lên / xuống | `WPNAV_SPEED_UP/DN` 200/150 cm/s | `WP_SPD_UP/DN` 2.5/1.5 m/s |
| Gia tốc, bán kính WP | `WPNAV_ACCEL` 250, `WPNAV_RADIUS` 200 cm | `WP_ACC` 2.5 m/s², `WP_RADIUS_M` 2 m |
| Stream MAVLink | `SR0_*`, `SR1_*` | `MAV1_*`... |
| Kiểm tra ARM | `ARMING_CHECK` 1 | `ARMING_SKIPCHK` 0 |
| Khung | `FRAME_CLASS` 1 (quad iris) | `FRAME_CLASS` 4 (8 động cơ, SERVO1-8 = 33-40) |
| Gripper thả bóng | `SERVO9_FUNCTION` 28 | `SERVO16_FUNCTION` 28, GRIP_GRAB/NEUTRAL 1900, RELEASE 1100, RC5_OPTION 19 |
| Ga treo | `MOT_THST_HOVER` 0.53 | 0.176 |
| Mất tay cầm | `FS_THR_ENABLE` 0 | 3 (LAND) |

Tham số FC thật khác (4.7): RTL_ALT_M 3, LOIT_SPEED_MS 12.5, PILOT_SPD_UP 2.5, LAND_SPD_MS 0.5, GUID_TIMEOUT 3, RC8_OPTION 153 (công tắc ARM), FLTMODE_CH 6.
