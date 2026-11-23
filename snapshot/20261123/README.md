# Chụp FC thật 23/11/2026 17:57 (bàn thử, không cánh)

- `fc/params.parm`: 1032 tham số, Pixhawk 6C **ArduPilot 4.7.0** (git 1511f271). Mission / fence / rally trống.
- Chụp lại: `python3 flight_logs/fc_snapshot.py snapshot/<ngay>/fc` trên Jetson (cổng if02, không đụng MAVROS).

## Khác bản 05/11/2026 (`../20261105`), bỏ offset hiệu chuẩn IMU / la bàn

| Tham số | 05/11 | 23/11 | Ghi chú |
|---|---|---|---|
| `BATT_MONITOR` | 4 | **3** | chỉ đo áp (cảm biến dòng cháy) |
| `BATT_LOW_VOLT` / `BATT_CRT_VOLT` | 21 / 19 | **21.6 / 21** | 3.6 / 3.5 V/cell -> RTL / LAND |
| `BATT_ARM_VOLT` | 20 | **22.1** | 3.68 V/cell nghỉ - vẫn ARM được pin vơi |
| `FS_THR_ENABLE` | 3 (LAND) | **1 (RTL)** | user cố ý 23/11 |
| `GRIP_NEUTRAL` | 1900 (mở) | **1100 (đóng)** | |
| `RTL_ALT_M` | 3 | **5** | = độ cao bay tìm |
| `RC_OPTIONS` | 32 | **0** | bỏ bit 5 (kiểm ga khi tín hiệu 0 lúc ARM) |
| `MOT_THST_HOVER` / `MOT_THST_EXPO` | 0.176 / 0.58 | **0.2 / 0.6** | `MOT_HOVER_LEARN` 2 sẽ tự học lại khi bay |
| `INS_GYRO_FILTER` | 46 | **42** | |
| `ATC_RAT_RLL/PIT_FLTD/FLTT`, `ATC_RAT_YAW_FLTT` | 23 | **21** | |
| `MAV2_*` (SERIAL1 = TELEM1) | EXTRA1 10, EXTRA2 4, POSITION 3, EXT_STAT 2... | **tất cả 0** | ⚠ cổng SiK: GCS phải tự xin luồng |
| `MAV4_EXT_STAT` / `MAV4_RC_CHAN` (SERIAL7) | 10 / 10 | **0 / 0** | |
| `MIS_TOTAL` | 8 | 0 | đã xoá đường bay |
