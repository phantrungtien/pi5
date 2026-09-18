# tracking_pi5 — bám người bằng ONNX trên Raspberry Pi 5

Vision node cho drone follow-me. Chạy **thuần onnxruntime + OpenCV**, không torch,
không ultralytics, không torchreid — cài trên Pi 5 chỉ mất 3 gói pip.

Xuất ra 1 lệnh cấp cao mỗi frame (`FOLLOW` / `BACK_OFF` / `SEARCH` / `IDLE`) kèm
`ang_x`, `ang_y`, `z` cho node điều khiển; kèm khung hình debug để soi mọi thứ đang
diễn ra — trên Pi khung đó đi ra **MJPEG cổng 8080** cho GUI_NATIVE xem, không phải
cửa sổ `cv2.imshow` (Pi bay không có màn hình).

---

## File

| file | vai trò |
|---|---|
| `person_follow_reid.py` | **chương trình chính**: bám 1 người, ReID, khoảng cách, lệnh, ROS2, GUI |
| `track.py` | `Detector` (ONNX + letterbox) và `IoUTracker` (gán ID). `person_follow_reid` import từ đây. Chạy độc lập được để soi detector |
| `yolo11n-pose.onnx` | box + 17 keypoint COCO, 1 lần chạy (11 MB) |
| `osnet_x0_25.onnx` | ReID 512-D, để nhận lại đúng người sau khi mất (870 KB) |
| `yolov8n-drone.onnx` | model drone gốc của bạn, `track.py` mặc định dùng nó (không liên quan luồng follow) |
| `run_gui.py` | **cách chạy trên Pi**: lấy khung từ webcam (gst) hoặc topic ROS, gọi `person_follow_reid.main()`, phát MJPEG ra 8080. Không sửa một dòng nào của `person_follow_reid.py` |
| `mjpeg-stream.service` | unit systemd đang chạy trên Pi (`/etc/systemd/system/`) |
| `Dockerfile.gst` | build `drone-vision:humble-gst` = image cũ + gstreamer |

---

## Cài

Trên **máy dev** (có màn hình), để chạy trần và xem cửa sổ:

```bash
pip install opencv-python onnxruntime numpy      # KHÔNG phải -headless, xem bẫy dưới
```

Bản đã kiểm: onnxruntime 1.23.2, OpenCV 4.8.1, numpy 1.26.4.

Tuỳ chọn (thiếu vẫn chạy, tự degrade và in cảnh báo):
- `rclpy` — không có thì chỉ in lệnh ra console, không publish.
- `pymavlink` — không có thì coi roll = pitch = 0 (camera luôn ngang).

⚠️ **Bẫy đã cắn thật trên Pi:** gói cài sẵn ở đó là `opencv-python-headless`. Bản
này build **không có V4L2 lẫn GUI**, nên `cv2.VideoCapture("/dev/video0")` trả về
`False` và `cv2.imshow` chết — mà **không có một dòng lỗi nào** nói vì sao. Kiểm
bằng `cv2.getBuildInformation()`: không thấy dòng `V4L/V4L2` nghĩa là dính.
Trên Pi ta không sửa bằng cách cài lại OpenCV, mà đi vòng qua gstreamer (xem dưới).

## Chạy trên Pi 5 (cách đang dùng)

Chạy trong container `drone-cam` do systemd quản, **không** chạy trần:

```bash
sudo systemctl restart mjpeg-stream      # khởi động lại
journalctl -u mjpeg-stream -f            # xem log
```

Xem ảnh: mở GUI_NATIVE (nó tự suy `http://<ip-pi>:8080/stream` từ `connections.yaml`),
hoặc `curl http://<ip-pi>:8080/stream`.

Vì sao phải trong container, ba lý do độc lập:
- `rclpy` chỉ có trong đó → `/follow/command` mới publish thật thay vì chỉ in.
- OpenCV trên host là bản headless không có V4L2 (xem bẫy ở trên).
- Cổng 8080 đã là hợp đồng với GUI_NATIVE, giữ nguyên thì GUI không sửa dòng nào.

`run_gui.py` có **hai nguồn khung**, cùng một đầu ra:

```bash
python3 run_gui.py --device /dev/video0     # gst đọc MJPG thẳng từ webcam (ĐANG DÙNG)
python3 run_gui.py --topic /image_raw       # subscriber ROS, cần v4l2_camera_node chạy kèm
```

Chọn `--device` vì webcam nén MJPG sẵn trong phần cứng, chỉ còn `jpegdec`. Đường
`--topic` phải qua `v4l2_camera_node`, mà node đó chỉ làm được YUYV (Humble chưa
giải mã MJPG) — **đo thật: YUYV 720p tụt xuống 0.6–0.8 Hz** khi ONNX đang chạy,
tức khung cũ 1.5 giây. Giữ `--topic` lại cho lúc nguồn là camera Gazebo.

**Hình đi theo nhịp model, không tách.** Chỉ phát khung pf đã vẽ xong nên hộp/xương
khớp 100% với hình. Đang chạy `--model yolo11n-pose-320.onnx --stream-every 1`:
~9 fps (bản 416 + `--stream-every 2` cũ chỉ 3.6 fps); đánh đổi là sót người ở xa.

Đã thử tách hình ra 15 fps rồi dán lớp phủ lên khung mới hơn (2026-09-15) và bỏ:
lớp phủ trễ 160–290 ms, lệch rõ khi vung tay/đi nhanh. Bù bằng optical flow (lệch
9.8 vs 10.1 px không bù) hay ngoại suy vận tốc hộp (18.2 vs 16.5 px) đều **không hơn
không bù** — ở cự ly gần hộp đổi hình dạng theo tay chân chứ không trượt đều, còn
cửa sổ optical flow ở mép người ôm cả nền đứng yên. Muốn hình nhanh hơn mà vẫn khớp
thì phải làm model nhanh hơn, không phải bù lớp phủ.

## Chạy trên máy dev

```bash
python3 person_follow_reid.py --source 0          # camera, có cửa sổ cv2
python3 person_follow_reid.py --source vid.mp4    # file
python3 person_follow_reid.py --selftest          # kiểm logic, không cần camera
```

Phím trong cửa sổ: **q** thoát · **r** bỏ khóa mục tiêu (khóa lại người to nhất).
Chạy qua `run_gui.py` thì **không có cửa sổ nên hai phím này vô hiệu** — muốn bỏ
khóa mục tiêu thì `systemctl restart mjpeg-stream`.

Cờ hay dùng: `--conf 0.4` · `--threads 4` · `--width 1280 --height 720` ·
`--model` / `--reid` để đổi đường dẫn model.

---

## Luồng xử lý

```
frame
 └─ yolo11n-pose.onnx  ──> box + 17 keypoint          (NMS đã nằm trong model)
     └─ IoUTracker     ──> ID giữa các frame
         └─ osnet ReID ──> khóa/bám/nhận lại đúng 1 người
             ├─ classify_view   FULL / BODY / HEAD / NONE
             ├─ person_center   tâm bám (luôn nằm trong bbox)
             ├─ CenterFilter    ổn định tâm, trễ 0
             ├─ estimate_distance  Z mét (mono, bù roll/pitch)
             └─ decide_command  FOLLOW / BACK_OFF / SEARCH / IDLE
```

Model pose cho cả box lẫn keypoint trong một lần chạy, nên **không có nhánh detect
riêng** — không cần ghép 2 nhánh bằng IoU như bản chạy trên máy có GPU.

### Tâm bám — phần đã tốn nhiều công nhất

Ba quy tắc, mỗi cái sinh ra từ một lỗi thật đo được trên camera:

1. **Không ngưỡng cứng.** Tâm = `w × (tâm thân đo từ vai+hông) + (1−w) × (trọng tâm
   phần nhìn thấy được)`, `w` trượt liên tục theo conf hông (`HIP_BLEND`). Cắt cứng ở
   `conf > 0.5` làm tâm nhảy mỗi khi hông nhấp nháy qua ngưỡng.
2. **Không bao giờ ngắm ra ngoài vùng thấy được.** `clamp_to_box()` kẹp tâm vào trong
   bbox. Bản ngoại suy theo nhân trắc học từng đẩy tâm xuống y=990 trong khung cao 720
   — **100% số frame nằm ngoài ảnh** — vì model pose đoán vai ở dưới mép ảnh.
3. **Neo + lọc độ lệch, không lọc cả tâm** (`CenterFilter`). Tâm = *neo* + *độ lệch*.
   Neo là điểm tin cậy nhất (mặt: nhiễu ~1.1 px; vai: ~8.9 px) nên bám người tức thì;
   chỉ độ lệch — biến thiên chậm — mới bị EMA dập.

Đo trên 150 frame thật + chuyển động ramp đã biết trước:

| cách tính tâm | nhiễu | trễ |
|---|---|---|
| thô | 20.8 px | 0 px |
| EMA thẳng lên tâm (a=0.9) | 1.5 px | **35 px** ← mượt nhưng vô dụng cho điều khiển |
| **neo + EMA độ lệch (đang dùng)** | **1.2 px** | **0 px** |

Đổi neo (mặt → vai khi người quay lưng) được quy độ lệch về neo mới nên đầu ra liên tục.

---

## Giao diện debug

- **Khung xanh + `id`** — mọi người đang được track. Khung dày = mục tiêu đang khóa.
- **Skeleton xanh** — 17 keypoint của mục tiêu.
- **Chấm xám nhỏ** — tâm thô, trước lọc. **Chấm đỏ** — tâm đã ổn định (thứ được publish).
  Đặc = tâm thân đo trực tiếp, rỗng = có phần suy ra. Lắc nhẹ đầu: xám nhảy, đỏ đứng yên.
- **Đường đỏ + dấu `+` vàng** — độ lệch từ tâm khung hình tới mục tiêu.
- **HUD trên**: `FPS · infer ms · det` (số detection) `· live` (số track) `· views` (số ảnh trong gallery ReID).
- **HUD dưới**: `CMD` — xanh FOLLOW, cam BACK_OFF, đỏ SEARCH, xám IDLE.
- Dòng giữa: `view=` tầm nhìn · `center=` nguồn tâm (`torso` / `visible` / `mixN` / `face` / `bbox`) · `neo=` điểm neo · `ang_x/ang_y`.

`SEARCHING... lost=N` nghĩa là từng khóa được mục tiêu và đang tìm lại; quá
`max_lost=90` frame thì bỏ khóa hẳn.

---

## Giao tiếp ROS2

Publish **mỗi frame** (QoS mặc định, RELIABLE depth 10):

| topic | kiểu | nội dung |
|---|---|---|
| `/follow/command` | `std_msgs/String` | `FOLLOW` \| `BACK_OFF` \| `SEARCH` \| `IDLE` |
| `/follow/target` | `geometry_msgs/Vector3` | `x` = ang_x (độ, + là người lệch PHẢI) · `y` = ang_y (độ, + là DƯỚI đường chân trời) · `z` = mét, **`z ≤ 0` nghĩa là không có khoảng cách** |

Ý nghĩa lệnh: `FOLLOW` bám bình thường · `BACK_OFF` quá gần hoặc chỉ thấy đầu, lùi ra ·
`SEARCH` đang mất mục tiêu, hold + quét, **không tiến** · `IDLE` chưa khóa ai.

Console chỉ in khi lệnh **đổi**, không phải mỗi frame.

---

## Cần chỉnh trước khi bay

Ở đầu `person_follow_reid.py`:

| hằng số | mặc định | ghi chú |
|---|---|---|
| `CAM_INDEX` | 0 | `v4l2-ctl --list-devices` để kiểm |
| `HFOV_DEG` | 70.0 | **góc nhìn ngang camera thật** — sai cái này là `ang_x/ang_y/Z` sai hết |
| `ADULT_HEIGHT_M` | 1.70 | chiều cao người được bám |
| `STANDOFF_MIN_M` | 1.5 | dưới ngưỡng này → `BACK_OFF` |
| `MAV_CONNECT` | `None` | vd `"/dev/serial0"` hoặc `"udp:127.0.0.1:14550"` để bù roll/pitch |

`CENTER_OFFSET_EMA` và `VIEW_STICKY_N` **không còn phải chỉnh tay** — xem mục dưới.

**Kiểm dấu 1 lần sau khi nối MAVLink**: nghiêng/chúi drone bằng tay trước một vật cố
định, `ang_x/ang_y/Z` phải **giữ nguyên**. Lệch ngược thì đổi dấu roll hoặc pitch trong
`read_attitude()`.

### Ngưỡng đếm bằng GIÂY, không bằng frame

Mọi ngưỡng thời gian trước đây đếm bằng **frame** và được chỉnh trên laptop 22 fps.
Pi chạy 2.5–3.6 fps nên cùng con số đó dài ra **6–9 lần** tính theo giây — lỗi im
lặng, không có gì báo:

| ngưỡng | frame (cũ) | ở 22 fps | ở 3.6 fps | giờ ở 3.6 fps |
|---|---|---|---|---|
| `max_lost` bỏ khóa mục tiêu | 90 | 4.1 s | **25 s** | 14 frame = 3.9 s |
| `max_age` xóa track | 15 | 0.7 s | **4.2 s** | 3 frame = 0.83 s |
| median `Z` | 7 | 0.32 s | **1.9 s** | 3 frame = 0.83 s |
| `view` sticky | 3 | 0.14 s | **0.83 s** | 2 frame = 0.55 s |
| EMA tâm `a` | 0.9 | tau 0.43 s | **tau 2.7 s** | a = 0.52, tau 0.43 s |

Drone `SEARCH` 25 giây mới chịu nhả mục tiêu, và `Z` bạn nhận là trung vị của 1.9
giây trước — nguy hiểm hơn hẳn chuyện fps thấp.

Nên giờ khai báo bằng giây (`LOST_S`, `TRACK_AGE_S`, `DIST_MED_S`, `VIEW_STICKY_S`,
`CENTER_TAU_S`) rồi `_tune_for_fps()` quy sang frame theo nhịp **đo được lúc chạy**.
Giá trị giây chọn sao cho ở 22 fps ra **đúng con số cũ** — chạy trên laptop hành vi
không đổi.

Ba chi tiết không hiển nhiên:

- Gọi **mỗi frame**, không một lần lúc khởi động. Nhịp đổi thật trong lúc bay: đo
  được `det 2 → infer 242 ms` nhưng `det 6 → infer 320 ms` (ReID chạy nhiều crop
  hơn). Ngưỡng tính theo nhịp cũ sẽ sai đúng vào lúc cảnh đông người.
- EMA quy theo `a = e^(-dt/tau)`, tức giữ nguyên **hằng số thời gian**, không giữ
  nguyên hệ số `a`. Giữ `a = 0.9` ở 3.6 fps là tau 2.7 giây, tâm bám trễ hẳn.
- Hai bộ lọc có **sàn cứng**: median `Z` tối thiểu 3 mẫu (median cần 3 mẫu mới bỏ
  được 1 giá trị bậy; công thức ra 1 là mất hẳn khả năng lọc outlier), `view` sticky
  tối thiểu 2 (n = 1 nghĩa là `Sticky` không còn chống nhấp nháy nữa).

---

## Hiệu năng

Đo trên x86 (laptop), 1280×720:

| | ms/frame |
|---|---|
| yolo11n-pose 640×640 | 25 |
| ReID 4 người | 3 |
| cả vòng lặp kèm vẽ | ~41 (≈22 fps) |

**Đo thật trên Pi 5** (4 core, 960×540 q70, gst MJPG, không còn `v4l2_camera_node`):

| | |
|---|---|
| `infer` khi có 2 người | 242 ms |
| `infer` khi có 6 người | 320 ms (ReID chạy 6 crop) |
| nhịp vòng lặp | **2.5–3.6 fps** |
| `/follow/command` | cùng nhịp đó, jitter ±13 ms |
| CPU | 350% / 400%, 64 °C (chưa throttle) |
| nguồn camera | 15.0 Hz đều, ~200 KB/s ra mạng |

Ước "5–6 fps" ban đầu là lạc quan. Nút thắt nằm nguyên ở ONNX chứ không ở camera
(nguồn 15 Hz thừa sức). Muốn nhanh hơn thì phải **đổi model**, chỉnh code không
được: export lại pose ở `imgsz=416` (640²/416² ≈ 2.4× → ~7–8 fps), hoặc lượng tử
hoá INT8. Model không dùng shape động (`dynamic=False`) nên **không hạ imgsz lúc
chạy được**.

### Trễ: KHÔNG tích luỹ

Ba chỗ trong đường ống đều chỉ giữ **khung mới nhất**, không xếp hàng: `GstCap._pump`
ghi đè `self.frame`, `mjpeg_server._luu` ghi đè `_latest`, `Handler` chỉ gửi khi `_seq`
đổi. Đo: ngưng đọc stream 8 giây rồi đọc lại → **0 khung dồn về**. Mạng nghẽn thì mất
khung, không phải xem quá khứ.

Trễ cố định ≈ tuổi khung (≤ 67 ms ở 15 Hz) + `infer` (~250 ms) + encode (~25 ms) +
truyền ≈ **0.35 s**, không lớn dần.

### Bitrate phải vừa cái uplink, không vừa con mắt

Đo thật giữa Pi và laptop trên cùng SSID 5 GHz, `-57 dBm`, link báo 263 Mbit/s:

| chiều | thông lượng thực |
|---|---|
| laptop → Pi | 59 Mbps |
| **Pi → laptop** | **7 Mbps**, 5% mất gói, RTT 2.7–317 ms |

Lệch 8 lần. `power_save` đã off cả hai đầu; `tx failed: 2860` trên `wlan0`; Pi và
laptop còn bám **hai AP khác nhau** (ch157 vs ch161) cùng SSID.

Hệ quả: 1280×720 q80 = 400 KB/s chiếm ~45% cái uplink hỏng đó → TCP sụp từng đợt,
client nhận **0.2 fps** với khoảng trống 22 giây trong khi Pi vẫn mã hoá 3.8 fps.
Hạ về 960×540 q70 (~200 KB/s) là client nhận đúng nhịp Pi.

Nên nếu thấy giật: **đo `nen … fps` trong log trước**. Log báo 3 fps mà client thấy
0.2 fps thì lỗi ở mạng, không phải ở model — hai thứ này sửa ở hai chỗ hoàn toàn khác
nhau. Sửa gốc thì ghim Pi vào cùng AP với laptop, hoặc cắm dây `eth0`.

## Kiểm thử

```bash
python3 person_follow_reid.py --selftest
python3 track.py --selftest
```

Không cần camera. Kiểm: 4 lớp `view`; tâm liên tục khi conf hông trượt; **tâm luôn nằm
trong bbox kể cả khi keypoint bị đoán bậy ra ngoài ảnh**; `CenterFilter` trễ 0 và dập
nhiễu ≥ 4×; đổi neo không gây nhảy; khoảng cách (gồm bù pitch 45° và bất biến roll 90°);
4 lệnh; vòng khóa → bám → ReID nhận lại → từ chối người khác; **ngưỡng theo giây quy
đúng ở cả 22 fps lẫn 3.6 fps** (cùng thời gian thật, sàn của median và sticky được
giữ, EMA ra đúng 0.9 ở 22 fps).

## Tạo lại model

```bash
# pose (từ ~/tracking_bitch hoặc ~/Downloads)
python3 -c "from ultralytics import YOLO; YOLO('yolo11n-pose.pt').export(format='onnx', imgsz=640, simplify=True, nms=True, opset=12)"

# ReID
python3 -c "
import torch, torchreid
m = torchreid.models.build_model('osnet_x0_25', num_classes=1000, pretrained=True); m.eval()
torch.onnx.export(m, torch.randn(2,3,256,128), 'osnet_x0_25.onnx', dynamo=False,
    input_names=['crops'], output_names=['feat'],
    dynamic_axes={'crops':{0:'n'},'feat':{0:'n'}}, opset_version=12)"
```

Chạy trên **máy dev**, không phải Pi (cần torch + ultralytics). `yolo11m-pose.onnx` đã
xoá khỏi folder: 152 ms/frame trên x86, trên Pi dưới 1 fps — đổi `yolo11n` thành
`yolo11m` ở lệnh trên nếu cần bản nặng cho máy có GPU.

## Hạn chế đã biết

- `IoUTracker` không có Kalman/ReID nội bộ — ID nhảy khi hai người cắt nhau hoặc drone
  xoay nhanh. `TargetTracker` + OSNet gỡ lại được, nhưng nếu ID switch gây phiền thì thay
  bằng ByteTrack/BoT-SORT.
- ReID dùng weight **ImageNet-pretrained**, không phải weight train trên tập ReID. Người
  mặc đồ giống nhau dễ nhầm. Nạp weight market1501 cho `osnet_x0_25` rồi export lại là khá hơn.
- `Z` là ước lượng mono theo giả định cao 1.70 m — sai tỷ lệ thẳng với sai số giả định đó.
- Chưa có hysteresis khi `view` nhấp nháy ngoài `Sticky` đếm frame.
- **`HFOV_DEG = 70.0` chưa hiệu chỉnh cho con camera thật.** Khung giờ là 16:9
  (960×540) chứ không phải 4:3 như lúc đặt hằng số này, nên càng lệch. Đo thấy
  `Z = 0.71 m` cho người ngồi cách ~1 m. **Đừng nối `z` vào điều khiển bay trước khi
  đo lại HFOV** — `ang_x/ang_y/Z` sai hết theo cùng một hằng số này.
- Chạy qua `run_gui.py` thì mất phím `q`/`r`. Bỏ khóa mục tiêu hiện phải restart
  service; muốn tử tế thì thêm một route HTTP `/reset` vào `Handler`.
- Mất topic `/image_raw` từ khi bỏ `v4l2_camera_node`. Chưa ai dùng nó (GUI chỉ đọc
  HTTP 8080), nhưng node nào cần ảnh qua ROS thì phải quay lại đường `--topic`.
