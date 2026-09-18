# drone-vision:humble — chay tren Pi5

Image xu ly anh cho drone: ROS 2 Humble + MAVROS + OpenCV + ONNX Runtime.
Build tu `ardu_ros:rviz-camera-ready`, dinh nghia o `Dockerfile` canh file nay.

Trong image co gi:

| | |
|---|---|
| ROS 2 | Humble (Ubuntu 22.04, Python 3.10) |
| MAVROS | `mavros`, `mavros_msgs` + dataset geographiclib `egm96-5` |
| Camera | `v4l2_camera`, `cv_bridge`, `image_transport` |
| Python | `numpy 1.26.4`, `cv2 4.11.0` (headless), `onnxruntime 1.23.2` |

Dung `opencv-python-headless` chu khong phai `python3-opencv` cua apt: apt keo
theo X11/GTK/gstreamer, vo ich tren drone khong man hinh (~300 MB thua).

`numpy` ghim `<2`. `rclpy` va `cv_bridge` trong image bien dich voi numpy 1.x,
tha cho onnxruntime keo numpy 2.x len la gay toan bo ROS Python.

## Chay

Mo container (camera + mang host + mount code tu home):

```bash
docker run --rm -it \
  --device /dev/video0 --group-add video \
  --network host \
  -v ~/mjpeg_server.py:/app/mjpeg_server.py \
  drone-vision:humble bash
```

- `--group-add video` — thieu cai nay thi container mo `/dev/video0` bao Permission denied.
- `--network host` — DDS va cong 8080 dung chung mang voi Pi, khoi map tung cong.
- Them `--device /dev/ttyAMA0` khi can noi flight controller (xem phan MAVROS).

Trong container, ROS chua duoc source san:

```bash
source /opt/ros/humble/setup.bash
```

## Camera + stream cho GUI

```bash
ros2 run v4l2_camera v4l2_camera_node --ros-args \
    -p pixel_format:=YUYV -p image_size:="[640,480]" &
python3 /app/mjpeg_server.py            # xem tai http://192.168.1.113:8080/stream
python3 /app/mjpeg_server.py --every 2  # 15 fps, ~1 MB/s cho duong WiFi
```

Do duoc tren Rapoo/Pi5: `/image_raw` 30,0 fps (p50 33ms, max 37ms), qua HTTP
29,9 fps / 2089 KB/s, `mjpeg_server` an 33% mot core.

**YUYV chu khong phai MJPG.** `v4l2_camera` ban Humble chua giai ma duoc MJPG:
in `Current pixel format is not supported yet: MJPG` roi nem
`cv_bridge::Exception` va chet han. YUYV nhanh hon (30 fps so voi 16 fps cua
duong MJPG cu), doi lai USB2 tai 18 MB/s tho.

**Webcam USB chi cho MOT tien trinh STREAMON.** `v4l2_camera_node` la chu duy
nhat cua `/dev/video0`; `mjpeg_server.py` va node ONNX deu doc topic `/image_raw`.
Dung mo device lan hai — tien trinh thu hai chi nhan duoc loi.

Kiem tra nhanh:

```bash
ros2 topic list                     # /image_raw, /camera_info
ros2 topic hz /image_raw            # ky vong ~30
```

Topic im ru ma khong bao loi gi? Gan nhu chac la QoS: `v4l2_camera` publish
BEST_EFFORT, subscriber de mac dinh RELIABLE thi hai ben khong khop va khong co
khung nao ve — khong loi, khong canh bao. Dung `qos_profile_sensor_data`.

## MAVROS

Chua thu duoc: luc dung image nay chua co flight controller nao cam vao Pi
(khong thay `/dev/ttyUSB*` hay `/dev/ttyACM*`). Duong day noi la `/dev/ttyAMA0`
(UART tren chan GPIO), phai truyen vao container:

```bash
docker run --rm -it --device /dev/ttyAMA0 --group-add dialout --network host \
  drone-vision:humble bash

ros2 launch mavros apm.launch fcu_url:=/dev/ttyAMA0:921600   # ArduPilot
ros2 topic echo /mavros/state                                # kiem tra ket noi
```

`px4.launch` neu chay PX4. Doi `apm` <-> `px4` la doi bo plugin, khong phai doi
mot tham so.

`mavros-extras` (plugin `vision_pose_estimate`, `landing_target`,
`obstacle_distance`) **chua cai**. Can khi day pose tu vision xuong FC hoac lam
precision landing:

```bash
# them dong nay vao Dockerfile roi build lai, layer khac van con cache
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-mavros-extras && rm -rf /var/lib/apt/lists/*
```

## ONNX Runtime

Chi co CPU: providers = `['AzureExecutionProvider', 'CPUExecutionProvider']`.
Pi5 khong co GPU/NPU cho onnxruntime dung. Canh bao
`GPU device discovery failed` luc import la vo hai, wheel nao cung in.

```python
import onnxruntime as ort
so = ort.SessionOptions()
so.intra_op_num_threads = 4        # Pi5 co 4 core
sess = ort.InferenceSession("model.onnx", so, providers=["CPUExecutionProvider"])
```

Don bay lon nhat la **quantize model sang INT8**, khong phai provider.

## Build lai

```bash
cd ~/drone-img && docker build -t drone-vision:humble .
```

Layer xep theo do on dinh nen sua dong cuoi khong keo lai `mavros`:
apt ROS → dataset geographiclib → pip → `v4l2_camera`.

Log build ra file, dung `| tail`: buildkit khong xa ra tung dong nen `tail` nuot
het den luc xong.

```bash
docker build -t drone-vision:humble . > /tmp/build.log 2>&1
```

## Khi camera khong mo duoc

```bash
fuser -v /dev/video0    # ai dang giu
```

`pipewire` giu fd la binh thuong, no khong STREAMON nen khong can. Neu thay
`mjpeg_server.py` ban cu (`~/mjpeg_server.py.v4l2.bak` — ban tu mo
`/dev/video0`) thi tat no di, gio chu device la `v4l2_camera_node`.
