# pi5 — máy tính nhúng của drone

Mọi thứ chạy trên Raspberry Pi 5 gắn trên drone: đọc webcam, bám người, phát MJPEG
ra cổng 8080 cho trạm mặt đất [`GUI_NATIVE`](https://github.com/phantrungtien/GUI_Groundstation).

Repo này **soi gương thư mục `/home/hoaibac` trên Pi**: file ở gốc repo là file ở gốc
`~`, thư mục con giữ nguyên tên. Đường dẫn trong `systemd/mjpeg-stream.service` là
tuyệt đối (`/home/hoaibac/...`), nên đặt sai chỗ là service không lên.

## Cần gì để chạy

Ba thứ **không nằm trong repo này**, phải có sẵn trên máy:

| Thứ | Ở đâu | Vì sao để ngoài |
|---|---|---|
| `~/phantrungtien-fire-detection` | [github.com/phantrungtien/fire-detection](https://github.com/phantrungtien/fire-detection) | Đã có repo riêng. Service mount `models/` của nó vào `/app/fire-models` |
| `~/tien_drone_ws` | workspace ROS 2 (có `.git` cục bộ, chưa có remote) | Chưa đẩy lên đâu cả — xem *Còn nợ* ở cuối |
| image `drone-vision:humble-gst` | build tại chỗ, xem `drone-img/` rồi `Dockerfile.gst` | Image Docker, không bỏ vào git được |

## Dựng lại từ đầu

```bash
git clone https://github.com/phantrungtien/pi5.git
cp -r pi5/{*.py,*.sh,*.xml,Dockerfile.gst,restore_drone_cam.txt} ~/
cp -r pi5/tracking_pi5 pi5/drone-img ~/

# 1. image nen: ROS 2 Humble + MAVROS + OpenCV + ONNX Runtime
cd ~/drone-img && docker build -t drone-vision:humble .
# 2. them gstreamer (doc thang MJPG tu webcam, khoi chuyen mau bang CPU)
cd ~ && docker build -t drone-vision:humble-gst -f Dockerfile.gst .
# 3. models cho phan lua
git clone https://github.com/phantrungtien/fire-detection.git ~/phantrungtien-fire-detection

sudo cp pi5/systemd/mjpeg-stream.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now mjpeg-stream
```

Kiểm tra: `docker logs -f drone-cam` phải ra dòng như

```
[mjpeg] nguon  15.0 Hz  ->  nen  12.2 fps,   1153 KB/s, yolo 167 ms
```

`nguon` là nhịp camera đẩy khung vào, `nen` là nhịp phát ra HTTP. **Hai con số này
khác nhau là bình thường; `nguon` bằng 0 mà `nen` vẫn chạy thì camera đã chết** —
xem *Bẫy đã gặp* bên dưới.

## Có gì trong đây

### Gốc repo — file rời ở `~`

| Tệp | Việc |
|---|---|
| `mjpeg_server.py` | Máy chủ HTTP MJPEG cổng 8080, đếm nhịp, giữ khung mới nhất |
| `fire_tracker.py` | Phát hiện lửa/khói bằng ONNX, chạy song song ở nhịp riêng (`--fire-hz`) |
| `fire_stream.sh` | Chạy tay luồng lửa, không qua systemd |
| `to5g.sh` / `to2g.sh` | Ép WiFi sang băng 5 GHz / 2,4 GHz. Băng tần quyết định fps hơn mọi thứ khác — đo được 2,5 fps ở 2,4 GHz so với 16 fps ở 5 GHz |
| `fastdds_udp.xml` | Cấu hình DDS cho ROS 2 qua UDP |
| `Dockerfile.gst` | Thêm GStreamer lên trên `drone-vision:humble` |
| `pi_probe.py`, `do_fps.py`, `luu_khung.py` | Đo: thông tin máy, nhịp khung, lưu khung ra đĩa |
| `restore_drone_cam.txt` | Ghi chú dựng lại container bằng tay |

### `tracking_pi5/` — bám người

`person_follow_reid.py` là phần chính (YOLO pose + ReID). `run_gui.py` bọc nó lại cho
container: thay `cv2.VideoCapture` bằng `GstCap`/`RosCap`, thay `cv2.imshow` bằng
nén JPEG đẩy sang `mjpeg_server`. `follow_control.py` sinh lệnh bám, `track.py` là
bộ theo vết. Các tệp `_*.py` là script đo lặt vặt (int8, spin, kích thước, độ chính xác).

### `drone-img/` — image Docker nền

`Dockerfile` + README mô tả image `drone-vision:humble`.

### `systemd/` — service

`mjpeg-stream.service`, `Restart=always`. Chính chỗ này chọn `--model yolo11n-pose-320.onnx`
và `--stream-every 1`; phần chú thích trong file giải thích vì sao 320 chứ không 416.

## Model

Chỉ **ba** model vào git, đúng những cái code nạp thật:

| Tệp | Ai nạp |
|---|---|
| `yolo11n-pose-320.onnx` | service, qua `--model` |
| `yolo11n-pose-416.onnx` | mặc định `person_follow_reid.POSE_ONNX` khi không truyền `--model` |
| `osnet_x0_25.onnx` | ReID 512-D |

Các biến thể 224/256/288/`-int8`/`yolov8n-drone` chỉ dùng để đo so sánh và **đã loại**
khỏi git (xem `.gitignore`) — chúng nặng 12 MB mỗi cái và xuất lại được từ nguồn
gốc bằng `_sizes.py` / `_int8.py`.

Model lửa `best_nano_111_320.onnx` nằm ở repo `fire-detection`, không ở đây.

## Bẫy đã gặp — đọc trước khi mất một buổi

**Nguồn video đứng hình mà mọi chỉ số vẫn đẹp.** 18/09/2026: webcam UVC (Microdia
`0c45:636b`) ngừng đẩy khung lúc 15:07:37. Nhân **không ghi một dòng `dmesg` nào**,
thiết bị vẫn liệt kê trên USB, fd vẫn mở. `v4l2src` kẹt trong `poll()` vô hạn, luồng
đọc kẹt theo trong `stdout.read()`, và `GstCap.read()` trả về khung cuối *không kèm
dấu thời gian* — nên máy chủ nén lại đúng khung đó và phát ra 11,6 fps hoàn toàn hợp
lệ. Đo thẳng vào luồng: **12 khung trong 1,03 s trùng khít từng byte** (cùng md5, cùng
68 430 byte). Giao diện trạm mặt đất cũng không phát hiện được, vì nó canh khoảng ngắt
giữa hai khung chứ không so nội dung.

Đã vá trong `tracking_pi5/run_gui.py`:

- mỗi khung đóng dấu thời gian lúc đọc ra khỏi ống
- khung quá **2 s** thì `show()` **thôi phát** — luồng HTTP im hẳn, đó là thứ trạm mặt
  đất phát hiện được bằng cơ chế sẵn có
- nguồn im quá **5 s** thì luồng canh giết rồi dựng lại `gst-launch` (giết là cách duy
  nhất đánh thức `_pump` đang kẹt trong `read()`)
- hạn cho **khung đầu** tách riêng và dài hơn (20 s), vì MJPG mất ~8 s thương lượng USB
  — đặt chung một hạn thì luồng canh giết gst mỗi giây trước khi nó kịp nói

Chặn ở chỗ **phát** chứ không cho `read()` trả thất bại, vì `person_follow_reid` làm
`if not ok: break` — trả thất bại ở đó là giết cả chương trình.

**Camera treo cứng thì phần mềm không gỡ được.** Cùng sự cố trên, sau khi camera chết
thì cả ba cách đều thua: khởi động lại container (systemd tự làm 346 lần, đều chết vì
`--device /dev/video0` trỏ vào node không còn), unbind/bind cổng USB, và `authorized`
0→1. Nhân trả `can't set config #1, error -71` mỗi lần. **Rút ra cắm lại là cách duy
nhất**, và cắm sang cổng khác thì lên ngay.

**Kiểm tra camera sống thật hay sống giả** — đọc vài khung rồi so mã băm, đừng tin fps:

```bash
python3 - <<'EOF'
import urllib.request, hashlib, time
r = urllib.request.urlopen("http://localhost:8080/stream", timeout=10)
buf = b""; h = []; t0 = time.time()
while len(h) < 14 and time.time() - t0 < 18:
    buf += r.read(4096)
    while (a := buf.find(b"\xff\xd8")) >= 0 and (b := buf.find(b"\xff\xd9", a+2)) >= 0:
        h.append(hashlib.md5(buf[a:b+2]).hexdigest()); buf = buf[b+2:]
print(f"khung khac nhau: {len(set(h))} / {len(h)}")   # bang 1 la camera da chet
EOF
```

**Địa chỉ IP đổi theo DHCP.** Đã đổi thật hai lần trong một ngày (`.40` → `.105` → `.28`),
và hôm đó trạm mặt đất mất camera trong khi luồng trên Pi vẫn chạy tốt. Dùng tên máy
`hoaibac-desktop.local` (mDNS/avahi) chứ đừng ghi cứng IP — đặt DHCP reservation trên
router thì ra bãi bay đổi router khác là làm lại.

**WiFi băng 2,4 GHz giết fps.** 2,5 fps ở 2,4 GHz so với 16 fps ở 5 GHz. `to5g.sh` có
sẵn để ép băng. Suy hao khoảng cách cũng đủ làm tụt: đo được tốc độ liên kết rơi từ
175 Mbit/s (−64 dBm) xuống 7,2 Mbit/s (−70 dBm), và nhịp khung chuyển từ đều sang dồn
cụm (p50 23 ms nhưng p90 444 ms, đỉnh 4,77 s).

## Còn nợ

- **Đường hồi phục của luồng canh chưa được quan sát chạy thành công.** Nó mới nổ đúng
  một lần, và lần đó camera chết hẳn nên không thể phục hồi. Nhánh "giết gst → dựng lại
  → có khung trở lại" chưa có bằng chứng. Thử: `pkill gst-launch-1.0` rồi xem log có
  quay về `nguon 15.0 Hz` không.
- **USB autosuspend vẫn bật** (`power/control = auto`, ngưỡng 2 s) ở cổng camera đang
  dùng. Đây là *nghi can* gây treo webcam — dmesg lúc hỏng có `Failed to suspend device,
  error -32` lặp lại — nhưng **chưa chứng minh được**. Tắt bằng
  `echo on > /sys/bus/usb/devices/<cong>/power/control`.
- **`~/tien_drone_ws` chưa có remote.** Nó có `.git` cục bộ, sống duy nhất trên thẻ nhớ
  của Pi. Mất thẻ là mất.
