# source trong ExecStart cua drone-ros va mjpeg-stream: chon CYCLONEDDS_URI MOT lan moi lan boot,
# moi unit (va moi lan Restart) doc chung file -> node cua hai service luon cung mang DDS.
# 04/11/2026: truoc day ExecStartPre cho tailscale0 VO HAN -> 4G cam muon 18 phut = MAVROS/lua tre
# 18 phut; ngoai bai mat 4G/Tailscale = khong bao gio chay. Gio cho toi 60 s, khong co thi DDS chi
# noi bo Jetson (lo): MAVROS/lua/bridge van chay, tablet van noi qua WebSocket 8765 + MAVLink UDP khi
# Tailscale len sau. Chi tram mat dat khong thay topic ROS toi lan boot sau, hoac:
#   rm /tmp/dds-uri.*; sudo systemctl restart mjpeg-stream drone-ros
# ros2 CLI tren Jetson: export CYCLONEDDS_URI=$(cat /tmp/dds-uri.$(cat /proc/sys/kernel/random/boot_id))
f=/tmp/dds-uri.$(cat /proc/sys/kernel/random/boot_id)
if [ ! -s "$f" ]; then
  uri=file:///home/robomango/jetson/cyclonedds_local.xml
  for i in $(seq 30); do
    if ip -4 addr show dev tailscale0 2>/dev/null | grep -q inet; then
      uri=file:///home/robomango/jetson/cyclonedds_tailscale.xml; break
    fi
    [ -s "$f" ] && break   # unit kia da chon
    sleep 2
  done
  # ln khong -f: nguyen tu, unit nao ghi truoc thi thang, file luon day du noi dung
  echo "$uri" > "$f.$$" && ln "$f.$$" "$f" 2>/dev/null; rm -f "$f.$$"
fi
export CYCLONEDDS_URI=$(cat "$f")
echo "DDS: $CYCLONEDDS_URI"
