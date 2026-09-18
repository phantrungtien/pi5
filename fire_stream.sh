#!/bin/bash
# Chay stream phat hien lua len http://<pi>:8080/stream
#   ./fire_stream.sh                          # ONNX 640 (mac dinh)
#   ./fire_stream.sh best_nano_111.pt
#   ./fire_stream.sh best_nano_111_320.onnx 0.2   # 320 kem nhay hon -> ha conf
#   ./fire_stream.sh stop
#
# Dung pidfile chu KHONG dung `pkill -f mjpeg_server`: chuoi do nam ngay trong
# dong lenh cua chinh shell dang chay, nen pkill tu giet minh truoc khi kip khoi
# dong lai. Da dinh hai lan.
set -u
MODELS=~/phantrungtien-fire-detection/models
PIDF=~/fire_stream.pid
LOG=~/fire_stream.log

[ -f "$PIDF" ] && kill "$(cat "$PIDF")" 2>/dev/null; rm -f "$PIDF"; sleep 2
[ "${1:-}" = stop ] && { echo "da dung"; exit 0; }

MODEL=${1:-$MODELS/best_nano_111_640.onnx}
case "$MODEL" in /*) ;; *) MODEL=$MODELS/$(basename "$MODEL") ;; esac
# .pt can torch/ultralytics -> venv; .onnx chi can onnxruntime -> python he thong
case "$MODEL" in *.pt) PY=~/fire-venv/bin/python ;; *) PY=python3 ;; esac
# Ban 320 mat do nhay: tren samples/house.png confidence tut 0.63 -> 0.45, va
# no sot 3/8 anh o nguong 0.35. Mac dinh ha xuong 0.2 cho rieng no, con lai 0.35.
case "${2:-}" in "") case "$MODEL" in *_320.onnx) CONF=0.2 ;; *) CONF=0.35 ;; esac ;;
                 *) CONF=$2 ;; esac

cd ~ || exit 1
nohup "$PY" ~/mjpeg_server.py --device /dev/video0 --width 960 --height 540 \
    --every 2 --quality 70 --detect "$MODEL" --conf "$CONF" > "$LOG" 2>&1 &
echo $! > "$PIDF"
echo "chay $(basename "$MODEL") conf=$CONF bang $PY (pid $(cat "$PIDF")) -> http://$(hostname -I | awk '{print $1}'):8080/stream"
