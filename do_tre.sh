#!/bin/bash
# Chay o TRAM MAT DAT: lay script tu drone, chay h264_feedback nen + do tre. Ctrl+C tat ca hai.
D=robomango@100.117.32.3
mkdir -p ~/lat && cd ~/lat || exit 1
scp $D:jetson/latency_stamp.py $D:jetson/h264_feedback.py . || exit 1
python3 h264_feedback.py --drone 100.117.32.3 >/dev/null &
trap "kill $!" EXIT
python3 latency_stamp.py udp://@:5610 $D
