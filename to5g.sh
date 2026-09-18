#!/bin/bash
# Chuyen EB4 sang 5 GHz. Neu 25s khong noi lai duoc thi TU LUI ve tu dong,
# de khong mat Pi khoi mang nhu luc 22:50.
exec >>/tmp/to5g.log 2>&1
echo "=== $(date +%H:%M:%S) bat dau ==="
nmcli connection modify EB4 802-11-wireless.band a
nmcli connection up EB4
for i in $(seq 1 25); do
    sleep 1
    state=$(nmcli -t -f GENERAL.STATE dev show wlan0 | cut -d: -f2)
    case "$state" in *connected*) echo "len duoc sau ${i}s"; break;; esac
done
freq=$(iw dev wlan0 link | awk '/freq:/{print $2}')
echo "freq hien tai: $freq"
if [ -z "$freq" ] || [ "${freq%%.*}" -lt 3000 ]; then
    echo "KHONG len duoc 5GHz -> lui ve tu dong"
    nmcli connection modify EB4 802-11-wireless.band ""
    nmcli connection up EB4
fi
iw dev wlan0 link | grep -E "freq|signal|bitrate"
echo "=== xong ==="
