#!/bin/bash
# Bo khoa band de card bam con 2.4GHz manh thay vi 5GHz yeu dang bgscan ban moi
# 30s. Neu 30s khong noi lai duoc thi TU LUI ve band a, de khong mat Pi khoi mang.
#
# KHONG ghi cung ten profile nua. Ghi cung "EB4" nghia la doi WiFi la phai sua
# script — va luc can no nhat (dang o mang la, link toi) thi no lai la cai sai.
# Ten profile hoi nmcli, thu dang cam trong tay thi khong phai di tim.
exec >>/tmp/to2g.log 2>&1
echo "=== $(date +%H:%M:%S) bat dau ==="

DEV=${1:-wlan0}
CONN=$(nmcli -t -f NAME,DEVICE connection show --active | awk -F: -v d="$DEV" '$2==d{print $1; exit}')
if [ -z "$CONN" ]; then
    echo "KHONG co ket noi nao dang chay tren $DEV — khong lam gi"
    exit 1
fi
BAND_CU=$(nmcli -g 802-11-wireless.band connection show "$CONN")
echo "profile: $CONN   band cu: ${BAND_CU:-<trong>}"

nmcli connection modify "$CONN" 802-11-wireless.band ""
nmcli connection up "$CONN"
for i in $(seq 1 30); do
    sleep 1
    state=$(nmcli -t -f GENERAL.STATE dev show "$DEV" | cut -d: -f2)
    case "$state" in *connected*) echo "len duoc sau ${i}s"; break;; esac
done

freq=$(iw dev "$DEV" link | awk '/freq:/{print $2}')
echo "freq hien tai: ${freq:-<khong co>}"
if [ -z "$freq" ]; then
    echo "KHONG noi duoc gi -> lui ve band ${BAND_CU:-a}"
    nmcli connection modify "$CONN" 802-11-wireless.band "${BAND_CU:-a}"
    nmcli connection up "$CONN"
fi
iw dev "$DEV" link | grep -E "freq|signal|bitrate"
ip -4 addr show "$DEV" | grep inet
echo "=== xong ==="
