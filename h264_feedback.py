#!/usr/bin/env python3
"""Chay o TRAM MAT DAT: nhan H.264 MPEG-TS/UDP tu drone, dem goi mat, bao ve drone.

    drone :5600 -> [h264_feedback.py] -> 127.0.0.1:5610 -> GUI / ffplay
                         |
                         +-- "loss=0.0123" moi giay -> drone :5601 (run_gui --h264-fb-port 5601)

GUI doi cong nghe tu 5600 sang 5610 (hoac --out). Chi dung stdlib.

    python3 h264_feedback.py --drone 100.117.32.3
    ffplay -fflags nobuffer udp://127.0.0.1:5610      # xem thu

Mat goi tinh theo continuity counter 4 bit cua TS (moi PID): lech = so goi mat.
ponytail: CC chi 4 bit -> mat >15 goi lien tiep tren mot PID bi dem thieu (mod 16);
van du de thay "dang mat" va ha bitrate. Can chinh xac hon thi chuyen sang RTP (seq 16 bit).
"""
import argparse, random, socket, time


def count_loss(data, last):
    """(goi nhan, goi mat) trong mot datagram TS; `last` = {pid: cc} giu giua cac lan."""
    got = lost = 0
    for i in range(0, len(data) - 187, 188):
        if data[i] != 0x47:
            continue
        pid = ((data[i + 1] & 0x1F) << 8) | data[i + 2]
        if pid == 0x1FFF or not data[i + 3] & 0x10:      # null / khong co payload: CC dung yen
            continue
        cc = data[i + 3] & 0x0F
        got += 1
        if pid in last and cc != last[pid]:               # cc trung = goi lap, hop le
            lost += (cc - last[pid] - 1) & 0x0F
        last[pid] = cc
    return got, lost


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--drone", required=True, help="IP Tailscale cua drone, vd 100.117.32.3")
    p.add_argument("--fb-port", type=int, default=5601)
    p.add_argument("--listen", type=int, default=5600)
    p.add_argument("--out", default="127.0.0.1:5610", help="chuyen tiep video toi GUI")
    p.add_argument("--drop", type=float, default=0.0, help="THU NGHIEM: bo ngau nhien ti le nay")
    a = p.parse_args()

    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("0.0.0.0", a.listen))
    rx.settimeout(1.0)
    tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    oh, op = a.out.rsplit(":", 1)
    out, fb = (oh, int(op)), (a.drone, a.fb_port)

    last, got, lost, t0 = {}, 0, 0, time.monotonic()
    while True:
        try:
            data = rx.recv(2048)
            if not (a.drop and random.random() < a.drop):
                tx.sendto(data, out)
                g, l = count_loss(data, last)
                got, lost = got + g, lost + l
        except socket.timeout:
            pass
        now = time.monotonic()
        if now - t0 >= 1.0:
            if got:                      # khong nhan gi thi im lang; drone tu hieu la mat het
                loss = lost / (got + lost)
                tx.sendto(f"loss={loss:.4f}".encode(), fb)
                print(f"{got + lost:5d} goi  mat {loss:6.1%}", flush=True)
            got, lost, t0 = 0, 0, now


if __name__ == "__main__":
    main()
