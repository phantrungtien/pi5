#!/usr/bin/env python3
"""Tien trinh con cua run_gui.GstCap tren Jetson: nhu gst-launch, them hai thu ma gst-launch khong co:

  * zoom doi LUC DANG BAY: set_property vung cat cua nvvidconv. Do 23/10/2026 (4K -> 720p):
    khung thu 2 sau lenh (~70 ms) da dung vung moi, 31 fps o x1/x2/x3/x5. gst-launch thi phai
    dung lai pipeline, ~8 s mat hinh giua luc tim lua.
  * nhanh XAM toan canh 960x540 cho ground_range (khong theo zoom), sau `valve`: dong thi VIC
    khong ton gi; mo (luc ha de tha) chinh 30 fps + xam 29 fps (do 23/10/2026, VIC ghim max).

VI SAO TIEN TRINH CON chu khong gi ngay trong run_gui: nvjpegdec thinh thoang chet
("gst_adapter_push: assertion 'GST_IS_BUFFER (buf)' failed" roi SEGV - 23/10/2026 ba lan trong
3 gio voi gst-launch, hai lan trong 30 s khi chay trong tien trinh). O day chet thi chi mat con
nay, GstCap._watch dung lai nhu voi gst-launch; trong tien trinh thi chet ca service.

    gst_zoom.py DEV CAM_W CAM_H FPS W H L T CW CH GRAY_FD
    stdin : moi dong mot lenh "crop L T CW CH" | "gray 0" | "gray 1"; EOF = cha chet -> thoat
    fd 1  : khung BGR W*H*3 noi duoi nhau (nhu fdsink cua gst-launch)
    GRAY_FD: khung GRAY8 960x540 noi duoi nhau, chi khi "gray 1"
"""
import os
import sys
import threading

import gi

gi.require_version("Gst", "1.0")
from gi.repository import Gst  # noqa: E402

GRAY = (960, 540)


def main():
    dev, cam_w, cam_h, fps, w, h, l, t, cw, ch, gray_fd = sys.argv[1:12]
    gray_fd = int(gray_fd)
    Gst.init(None)
    gw, gh = GRAY
    pl = Gst.parse_launch(
        f"v4l2src device={dev} io-mode=2 ! image/jpeg,width={cam_w},height={cam_h},"
        f"framerate={fps}/1 ! nvjpegdec ! video/x-raw(memory:NVMM) ! tee name=t "
        "t. ! queue max-size-buffers=1 leaky=downstream ! nvvidconv name=cut ! "
        f"video/x-raw,format=BGRx,width={w},height={h} ! videoconvert ! video/x-raw,format=BGR ! "
        "appsink name=a max-buffers=1 drop=true sync=false "
        "t. ! queue max-size-buffers=1 leaky=downstream ! valve name=v drop=true ! nvvidconv ! "
        f"video/x-raw,format=GRAY8,width={gw},height={gh} ! "
        # async=false: valve dong thi nhanh nay khong co khung preroll -> ca pipeline ket PAUSED
        "appsink name=g max-buffers=1 drop=true sync=false async=false")
    cut, valve = pl.get_by_name("cut"), pl.get_by_name("v")

    def crop(l, t, cw, ch):
        # left/right/top/bottom la TOA DO x0,x1,y0,y1 (xem GstCap._decode)
        for k, v in (("left", l), ("right", l + cw), ("top", t), ("bottom", t + ch)):
            cut.set_property(k, v)

    crop(int(l), int(t), int(cw), int(ch))
    pl.set_state(Gst.State.PLAYING)

    def pump(sink, fd, n):
        while True:
            s = sink.emit("try-pull-sample", Gst.SECOND)
            if s is None:
                if sink.get_property("eos"):
                    os._exit(1)
                continue
            b = s.get_buffer()
            ok, m = b.map(Gst.MapFlags.READ)
            if not ok:
                continue
            try:
                data = memoryview(m.data)[:n]
                while data:                        # pipe 64 KB: write ghi tung phan
                    data = data[os.write(fd, data):]
            except OSError:                        # cha da dong pipe
                os._exit(0)
            finally:
                b.unmap(m)

    threading.Thread(target=pump, args=(pl.get_by_name("a"), 1, int(w) * int(h) * 3), daemon=True).start()
    threading.Thread(target=pump, args=(pl.get_by_name("g"), gray_fd, gw * gh), daemon=True).start()
    for line in sys.stdin:
        cmd = line.split()
        if cmd[:1] == ["crop"] and len(cmd) == 5:
            crop(*map(int, cmd[1:]))
        elif cmd[:1] == ["gray"] and len(cmd) == 2:
            valve.set_property("drop", cmd[1] != "1")
    os._exit(0)  # stdin EOF = run_gui chet: khong de con nay giu camera


if __name__ == "__main__":
    main()
