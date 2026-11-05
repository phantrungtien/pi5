#!/usr/bin/env python3
"""Liet ke / tai log DataFlash (.BIN) tu Pixhawk qua cong USB thu 2 (if02, MAVROS giu if00).
  python3 fc_logs.py            # liet ke
  python3 fc_logs.py 12 13      # tai log id 12, 13 vao thu muc nay
ponytail: tai tuan tu tung goi 90 B, khong xin lai goi thieu theo khoang; USB nhanh nen du.
"""
import sys, time, os
from datetime import datetime
from pymavlink import mavutil

PORT = "/dev/serial/by-id/usb-Holybro_Pixhawk6C_48003C000F51333337383131-if02"
m = mavutil.mavlink_connection(PORT, baud=115200, source_system=250)
m.wait_heartbeat(timeout=10)

def entries():
    m.mav.log_request_list_send(m.target_system, m.target_component, 0, 0xffff)
    out, n = {}, None
    while True:
        e = m.recv_match(type="LOG_ENTRY", blocking=True, timeout=3)
        if e is None: break
        if e.num_logs == 0: break
        out[e.id] = e; n = e.num_logs
        if len(out) >= n: break
    return out

def fetch(e):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        f"log{e.id}_{datetime.fromtimestamp(e.time_utc):%Y%m%d_%H%M}.BIN" if e.time_utc else f"log{e.id}.BIN")
    buf = bytearray(e.size); got = 0; ofs = 0; t0 = time.time()
    while ofs < e.size:
        m.mav.log_request_data_send(m.target_system, m.target_component, e.id, ofs, min(e.size - ofs, 90 * 512))
        end = min(e.size, ofs + 90 * 512)
        while ofs < end:
            d = m.recv_match(type="LOG_DATA", blocking=True, timeout=2)
            if d is None: break                      # mat goi -> xin lai tu ofs
            if d.id != e.id or d.ofs != ofs: continue
            buf[ofs:ofs + d.count] = bytes(d.data[:d.count]); ofs += d.count
            if d.count == 0: ofs = e.size
        print(f"\r  log {e.id}: {ofs/1024:.0f}/{e.size/1024:.0f} KB", end="", flush=True)
    m.mav.log_request_end_send(m.target_system, m.target_component)
    open(path, "wb").write(buf[:ofs])
    print(f"  -> {path} ({time.time()-t0:.0f} s)")

es = entries()
if len(sys.argv) == 1:
    for i, e in sorted(es.items()):
        t = datetime.fromtimestamp(e.time_utc).strftime("%Y-%m-%d %H:%M") if e.time_utc else "?"
        print(f"{i:4d}  {t}  {e.size/1024/1024:7.1f} MB")
else:
    for a in sys.argv[1:]:
        fetch(es[int(a)])
