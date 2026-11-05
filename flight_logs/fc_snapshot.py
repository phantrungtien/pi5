#!/usr/bin/env python3
"""Chup trang thai FC that ve file (chay TREN Jetson, qua cong USB thu 2 nen khong dung MAVROS):
toan bo tham so, phien ban firmware, duong bay / rao / rally dang nap. De lam viec khong co drone.

    python3 fc_snapshot.py /tmp/fc_snap        # -> params.parm, version.txt, mission/fence/rally.waypoints
"""
import os
import sys
import time

os.environ["MAVLINK20"] = "1"  # mission_type (rao/rally) chi co o MAVLink2
from pymavlink import mavutil  # noqa: E402

PORT = "/dev/serial/by-id/usb-Holybro_Pixhawk6C_48003C000F51333337383131-if02"
TYPES = {"mission": 0, "fence": 1, "rally": 2}


def params(m):
    m.mav.param_request_list_send(m.target_system, m.target_component)
    got, total = {}, None
    for _ in range(3):  # luot dau + 2 luot hoi lai cai rot
        while True:
            r = m.recv_match(type="PARAM_VALUE", blocking=True, timeout=2)
            if r is None:
                break
            total = r.param_count
            if r.param_index < total:
                got[r.param_index] = (r.param_id, r.param_value)
        miss = [i for i in range(total or 0) if i not in got]
        if total and not miss:
            break
        for i in miss:
            m.mav.param_request_read_send(m.target_system, m.target_component, b"", i)
    return total, sorted(got.values())


def items(m, mtype):
    m.mav.mission_request_list_send(m.target_system, m.target_component, mtype)
    c = m.recv_match(type="MISSION_COUNT", blocking=True, timeout=3)
    if c is None:
        return None
    out = []
    for seq in range(c.count):
        for _ in range(3):
            m.mav.mission_request_int_send(m.target_system, m.target_component, seq, mtype)
            it = m.recv_match(type="MISSION_ITEM_INT", blocking=True, timeout=2)
            if it is not None and it.seq == seq:
                out.append(it)
                break
        else:
            raise RuntimeError(f"loai {mtype}: khong lay duoc muc {seq}")
    m.mav.mission_ack_send(m.target_system, m.target_component, 0, mtype)
    return out


def main(outdir):
    os.makedirs(outdir, exist_ok=True)
    m = mavutil.mavlink_connection(PORT, baud=115200, source_system=250)
    m.wait_heartbeat(timeout=10)
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")

    m.mav.command_long_send(m.target_system, m.target_component,
                            mavutil.mavlink.MAV_CMD_REQUEST_AUTOPILOT_CAPABILITIES, 0, 1, 0, 0, 0, 0, 0, 0)
    v = m.recv_match(type="AUTOPILOT_VERSION", blocking=True, timeout=5)
    fw = "?"
    if v:
        x = v.flight_sw_version
        git = bytes(v.flight_custom_version).rstrip(b"\0").decode(errors="replace")
        fw = f"{x >> 24 & 255}.{x >> 16 & 255}.{x >> 8 & 255} (git {git})"
    with open(f"{outdir}/version.txt", "w") as f:
        f.write(f"chup {stamp}\nArduPilot {fw}\nsysid {m.target_system}\n")

    total, ps = params(m)
    with open(f"{outdir}/params.parm", "w") as f:
        f.write(f"# FC that (Pixhawk 6C, ArduPilot {fw}) chup {stamp} - {len(ps)}/{total} tham so\n")
        for name, val in ps:
            # so nguyen ghi du chu so (ID cam bien 8 chu so, %g se lam tron); so thuc 7 chu so = float32
            f.write(f"{name},{int(val) if val.is_integer() else f'{val:.7g}'}\n")
    print(f"tham so: {len(ps)}/{total}")

    for name, mtype in TYPES.items():
        its = items(m, mtype)
        if its is None:
            print(f"{name}: FC khong tra MISSION_COUNT")
            continue
        with open(f"{outdir}/{name}.waypoints", "w") as f:
            f.write("QGC WPL 110\n")
            for it in its:
                f.write("\t".join(str(a) for a in (
                    it.seq, it.current, it.frame, it.command, it.param1, it.param2, it.param3, it.param4,
                    it.x / 1e7, it.y / 1e7, it.z, it.autocontinue)) + "\n")
        print(f"{name}: {len(its)} muc")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "fc_snapshot")
