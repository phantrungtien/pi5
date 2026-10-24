# Subscribe to a topic through rosbridge and report rate + one sample.
import json, sys, time
from tornado import ioloop, websocket

URL = sys.argv[1] if len(sys.argv) > 1 else "ws://localhost:9090"
TOPIC = sys.argv[2] if len(sys.argv) > 2 else "/mavros/imu/data"
SECS = 10


async def main():
    ws = await websocket.websocket_connect(URL)
    ws.write_message(json.dumps({"op": "subscribe", "topic": TOPIC}))
    n, first, t0 = 0, None, time.time()
    while time.time() - t0 < SECS:
        msg = await ws.read_message()
        if msg is None:
            break
        m = json.loads(msg)
        if m.get("op") == "publish":
            n += 1
            first = first or m["msg"]
    ws.close()
    print(f"{TOPIC}: {n} msgs in {SECS}s = {n / SECS:.1f} Hz")
    if first:
        print("sample linear_acceleration:", first.get("linear_acceleration"))
    assert n > 0, "no messages through rosbridge"


ioloop.IOLoop.current().run_sync(main)
