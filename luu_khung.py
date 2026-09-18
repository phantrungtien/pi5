"""Luu N khung SACH tu /image_raw ra file. Khong ve gi len anh."""
import os, sys, time
import cv2, numpy as np, rclpy
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import Image

OUT, N, CACH = "/out", 20, 0.5
os.makedirs(OUT, exist_ok=True)
rclpy.init()
node = rclpy.create_node("luu_khung")
state = {"n": 0, "t": 0.0}

def on_image(msg):
    if state["n"] >= N or time.time() - state["t"] < CACH:
        return
    state["t"] = time.time()
    a = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.step // 3, 3)[:, : msg.width]
    cv2.imwrite(f"{OUT}/{state['n']:02d}.jpg", a[:, :, ::-1], [cv2.IMWRITE_JPEG_QUALITY, 95])
    state["n"] += 1

node.create_subscription(Image, "/image_raw", on_image,
                         QoSProfile(depth=1, reliability=QoSReliabilityPolicy.BEST_EFFORT,
                                    history=QoSHistoryPolicy.KEEP_LAST))
t0 = time.time()
while state["n"] < N and time.time() - t0 < 30:
    rclpy.spin_once(node, timeout_sec=0.2)
print("luu", state["n"], "khung")
