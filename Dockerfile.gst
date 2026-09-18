FROM drone-vision:humble
# gstreamer de doc thang MJPG tu webcam: USB da nen san, chi con jpegdec.
# Re hon nhieu so voi v4l2_camera_node chuyen YUYV->RGB8 720p bang CPU.
RUN apt-get update  && apt-get install -y --no-install-recommends       gstreamer1.0-tools gstreamer1.0-plugins-base gstreamer1.0-plugins-good  && rm -rf /var/lib/apt/lists/*
