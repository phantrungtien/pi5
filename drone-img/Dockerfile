FROM ardu_ros:rviz-camera-ready

# ROS bits: mavros core + image pipeline glue
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-mavros \
      ros-humble-cv-bridge \
      ros-humble-image-transport \
 && rm -rf /var/lib/apt/lists/*

# geographiclib egm96 dataset — mavros global position needs it at runtime
RUN /opt/ros/humble/lib/mavros/install_geographiclib_datasets.sh

# ponytail: headless opencv wheel, not python3-opencv (apt drags X11/GTK/gstreamer).
# numpy pinned <2 — rclpy/cv_bridge here are compiled against numpy 1.x.
RUN pip3 install --no-cache-dir "numpy<2" onnxruntime opencv-python-headless

# node camera: v4l2_camera_node lam chu duy nhat /dev/video0
RUN apt-get update && apt-get install -y --no-install-recommends \
      ros-humble-v4l2-camera \
 && rm -rf /var/lib/apt/lists/*
