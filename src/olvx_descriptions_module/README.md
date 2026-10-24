# Olive Components Description

This package contains ROS-related information about individual olive components such as olive SERVO, olive IMU, olive CAMERA.

It has .urdf definitions, meshes, collision info etc.

Note: This does not contain the urdf for olive kits (such as olive OWL or olive ANT).

## Usage

```
unzip meshes/olv-chassis01.zip -d meshes/

sudo apt install ros-humble-xacro

git clone https://gitlab.com/oliverobotics/hardware/descriptions/olv_module_descriptions.git

colcon build

source install/setup.bash

ros2 launch olv_module_descriptions visualize_olive_imu.launch.py

```
