# ROS 2: publish nominal hand-eye fr3_link8 -> camera_link (realsense-ros publishes camera_link -> *_optical_frame)
from launch import LaunchDescription
from launch_ros.actions import Node
def generate_launch_description():
    return LaunchDescription([Node(package='tf2_ros', executable='static_transform_publisher', name='handeye_fr3_d435',
        arguments=['--x','0.04773','--y','0.022981','--z','0.06075',
                   '--qx','0.653281','--qy','0.270598','--qz','0.653281','--qw','-0.270598',
                   '--frame-id','fr3_link8','--child-frame-id','camera_link'])])
