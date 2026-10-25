cmake_minimum_required(VERSION 3.8)
project(lego_sort_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Brick.msg"
  "msg/Observation.msg"
  "srv/ObservePile.srv"
  "srv/MotionControl.srv"
  "srv/TaskControl.srv"
  "srv/VerifyPlacement.srv"
  "action/PickBrick.action"
  "action/PlaceBrick.action"
  DEPENDENCIES geometry_msgs builtin_interfaces)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
