cmake_minimum_required(VERSION 3.8)
project(fr3_pose_pick)
find_package(ament_cmake REQUIRED)
install(PROGRAMS
  scripts/sort_pile.py scripts/sort_gui.py
  scripts/brick_publisher.py scripts/palletize_listener.py scripts/stack_listener.py
  scripts/pose_pick.py scripts/pose_pick_auto.py scripts/pose_pick_palletize.py scripts/pose_pick_stacks.py
  scripts/record_pose.py scripts/go_to_pose.py scripts/camera_frame.py
  scripts/measure_grasp_offset.py scripts/measure_hand_zero.py scripts/pallet.py scripts/pallet_tool.py
  scripts/smooth_motion.py scripts/trajectory_smoothing.py scripts/pick_place.py scripts/spawn_scene.py
  DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY config launch DESTINATION share/${PROJECT_NAME})
install(FILES README.md DESTINATION share/${PROJECT_NAME})
ament_package()
