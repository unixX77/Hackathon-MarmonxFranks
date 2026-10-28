# Source this in EVERY terminal:  source <this folder>/env.sh
# ROS 2 Humble + franka_ros2 + this workspace, ROS kept on this computer, and the shell moved to the repository root
# (the commands use paths relative to it). ROS_LOCALHOST_ONLY / ROS_DOMAIN_ID must be the same in every terminal.
source /opt/ros/humble/setup.bash
# franka_ros2 (franka_fr3_moveit_config, franka_msgs, franka_gripper) is built in its own workspace
FRANKA_WS=${FRANKA_WS:-$HOME/Documents/Hackathon/hackathon_ros2_ws}
if [ -f "$FRANKA_WS/install/setup.bash" ]; then
  source "$FRANKA_WS/install/setup.bash"
else
  echo "WARNING: no franka_ros2 workspace at $FRANKA_WS: export FRANKA_WS=<its folder> and source this again"
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$ROOT/install/setup.bash" ] && source "$ROOT/install/setup.bash" || echo "WARNING: not built yet: run ./build.sh"
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=77
cd "$ROOT"
echo "ready: $(pwd)  ROS_DOMAIN_ID=$ROS_DOMAIN_ID  fr3_pose_pick -> $(ros2 pkg prefix fr3_pose_pick 2>/dev/null || echo 'not built')"
