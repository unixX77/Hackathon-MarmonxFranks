# Source this in EVERY terminal:  source ~/hackathon_repo/env.sh
# Sets up ROS + this workspace, goes to the repo root, and keeps the simulation private
# (ROS_LOCALHOST_ONLY/ROS_DOMAIN_ID must be the same in every terminal).
source /opt/ros/humble/setup.bash
source "$(dirname "${BASH_SOURCE[0]}")/install/setup.bash"
export ROS_LOCALHOST_ONLY=1 ROS_DOMAIN_ID=77
cd "$(dirname "${BASH_SOURCE[0]}")"
echo "ready: $(pwd)  ROS_DOMAIN_ID=$ROS_DOMAIN_ID  fr3_pick_place -> $(ros2 pkg prefix fr3_pick_place)"
