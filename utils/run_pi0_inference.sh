#!/bin/bash

# Usage:
# bash run_pi0_inference.sh
# POLICY_HOST=192.168.1.200 bash run_pi0_inference.sh  # Optional: specify remote policy server
#conda create -n openpi python=3.10 -y
#cd ~/projects/openpi/packages/openpi-client
#pip install -e .
# conda activate openpi

# Start a new detached tmux session named 'tmp'
tmux new-session -d -s tmp
tmux split-window -h -t tmp:0.0
tmux split-window -v -t tmp:0.0
tmux split-window -v -t tmp:0.0
tmux split-window -v -t tmp:0.2

# Get script directory
script_dir=$(dirname "$(readlink -f "$0")")
echo "Script directory: $script_dir"

# Install openpi_client to system Python (for ROS2)
if ! python3 -c "import openpi_client" 2>/dev/null; then
    echo "Installing openpi_client to system Python..."
    pip install -e /app/openpi/packages/openpi-client
fi

# Configuration (can be overridden by environment variables)
POLICY_HOST="${POLICY_HOST:-10.11.19.54}"
POLICY_PORT="${POLICY_PORT:-8000}"
ROBOT_IP="${ROBOT_IP:-192.168.24.245}"
ROBOT_NAME="${ROBOT_NAME:-arm_1}"
USE_REAL_ROBOT="${USE_REAL_ROBOT:-true}"
INSTRUCTION="${INSTRUCTION:-Move the medicine box into the basket.}"

echo "Policy server: $POLICY_HOST:$POLICY_PORT"
echo "Robot IP: $ROBOT_IP"

echo "Building the project..."
# Build the project first

cd $script_dir

#colcon build

tmux send-keys -t tmp:0.0 "cd $script_dir && rm -rf build install log && colcon build" C-m
while true; do
    pane_0_0=$(tmux capture-pane -pt tmp:0.0)
    if echo "$pane_0_0" | grep -q "Summary:"; then
        break
    fi
    sleep 0.2
done

sleep 1

echo "Launching the camera nodes..."
# Terminal 0.0: Camera nodes
tmux send-keys -t tmp:0.0 "cd $script_dir && \
                            source ./install/setup.bash && \
                            ros2 launch ur5e_teleopration v4l2_camera.launch.py robot_name:=ur5e" C-m

echo "Launching the UR5e Robot node..."
# Terminal 0.1: UR5e Robot node
tmux send-keys -t tmp:0.1 "cd $script_dir && \
                            source ./install/setup.bash && \
                            ros2 launch ur5e_teleopration ur5e_single_arm.launch.py \
                            robot_name:=$ROBOT_NAME \
                            robot_ip:=$ROBOT_IP \
                            use_real_robot:=$USE_REAL_ROBOT" C-m

tmux send-keys -t tmp:0.2 "conda activate openpi && \
                            pip install cv2 && \
                            pip install typing_extension" C-m

echo "Launching the PI0 UR5e control node..."
# Terminal 0.2: PI0 UR5e control node
tmux send-keys -t tmp:0.2 "cd $script_dir && \
                            source ./install/setup.bash && \
                            ros2 launch ur5e_teleopration pi0_ur5e.launch.py \
                            policy_host:=$POLICY_HOST \
                            policy_port:=$POLICY_PORT \
                            instruction:='$INSTRUCTION'" C-m