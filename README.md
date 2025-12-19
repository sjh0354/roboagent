# Robot Agent System Usage Guide

This guide explains how to set up and use the Vision-Language Model (VLM) based robot planners for both the Humanoid Robot (Unitree-G1) and the Robotic Arm (UR5e), including how to integrate with a RealSense D435 camera.

## 1. Installation & Setup

### Prerequisites
- Python 3.8+
- Intel RealSense SDK 2.0 (if using real camera)
- DashScope API Key (for Qwen VLM)

### Install Dependencies
Run the following command to install the required Python packages:

```bash
pip install -r requirements.txt
```

### Set API Key
You must set your DashScope API key environment variable for the VLM to work:

```bash
export DASHSCOPE_API_KEY='your-dashscope-api-key'
```

## 2. Testing the Camera

Before running the planners in real-robot mode, verify that your RealSense D435 camera is working correctly.

Run the standalone camera manager script:

```bash
python utils/realsense_manager.py
```

**Expected Output:**
- The script should initialize the camera.
- It will capture a test image.
- It will save the image to the `captured_images/` directory.
- It will print the path of the saved image.

If you see errors like `RuntimeError: No device detected`, check your USB connection.

## 3. Using the Humanoid Planner (Unitree-G1)

The Humanoid Planner manages navigation, talking, and environment control (AC, lights).

### Simulation Mode (Default)
Runs using pre-captured images from `simulation_images/`.

```bash
python planner/humanoid_planner_vlm.py
```

### Real Robot Mode (with Camera)
To run with the real camera integration, modify the initialization in your script or ensure the `simulation_mode` flag is set to `False` when initializing the planner/executor.

Currently, the `humanoid_planner_vlm.py` main block initializes in simulation mode by default. You can edit the file `humanoid_planner_vlm.py`:

```python
# In main() function:
planner = AutonomousVLMPlanner(
    model_name="qwen-vl-plus",
    simulation_mode=False,  # <--- Set to False for real camera
    verbose=True
)
```

## 4. Using the Robotic Arm Planner (UR5e)

The Arm Planner manages manipulation tasks like picking items from shelves and placing them on counters.

### Simulation Mode (Default)
```bash
python planner/arm_planner_vlm.py
```

### Real Robot Mode (with Camera)
Similar to the humanoid planner, edit `arm_planner_vlm.py` to enable real robot mode:

```python
# In main() function:
planner = AutonomousArmVLMPlanner(
    model_name="qwen-vl-plus",
    simulation_mode=False,  # <--- Set to False for real camera
    verbose=True
)
```

**Note:** The Arm Planner currently has `_get_default_observation_image` returning a simulation path. Ensure you update it or rely on the autonomous loop which calls `executor.get_current_observation()` (if refactored to use the executor) or `camera_manager.capture_image()` directly if integrated into the planner loop.

## 5. Troubleshooting

- **Camera not found:** Ensure `pyrealsense2` is installed and the camera is plugged into a USB 3.0 port.
- **VLM Errors:** Check your `DASHSCOPE_API_KEY` and internet connection.
- **Import Errors:** Ensure all files are in the same directory and `requirements.txt` packages are installed.
