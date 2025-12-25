# Repository Summary: Vision-Based Robotic Agent System

## Project Goal
This repository contains a vision-based autonomous planning system for a humanoid robot (Unitree-G1) and a robotic arm (UR5e). The system leverages Vision Language Models (VLM), specifically Qwen, to interpret visual scenes and generate action plans. It has evolved from a pure simulation to a system capable of interfacing with real-world hardware.

## Core Architecture
The system follows a perceive-plan-act cycle with the following key components:

1.  **Planners (`humanoid_planner_vlm.py`, `arm_planner_vlm.py`):**
    *   Autonomous agents that use VLM to plan the next step based on visual observations.
    *   Operate in a **"Half-Open-Loop"** mode: They assume action success but use fresh visual input (real or simulated) for each planning step.
    *   Support for human (humanoid) or humanoid (arm) interaction for clarifications.

2.  **Vision-Enabled Executors (`humanoid_executor_vision.py`, `arm_executor_vision.py`):**
    *   **Humanoid Executor:** Manages navigation and tool interaction for the Unitree-G1.
    *   **Arm Executor:** Manages manipulation tasks for the UR5e arm.
    *   **Vision Integration:** Both executors capture observations (images) before/after actions and can use VLM to generate textual descriptions of these observations.
    *   **Hardware Interfacing:** The arm executor can trigger real-world actions via a PI0 inference script (`run_pi0_inference.sh`).
    *   **Speech Capability:** Both executors are equipped with a `TTSManager` (`utils/tts_manager.py`) to synthesize speech, supporting DashScope's CosyVoice (if configured) and a system-level fallback (`espeak`).

3.  **Hardware & Vision Management:**
    *   **RealSense Manager (`realsense_manager.py`):** Handles image capture from Intel RealSense D435 cameras for real-world robot vision.
    *   **Simulation Image Manager (`simulation_image_manager.py`):** Manages simulated world state and provides corresponding images for testing.
    *   **VLM Client (`qwen_vlm_client.py`):** Interface for the Qwen VLM API (DashScope).
    *   **TTS Manager (`tts_manager.py`):** Manages text-to-speech functionality, prioritizing CosyVoice (DashScope) and falling back to local system TTS.

4.  **PI0 Inference Integration:**
    *   Located in `utils/run_pi0_inference.sh`, this script sets up a ROS2 environment to run PI0 policy inference for real-world UR5e control, using `openpi`.

## Key Workflow & State
*   **Perception:** Captures real images via RealSense or simulated images via the Image Manager.
*   **Planning:** VLM analyzes the image and task to decide the next action.
*   **Execution:** Actions are dispatched to simulation or real hardware.
*   **Location Tracking:** The system explicitly tracks robot locations (e.g., "Room 01" / "home", "Room 02" / "store") to provide context to the VLM.

## Recent Findings & Enhancements
*   **TTS Integration:** Added `utils/tts_manager.py` to enable speech for both robots, supporting "CosyVoice" (example requested by user) via DashScope SDK.
*   **Real Hardware Support:** Transitioned from simulation-only to supporting real-world hardware (RealSense, UR5e via PI0).
*   **Closed-Loop Readiness:** While still primarily "half-open-loop," the infrastructure for visual verification (capturing images after actions and VLM analysis) is now fully integrated into the executors.
*   **Cooperative Model:** The system is designed for a humanoid robot to delegate tasks to a robotic arm, with the arm having its own VLM-based planning logic.