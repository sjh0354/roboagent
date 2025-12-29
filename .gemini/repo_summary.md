# Repository Summary: Vision-Based Robotic Agent System

## Project Goal
This repository contains a sophisticated Vision-Based Robotic Agent System designed to control a humanoid robot (Unitree-G1) and a robotic arm (UR5e) using a Vision-Language Model (VLM), specifically Qwen. The system has been refactored from a human-in-the-loop system to a more advanced, autonomous, closed-loop model. It leverages VLM to interpret visual scenes and generate action plans, and is capable of interfacing with both simulation and real-world hardware.

## Core Architecture
The system employs a perceive-plan-act cycle, with a clear separation of concerns between planning and execution, and integrates vision for autonomous verification.

1.  **The Planner (e.g., `humanoid_planner_vlm.py`, `arm_planner_vlm.py`):**
    *   Functions as the 'brain' of the system.
    *   Maintains the overall goal, receives visual context (an image), and a task history.
    *   Uses the VLM to decide the next single action to perform.
    *   Operates autonomously in a **closed-loop** fashion, continuously planning and verifying.

2.  **The Executor (e.g., `humanoid_executor_vision.py`, `arm_executor_vision.py`):**
    *   Functions as the 'body' of the system.
    *   Receives an action command from the planner and executes it.
    *   Utilizes its 'senses' (camera and VLM client) to provide feedback.
    *   Captures 'before' and 'after' images of the action.
    *   Uses the VLM (`qwen_vlm_client.py`'s `compare_images` function) to analyze changes and verify the success of the action, enabling closed-loop feedback.

3.  **Hardware & Vision Management:**
    *   **VLM Client (`utils/qwen_vlm_client.py`):** Provides the direct interface to the Qwen Vision-Language Model, with `compare_images` being central to autonomous action verification.
    *   **Simulation Image Manager (`utils/simulation_image_manager.py`):** Essential for the 'simulation-first' design. It provides appropriate environmental images to the VLM based on a simulated world state (e.g., location, device status), enabling realistic, vision-based planning and execution without real hardware.
    *   **RealSense Manager (`realsense_manager.py`):** Handles image capture from Intel RealSense D435 cameras for real-world robot vision.
    *   **TTS Manager (`tts_manager.py`):** Manages text-to-speech functionality, prioritizing CosyVoice (DashScope) and falling back to local system TTS.

4.  **Prompt Engineering:**
    *   **Prompt Templates (e.g., `template/humanoid_prompt_template_vlm.py`):** Contain the system prompts that instruct the VLM on how to behave in the autonomous, vision-first framework, including how to interpret visual feedback and when to ask for clarification.

5.  **PI0 Inference Integration:**
    *   Located in `utils/run_pi0_inference.sh`, this script sets up a ROS2 environment to run PI0 policy inference for real-world UR5e control, using `openpi`.

## Key Workflow & State
*   **Perception:** Captures images (real via RealSense or simulated via Image Manager) to provide visual context.
*   **Planning:** VLM analyzes the image and task history to decide the next action.
*   **Execution & Verification:** Actions are dispatched to simulation or real hardware. 'Before' and 'after' images are captured and sent to the VLM (via `compare_images`) for autonomous verification of action success, forming a closed-loop.
*   **Location Tracking:** The system explicitly tracks robot locations (e.g., "Room 01" / "home", "Room 02" / "store") to provide contextual information to the VLM.

## Recent Findings & Enhancements
*   **Architectural Refinement:** Transitioned to an autonomous, closed-loop system, with the planner acting as the 'brain' and the executor as the 'body', enabling self-correction through visual verification.
*   **Simulation-First Design:** Enhanced support for simulation through `SimulationImageManager` to facilitate development and testing.
*   **VLM-driven Verification:** Integrated `qwen_vlm_client.py`'s `compare_images` function as a core mechanism for executors to visually verify the outcome of actions.
*   **TTS Integration:** Added `utils/tts_manager.py` to enable speech for both robots, supporting "CosyVoice" (example requested by user) via DashScope SDK.
*   **Real Hardware Support:** Transitioned from simulation-only to supporting real-world hardware (RealSense, UR5e via PI0).
*   **Cooperative Model:** The system is designed for a humanoid robot to delegate tasks to a robotic arm, with the arm having its own VLM-based planning logic.