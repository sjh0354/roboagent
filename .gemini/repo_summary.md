# Repository Summary: Vision-Based Robotic Agent System

## Project Goal
This repository contains a vision-based autonomous planning system for a humanoid robot, which cooperates with a separate robotic arm planner. The system's primary objective is to use a Vision Language Model (VLM), specifically Qwen, to interpret visual scenes from a simulated environment and generate step-by-step action plans to achieve high-level user goals.

## Core Architecture
The system is composed of four main components that work together in a perceive-plan-act cycle:

1.  **Planners (`humanoid_planner_vlm.py`, `arm_planner_vlm.py`):**
    These are the "brains" of the system. They receive a user goal, observe the current state via an image from the simulation, and query the VLM to determine the single next best action to take. The humanoid and arm planners are distinct agents, suggesting a cooperative model where the humanoid handles navigation and can delegate manipulation tasks to the arm.

2.  **Executor (`humanoid_executor_vision.py`):**
    This is the "body" of the robot. It takes the action chosen by the planner (e.g., "move to the living room") and executes it within the simulation. It inherits from a base executor (`humanoid_executor.py`) that defines the primitive actions available to the robot.

3.  **Simulation Image Manager (`simulation_image_manager.py`):**
    This is the "world" or state machine for the simulation. It tracks the current state of the environment (e.g., robot location, status of devices like the AC or lights) and provides the correct image that corresponds to that state. When the executor performs an action, the manager updates its state and provides the new image for the next planning cycle.

4.  **VLM Client (`qwen_vlm_client.py`):**
    This is the dedicated interface to the external Qwen VLM. It formats the requests, sends the current image and prompt to the VLM API, and returns the model's response (the chosen action).

## Key Workflow & Discrepancy
The system currently operates in a **"Half-Open-Loop"** mode:
1. The planner gets a goal and the current image.
2. It asks the VLM for the next single step.
3. The executor performs that step.
4. The system **assumes the action was successful** and immediately moves on to plan the next step based on the new state.

Notably, the documentation (`VLM_PLANNER_README.md`) describes a more advanced, fully **closed-loop** system that would visually verify the success of each action by comparing "before" and "after" images. The `qwen_vlm_client.py` even contains a `compare_images` function for this. However, this verification step is **not currently implemented** in the main planning loop of `humanoid_planner_vlm.py`, representing a key difference between the documented design and the current implementation.
