# Repository Summary: Vision-Based Robotic Agent System

## Project Goal
This repository contains a sophisticated Vision-Based Robotic Agent System designed to control a humanoid robot (Unitree-G1) and a robotic arm (UR5e) using Vision-Language Models (VLM). The system has evolved from a human-in-the-loop setup to an autonomous, closed-loop (or "half-open-loop") model that leverages visual feedback for planning and verification.

## Core Architecture
The system employs a perceive-plan-act cycle, with a clear separation between the **Planner** (brain) and the **Executor** (body).

1.  **The Planner (e.g., `humanoid_planner_vlm.py`, `arm_planner_vlm.py`):**
    *   Functions as the decision-making core.
    *   Uses a VLM (default: **Gemini 2.5 Flash**) to analyze images directly and decide the next single action.
    *   Operates in a **"Half-Open-Loop"** mode: it assumes actions succeed to maintain flow but can be interrupted or receive clarifications.
    *   Maintains task history and context, including voice command interrupts.

2.  **The Executor (e.g., `humanoid_executor_vision.py`, `arm_executor_vision.py`):**
    *   Executes planned actions on hardware or in simulation.
    *   **Vision-Enabled:** Captures observations (images) before and after actions.
    *   **Verification:** Uses the VLM client to analyze visual changes and verify action success.
    *   **Grounding:** Utilizes Gemini's grounding for real-time web searches to answer user queries.

3.  **Hardware & Vision Management:**
    *   **VLM Clients (`gemini_vlm_client.py`, `qwen_vlm_client.py`):** Interfaces for Google Gemini and Alibaba Qwen models. Gemini 2.0 is the primary engine.
    *   **Simulation Image Manager (`simulation_image_manager.py`):** Manages a state-aware mock environment with pre-captured images for "simulation-first" development.
    *   **RealSense Manager (`realsense_manager.py`):** For UR5e arm visual feedback.
    *   **Dabai Camera Manager (`dabai_camera_manager.py`):** For Unitree-G1 humanoid visual feedback.
    *   **ASR & TTS:** Integrates **FunASR** for always-on voice command detection and **CosyVoice** (via DashScope) for high-quality speech synthesis.

4.  **PI0 Policy Integration:**
    *   Advanced manipulation for the UR5e arm is handled via **PI0 policy inference** running in a specialized Docker environment (`run_pi0_inference.sh`).

## Key Workflow & State
*   **Perception:** Captures real or simulated images to provide the VLM with direct visual context.
*   **Planning:** VLM analyzes the current image, task request, and history to generate the next action in JSON format.
*   **Execution:** Actions are dispatched to hardware or simulation. Location tracking (e.g., "Room 01" / "home") is automatically managed.
*   **Interruption:** Always-on ASR allows users to interrupt ongoing tasks with new voice commands.

## Recent Findings & Enhancements
*   **Gemini 2.0 Integration:** Shifted to Gemini as the default VLM for its superior planning and grounding (web search) capabilities.
*   **Half-Open-Loop Execution:** Refined the execution loop to be more responsive, assuming success for low-level actions while retaining visual verification in the background.
*   **Voice-First Interaction:** Integrated FunASR for wake-word detection and real-time task interruption.
*   **Dockerized PI0 Inference:** Stabilized the robotic arm's manipulation capabilities by wrapping PI0 inference in a managed Docker/tmux environment.
*   **Unified Navigation:** Standardized location tracking across simulation and real hardware using a consistent "Room ID" mapping.
