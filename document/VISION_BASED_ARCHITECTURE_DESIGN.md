# Vision-Based Autonomous Planner - Architecture Design

## Overview

Transform the current **human-in-the-loop interactive planner** into an **autonomous vision-based planner** that:
- Uses VLM (Vision-Language Model) to directly observe the environment via video frames
- Automatically verifies action success/failure through visual observation
- Eliminates manual human confirmation (except for initial task requests)
- Relies entirely on visual feedback for planning decisions

## Current vs. Proposed Architecture

### Current (Human-in-Loop, Text-Only)
```
Request → Text LLM Plans → Execute → [WAIT FOR HUMAN] → Manual Text Feedback → Plan Next
```

### Proposed (Vision-Based Autonomous, VLM Planner)
```
Request → Capture Frame → VLM Planner (sees image) → Execute → Capture Frame → VLM Planner
              ↑                  ↓                                      ↑            ↓
              └──────────────── Autonomous Loop with Visual Context ───────────────┘

KEY: The planner IS a VLM that receives images directly, not text descriptions!
```

## UPDATED DESIGN (Based on Feedback)

### Critical Architectural Decision

**The planner itself uses VLM with direct image input** - not text-only LLM with text descriptions!

**Why this is better:**
- VLM sees full visual detail (object positions, states, spatial relationships)
- No information loss from text description conversion
- Planner can directly observe what text cannot describe (colors, layouts, conditions)
- More robust to unexpected visual states

**Implementation:**
- Planning model: `qwen-vl-plus` (VLM)
- Each planning call includes: current observation image + text prompt
- Planner sees the world directly through vision

### User Requirements (Confirmed)
1. ✅ Planner uses images directly (VLM), not text descriptions
2. ✅ Use qwen-vl-plus model
3. ✅ Focus on simulation mode first (user provides mock images)
4. ✅ Keep interactive mode code intact (for human intention clarifications)

## Key Changes

### 1. VLM Integration Module

**New File**: `qwen_vlm_client.py`

```python
class QwenVLMClient:
    """Client for Qwen VLM (qwen-vl-plus, qwen-vl-max)"""

    def __init__(self, api_key, model_name="qwen-vl-plus"):
        # Initialize OpenAI-compatible client
        # Base URL: https://dashscope.aliyuncs.com/compatible-mode/v1
        pass

    def analyze_frame(self, image_path_or_base64, query):
        """
        Analyze a video frame with specific query

        Args:
            image_path_or_base64: Frame image (path or base64)
            query: What to observe (e.g., "Describe the current state of the room")

        Returns:
            str: VLM observation description
        """
        pass

    def verify_action_result(self, image_before, image_after, action_description):
        """
        Compare before/after frames to verify action success

        Args:
            image_before: Frame before action
            image_after: Frame after action
            action_description: What action was performed

        Returns:
            dict: {success: bool, observation: str, changes_detected: list}
        """
        pass
```

**Supported Qwen VLM Models**:
- `qwen-vl-plus` (recommended, cost-effective)
- `qwen-vl-max` (highest accuracy)
- `qwen2-vl-7b-instruct` (if available)

### 2. Camera/Frame Capture Module

**New File**: `camera_capture.py`

```python
class CameraCapture:
    """Handles video stream and frame capture"""

    def __init__(self, camera_id=0, simulation_mode=True):
        """
        Initialize camera

        Args:
            camera_id: Camera device ID (0 for default)
            simulation_mode: If True, use mock frames
        """
        pass

    def capture_current_frame(self):
        """
        Capture current frame from video stream

        Returns:
            numpy.ndarray or str: Frame image (or path in simulation)
        """
        pass

    def save_frame(self, frame, filename):
        """Save frame to disk for logging"""
        pass

    def get_mock_frame(self, context=None):
        """
        Generate mock frame for simulation
        (can return predefined test images)
        """
        pass
```

### 3. Modified Executor Architecture

**Updated**: `humanoid_executor.py`

```python
class HumanoidExecutor:
    def __init__(self, simulation_mode=True, verbose=True, enable_vision=True):
        self.simulation_mode = simulation_mode
        self.verbose = verbose
        self.enable_vision = enable_vision

        # NEW: Vision components
        if enable_vision:
            self.vlm_client = QwenVLMClient()
            self.camera = CameraCapture(simulation_mode=simulation_mode)

    def execute_action(self, action_type, action_name, parameters):
        """
        Execute action with automatic vision-based verification

        Flow:
        1. Capture frame BEFORE action
        2. Execute action
        3. Capture frame AFTER action
        4. Use VLM to compare and verify
        5. Return ExecutionResult with VLM observation
        """

        # Step 1: Capture before frame
        frame_before = self.camera.capture_current_frame()

        # Step 2: Execute action (existing logic)
        action_result = self._execute_action_internal(action_type, action_name, parameters)

        # Step 3: Capture after frame
        frame_after = self.camera.capture_current_frame()

        # Step 4: VLM verification
        if self.enable_vision:
            vlm_observation = self.vlm_client.verify_action_result(
                frame_before,
                frame_after,
                f"{action_type}.{action_name} with {parameters}"
            )

            # Update result with VLM feedback
            action_result.feedback = vlm_observation['observation']
            action_result.data['vlm_verification'] = vlm_observation

        return action_result

    def get_current_observation(self):
        """
        Get current visual observation (for planning context)
        """
        frame = self.camera.capture_current_frame()
        observation = self.vlm_client.analyze_frame(
            frame,
            "Describe the current scene in detail, focusing on objects, people, and their states."
        )
        return observation
```

### 4. Modified Planner Architecture

**Updated**: `humanoid_planner_interactive.py` → Rename to `humanoid_planner_autonomous.py`

Key changes:

```python
class AutonomousHumanoidPlanner:  # Renamed from Interactive
    """
    Autonomous Vision-Based Humanoid Planner

    Uses VLM observations for automatic feedback loop.
    Only interacts with humans for:
    - Initial task requests
    - Clarification questions
    - Final completion reports
    """

    def __init__(self, api_key=None, model_name=None, simulation_mode=True, verbose=True):
        # Initialize with vision-enabled executor
        self.executor = HumanoidExecutor(
            simulation_mode=simulation_mode,
            verbose=verbose,
            enable_vision=True  # NEW
        )

    def start_new_task(self, human_request):
        """
        Start new task and run autonomously until completion

        Args:
            human_request: Natural language task from human

        Returns:
            dict: Task completion summary
        """
        self.reset_conversation()
        self.original_request = human_request

        # Run autonomous loop
        return self._autonomous_execution_loop()

    def _autonomous_execution_loop(self):
        """
        Main autonomous execution loop

        Flow:
        1. Plan next step
        2. Execute step
        3. Get VLM observation automatically
        4. Provide observation as feedback to planner
        5. Repeat until task complete

        Returns when:
        - Task completes successfully
        - Error requires human intervention
        - Clarification needed from human
        """
        while not self.is_task_complete:
            # Plan next step
            step_plan = self.plan_next_step()

            if step_plan.get("next_step") is None:
                # Task complete
                break

            if step_plan.get("needs_human_input"):
                # Need clarification - pause and return
                return step_plan

            # Execute step
            execution_result = self.executor.execute_action(
                step_plan['next_step']['action_type'],
                step_plan['next_step']['action'],
                step_plan['next_step']['parameters']
            )

            # Auto-feedback from VLM observation
            vlm_feedback = execution_result.get_feedback_message()

            # Update context automatically
            self._add_execution_to_history(step_plan, execution_result)
            self._add_feedback_to_conversation(vlm_feedback)

        return self.get_task_summary()

    def ask_human_clarification(self, question):
        """
        Ask human for clarification (only when necessary)

        Returns:
            str: Human's response
        """
        print(f"\n🤖 Robot needs clarification: {question}")
        return input("Your response: ").strip()
```

### 5. Prompt Template Updates

**Updated**: `humanoid_prompt_template_interactive.py` → `humanoid_prompt_template_autonomous.py`

Changes to system prompt:

```python
AUTONOMOUS_SYSTEM_PROMPT = """
You are an autonomous planning system for a Unitree-G1 humanoid robot with vision capabilities.

KEY DIFFERENCES FROM INTERACTIVE MODE:
- You receive VISUAL OBSERVATIONS from a Vision-Language Model (VLM)
- Observations are automatic - captured after each action execution
- You do NOT wait for human confirmation (except for task requests)
- Use visual feedback to verify action success/failure

WHEN TO COMMUNICATE WITH HUMANS:
1. ONLY for initial task requests ("What would you like me to do?")
2. ONLY when you need clarification about user preferences
3. ONLY to report task completion
4. NEVER for step-by-step confirmations

VISUAL OBSERVATION FORMAT:
After each action, you will receive:
```
[VLM OBSERVATION]: <detailed visual description>
Changes detected: <list of visual changes>
Verification: <success/failure based on visual evidence>
```

PLANNING WITH VISION:
- Trust VLM observations as ground truth
- Use "sense" actions (get_observation) when you need current state
- Verify action success by comparing expected vs. observed visual changes
- If VLM reports failure, plan recovery actions automatically

... (rest of prompt)
"""
```

### 6. Response Format Updates

Update JSON response schema to include:

```json
{
  "current_step_analysis": {
    "visual_state": "What VLM currently observes",
    "task_progress": "What's been accomplished",
    "next_action_reasoning": "Why this action"
  },
  "next_step": {
    "step_number": 2,
    "action": "control_air_conditioner",
    "action_type": "tool",
    "parameters": {...},
    "expected_visual_outcome": "AC display should show 'ON', temperature 22°C",
    "verification_method": "VLM will check AC display state"
  },
  "needs_human_input": false,  // NEW
  "human_question": null       // NEW: Only if needs_human_input=true
}
```

## Implementation Plan

### Phase 1: VLM Integration (Core)
1. Create `qwen_vlm_client.py` with Qwen VLM API integration
2. Test VLM with sample frames (verify API works)
3. Create `camera_capture.py` with simulation support

### Phase 2: Executor Enhancement
1. Modify `humanoid_executor.py` to add vision components
2. Implement before/after frame capture
3. Add VLM observation to execution results
4. Test individual actions with vision

### Phase 3: Planner Transformation
1. Create `humanoid_planner_autonomous.py` (copy from interactive)
2. Replace human feedback loop with autonomous loop
3. Implement `_autonomous_execution_loop()`
4. Add clarification mechanism (minimal human interaction)

### Phase 4: Prompt Updates
1. Update system prompt for autonomous mode
2. Add vision-based examples
3. Update response format validation

### Phase 5: Testing & Documentation
1. Create autonomous planner tests
2. Update examples for new mode
3. Document vision-based architecture
4. Update REPO_GUIDE.md

## API Requirements

### Qwen VLM API (DashScope)

**Endpoint**: `https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions`

**Request Format**:
```python
{
    "model": "qwen-vl-plus",
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/jpeg;base64,..." # or direct URL
                    }
                },
                {
                    "type": "text",
                    "text": "Describe this scene in detail."
                }
            ]
        }
    ]
}
```

**Response**:
```python
{
    "choices": [{
        "message": {
            "content": "I observe a living room with..."
        }
    }]
}
```

## Simulation Mode Support

For testing without real cameras/APIs:

1. **Mock Frames**: Use predefined images representing different states
2. **Mock VLM**: Return scripted observations based on action context
3. **State Tracking**: Maintain simulated world state (AC on/off, lights, etc.)

Example mock VLM:
```python
def mock_vlm_observation(action, state):
    if action == "control_air_conditioner" and state['ac'] == 'on':
        return "The AC display shows 'ON' with temperature set to 22°C. Cool air is flowing."
    elif action == "navigate_to_store":
        return "Robot is now in Room 02 (store). Shelves with various items visible."
    # ... etc
```

## Benefits of This Architecture

1. **Full Autonomy**: No manual intervention during execution
2. **Grounded in Reality**: Visual verification prevents hallucination
3. **Robust**: Can detect unexpected states through vision
4. **Scalable**: Easy to add new actions without changing feedback logic
5. **Debuggable**: Visual logs provide clear execution history

## Backward Compatibility

Keep both modes available:
- `humanoid_planner_interactive.py` - Human-in-loop (original)
- `humanoid_planner_autonomous.py` - Vision-based (new)

User can choose based on use case:
- **Interactive**: For teaching, demos, safety-critical tasks
- **Autonomous**: For routine tasks, testing, long-running operations

## Next Steps

Please review this design and let me know:
1. Any changes you'd like to the architecture?
2. Should I proceed with implementation?
3. Which VLM model should we use (qwen-vl-plus recommended)?
4. Do you have access to real cameras, or should I focus on simulation mode first?
