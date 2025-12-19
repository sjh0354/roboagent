# arm_planner_vlm.py

"""
Vision-Based Autonomous Robotic Arm Planner
Uses VLM (Vision-Language Model) with direct visual observations for planning
"""

import os
import sys
# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime
from typing import Dict, Optional
from openai import OpenAI

from template.arm_prompt_template_vlm import (
    get_arm_vlm_system_prompt,
    get_arm_vlm_config,
    validate_arm_vlm_response,
    clean_json_response,
    list_available_vlm_models
)
from utils.qwen_vlm_client import QwenVLMClient


class AutonomousArmVLMPlanner:
    """
    Autonomous Robotic Arm Planner with Vision-Language Model (Half-Open-Loop Mode)

    Key features:
    - VLM planner receives images directly (not text descriptions)
    - Autonomous execution loop without feedback waiting (half-open-loop)
    - All actions assumed successful
    - Humanoid interaction only for: task responses, clarifications, status updates
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 model_name: str = "qwen-vl-plus",
                 simulation_mode: bool = True,
                 verbose: bool = True):
        """
        Initialize VLM-based autonomous arm planner

        Args:
            api_key: DashScope API key
            model_name: VLM model name (qwen-vl-plus, qwen-vl-max)
            simulation_mode: Use simulation images (True) or real camera (False)
            verbose: Print detailed logs
        """
        # Get API key
        self.api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key not found. Set DASHSCOPE_API_KEY environment variable or provide api_key parameter."
            )

        # Configuration
        self.config = get_arm_vlm_config(model_name)
        self.verbose = verbose
        self.simulation_mode = simulation_mode

        # Initialize OpenAI client (for VLM API)
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.config["base_url"]
        )

        # System prompt
        self.system_prompt = get_arm_vlm_system_prompt()

        # Initialize VLM client (for helper operations)
        self.vlm_client = QwenVLMClient(
            api_key=self.api_key,
            model_name=model_name,
            verbose=verbose
        )

        # Conversation state
        self.reset_conversation()

        if verbose:
            print(f"✅ AutonomousArmVLMPlanner initialized with model: {self.config['model']}")
            print(f"   Simulation mode: {simulation_mode}")

    def reset_conversation(self):
        """Reset conversation state for new task"""
        self.original_request = None
        self.conversation_history = []  # List of message dicts
        self.execution_history = []  # List of executed steps
        self.step_count = 0
        self.is_task_complete = False
        self.task_start_time = None
        self.waiting_for_humanoid = False
        self.humanoid_question = None

    def start_new_task(self, request: str, run_autonomously: bool = True,
                      observation_image: Optional[str] = None) -> Dict:
        """
        Start a new task from humanoid robot request

        Args:
            request: Natural language task from humanoid
            run_autonomously: If True, run full autonomous loop; if False, return first step only
            observation_image: Initial observation image path (if None, uses simulation default)

        Returns:
            dict: Task result (completion summary or next step if not autonomous)
        """
        self.reset_conversation()
        self.original_request = request
        self.task_start_time = datetime.now()

        if self.verbose:
            print("\n" + "="*70)
            print("🦾 NEW ARM TASK STARTED (VLM Half-Open-Loop Mode)")
            print("="*70)
            print(f"📝 Request: {request}")
            print(f"🕐 Started at: {self.task_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⚙️  Mode: All actions assumed successful")
            print("="*70 + "\n")

        # Set initial observation image
        self.current_observation_image = observation_image or self._get_default_observation_image()

        if run_autonomously:
            return self._run_autonomous_loop()
        else:
            return self.plan_next_step()

    def _get_default_observation_image(self) -> str:
        """Get default observation image for store workspace"""
        if self.simulation_mode:
            # For simulation, use default store image
            return "simulation_images/store/default.jpg"
        else:
            # For real mode, would capture from camera
            raise NotImplementedError("Real camera capture not yet implemented")

    def _run_autonomous_loop(self) -> Dict:
        """
        Run fully autonomous execution loop until completion or humanoid input needed (Half-Open-Loop Mode)
        All actions are assumed to execute successfully without waiting for feedback.

        Returns:
            dict: Task completion summary or request for humanoid input
        """
        while not self.is_task_complete and not self.waiting_for_humanoid:
            if self.verbose:
                print(f"\n📸 Current observation: {self.current_observation_image}")

            # Plan next step with image
            step_plan = self.plan_next_step_with_image(self.current_observation_image)

            if step_plan.get("error"):
                if self.verbose:
                    print(f"❌ Planning error: {step_plan['error']}")
                return step_plan

            if step_plan.get("needs_human_input"):
                # Pause for humanoid clarification
                self.waiting_for_humanoid = True
                self.humanoid_question = step_plan.get("humanoid_question")

                if self.verbose:
                    print(f"\n⏸️  PAUSED: Waiting for humanoid input")
                    print(f"❓ Question: {self.humanoid_question}")

                return step_plan

            if step_plan.get("next_step") is None:
                # Task complete
                self.is_task_complete = True
                if self.verbose:
                    print("\n✅ TASK COMPLETE (Half-Open-Loop Mode)")
                return step_plan

            # Execute step (assumed successful, no verification)
            self._execute_step_simulation(step_plan)

        # Loop complete
        if self.is_task_complete:
            return self.get_task_summary()
        elif self.waiting_for_humanoid:
            return {
                "status": "waiting_for_humanoid",
                "question": self.humanoid_question,
                "step_count": self.step_count
            }

    def provide_humanoid_response(self, response: str) -> Dict:
        """
        Provide humanoid's response to clarification question

        Args:
            response: Humanoid's answer

        Returns:
            dict: Task result after continuing with response
        """
        if not self.waiting_for_humanoid:
            return {"error": "Not waiting for humanoid input"}

        if self.verbose:
            print(f"\n🤖 Humanoid response received: {response}")

        # Add response to conversation
        self.conversation_history.append({
            "role": "user",
            "content": f"[HUMANOID RESPONSE]: {response}"
        })

        # Resume autonomous loop
        self.waiting_for_humanoid = False
        self.humanoid_question = None

        return self._run_autonomous_loop()

    def plan_next_step_with_image(self, image_path: str) -> Dict:
        """
        Plan next step with visual observation image

        Args:
            image_path: Path to current observation image

        Returns:
            dict: Next step plan
        """
        if self.is_task_complete:
            if self.verbose:
                print("⚠️  Task already complete")
            return {"error": "Task already complete"}

        # Build context message
        context_text = self._build_context_message()

        # Prepare messages with image
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]

        # Add conversation history
        for msg in self.conversation_history:
            messages.append(msg)

        # Add current request with image
        user_content = []

        # Add image first
        if os.path.exists(image_path):
            user_content.append(
                self.vlm_client.create_image_message(image_path)
            )
        else:
            user_content.append({
                "type": "text",
                "text": f"[Image not available: {image_path}]"
            })

        # Add context text
        user_content.append({
            "type": "text",
            "text": context_text
        })

        messages.append({
            "role": "user",
            "content": user_content
        })

        # Call VLM API
        try:
            if self.verbose:
                print(f"\n🧠 Planning next step with VLM ({self.config['model']})...")

            response = self.client.chat.completions.create(
                model=self.config["model"],
                messages=messages,
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"]
            )

            response_text = response.choices[0].message.content

            # Parse response
            step_plan = self._parse_step_plan(response_text)

            # Add to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response_text
            })

            # Display step plan
            if step_plan.get("next_step"):
                self._display_step_plan(step_plan)
            elif step_plan.get("next_step") is None and not step_plan.get("needs_human_input"):
                self.is_task_complete = True

            return step_plan

        except Exception as e:
            error_msg = f"Planning failed: {str(e)}"
            if self.verbose:
                print(f"❌ Error: {error_msg}")
            return {"error": error_msg}

    def plan_next_step(self) -> Dict:
        """
        Plan next step (convenience method)

        Returns:
            dict: Next step plan
        """
        return self.plan_next_step_with_image(self.current_observation_image)

    def _execute_step_simulation(self, step_plan: Dict):
        """
        Simulate step execution (Half-Open-Loop: Assumed Successful)

        Args:
            step_plan: Step plan from VLM planner
        """
        if "error" in step_plan or step_plan.get("next_step") is None:
            return

        next_step = step_plan["next_step"]

        # Record in execution history
        self.execution_history.append({
            "step_number": next_step.get("step_number"),
            "action": next_step.get("action"),
            "parameters": next_step.get("parameters"),
            "timestamp": datetime.now().isoformat()
        })

        self.step_count += 1

        # Display execution header
        if self.verbose:
            print("\n" + "="*70)
            print(f"⚡ EXECUTING STEP {next_step.get('step_number')} (Half-Open-Loop)")
            print("="*70)
            print(f"🎯 Action: {next_step.get('action')}")
            print(f"📦 Parameters: {json.dumps(next_step.get('parameters', {}), indent=2)}")
            print(f"✓ Assumed: SUCCESS (no verification)")
            print("="*70 + "\n")

        # Update observation image based on action (simulation)
        self._update_observation_after_action(next_step)

        # Mark as assumed successful
        self.execution_history[-1]["assumed_successful"] = True

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"📊 EXECUTION STATUS:")
            print(f"{'='*70}")
            print(f"✓ Action assumed successful (half-open-loop mode)")
            print(f"→ Continuing to next step...")
            print(f"{'='*70}\n")

    def _update_observation_after_action(self, next_step: Dict):
        """Update observation image based on executed action (simulation)"""
        action = next_step.get("action")
        params = next_step.get("parameters", {})

        # Map actions to observation images (simulation)
        if action == "pick_from_shelf":
            item = params.get("item_name")
            self.current_observation_image = f"simulation_images/store/picked_{item}.jpg"
        elif action == "place_on_counter":
            item = params.get("item_name")
            self.current_observation_image = f"simulation_images/store/counter_{item}.jpg"
        elif action == "speak":
            # No visual change for communication
            pass
        elif action == "get_observation":
            # No visual change for observation request
            pass
        else:
            # Default: keep current observation
            pass

    def _build_context_message(self) -> str:
        """Build context message for VLM planner"""
        context_parts = []

        # Original request
        context_parts.append(f"[ORIGINAL REQUEST FROM HUMANOID]: {self.original_request}")

        # Execution history
        if self.execution_history:
            context_parts.append(f"\n[STEPS EXECUTED SO FAR]: {len(self.execution_history)}")
            for exec_step in self.execution_history[-3:]:
                context_parts.append(
                    f"  - Step {exec_step['step_number']}: {exec_step['action']} "
                    f"with {exec_step['parameters']}"
                )
        else:
            context_parts.append("\n[STEPS EXECUTED SO FAR]: None (this is the first step)")

        # Current status
        context_parts.append(f"\n[CURRENT STATUS]: Planning step #{self.step_count + 1}")

        # Instruction
        context_parts.append(
            "\n[INSTRUCTION]: Based on the visual observation (image provided above) and execution history, "
            "plan the NEXT SINGLE STEP. Use the JSON format specified in the system prompt."
        )

        return "\n".join(context_parts)

    def _parse_step_plan(self, response_text: str) -> Dict:
        """Parse VLM response into step plan"""
        cleaned_text = clean_json_response(response_text)

        is_valid, message = validate_arm_vlm_response(response_text)
        if not is_valid:
            if self.verbose:
                print(f"⚠️  Validation warning: {message}")

        try:
            return json.loads(cleaned_text)
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"❌ JSON parsing failed: {str(e)}")
            return {
                "error": f"JSON parsing failed: {str(e)}",
                "raw_response": response_text[:500]
            }

    def _display_step_plan(self, step_plan: Dict):
        """Display step plan in readable format"""
        if not self.verbose:
            return

        print("\n" + "="*70)
        print("📋 NEXT STEP PLAN (Half-Open-Loop)")
        print("="*70)

        # Analysis
        analysis = step_plan.get("current_step_analysis", {})
        print(f"\n🔍 ANALYSIS:")
        print(f"  Visual State: {analysis.get('visual_state', 'N/A')}")
        print(f"  Progress: {analysis.get('task_progress', 'N/A')}")
        print(f"  Reasoning: {analysis.get('next_action_reasoning', 'N/A')}")

        # Next step
        next_step = step_plan.get("next_step")
        if next_step:
            print(f"\n⚡ NEXT STEP #{next_step.get('step_number', '?')}:")
            print(f"  Action: {next_step.get('action')}")
            print(f"  Type: {next_step.get('action_type')}")
            print(f"  Parameters: {json.dumps(next_step.get('parameters', {}), indent=4)}")

        print("="*70 + "\n")

    def get_task_summary(self) -> Dict:
        """Get task completion summary"""
        duration = None
        if self.task_start_time:
            duration = (datetime.now() - self.task_start_time).total_seconds()

        return {
            "original_request": self.original_request,
            "steps_executed": self.step_count,
            "is_complete": self.is_task_complete,
            "duration_seconds": duration,
            "execution_history": self.execution_history
        }

    def switch_model(self, model_name: str):
        """Switch to different VLM model"""
        old_model = self.config['model']
        self.config = get_arm_vlm_config(model_name)
        self.vlm_client.switch_model(model_name)
        if self.verbose:
            print(f"🔄 Switched VLM model: {old_model} → {model_name}")


def main():
    """Main function for testing"""
    print("🦾 Autonomous VLM-Based Robotic Arm Planner")
    print("="*70)

    # Check API key
    if not os.getenv("DASHSCOPE_API_KEY"):
        print("⚠️  DASHSCOPE_API_KEY environment variable not set")
        print("Set it using: export DASHSCOPE_API_KEY='your-key'")
        return

    try:
        # Initialize planner
        planner = AutonomousArmVLMPlanner(
            model_name="qwen-vl-plus",
            simulation_mode=True,
            verbose=True
        )

        print("\n" + "="*70)
        print("🎮 AUTONOMOUS ARM VLM PLANNER - Ready")
        print("="*70)
        print("📋 Commands:")
        print("  - Enter request from humanoid robot")
        print("  - 'models' or 'm': List available VLM models")
        print("  - 'switch <model>': Switch VLM model")
        print("  - 'status' or 's': Show task status")
        print("  - 'reset' or 'r': Reset for new task")
        print("  - 'quit' or 'q': Exit")
        print("="*70)

        while True:
            user_input = input("\n🦾 Enter humanoid request > ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ['quit', 'q']:
                print("👋 Goodbye!")
                break

            elif user_input.lower() in ['models', 'm']:
                list_available_vlm_models()
                continue

            elif user_input.lower().startswith('switch '):
                model_name = user_input[7:].strip()
                planner.switch_model(model_name)
                continue

            elif user_input.lower() in ['status', 's']:
                summary = planner.get_task_summary()
                print("\n📊 TASK STATUS:")
                print(json.dumps(summary, indent=2))
                continue

            elif user_input.lower() in ['reset', 'r']:
                planner.reset_conversation()
                print("🔄 Reset complete. Ready for new task.")
                continue

            # Start new task
            result = planner.start_new_task(user_input, run_autonomously=True)

            # Handle result
            if result.get("status") == "waiting_for_humanoid":
                response = input(f"\n❓ {result['question']}\nYour response > ").strip()
                result = planner.provide_humanoid_response(response)

            # Show final result
            if result.get("is_complete"):
                print("\n✅ Task completed successfully!")
            elif result.get("error"):
                print(f"\n❌ Error: {result['error']}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted. Goodbye!")
    except Exception as e:
        print(f"❌ Error: {str(e)}")


if __name__ == "__main__":
    main()
