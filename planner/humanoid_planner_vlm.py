# humanoid_planner_vlm.py

"""
Vision-Based Autonomous Humanoid Robot Planner
Uses VLM (Vision-Language Model) with direct visual observations for planning
"""

import os
import sys
# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime
from typing import Dict, Optional, List, Any

from template.humanoid_prompt_template_vlm import (
    get_humanoid_vlm_system_prompt,
    get_humanoid_vlm_config,
    validate_vlm_response,
    clean_json_response,
    list_available_vlm_models
)
from executor.humanoid_executor_vision import VisionEnabledExecutor
from utils.gemini_vlm_client import GeminiVLMClient
from utils.funasr_manager import FunASRManager


class AutonomousVLMPlanner:
    """
    Autonomous Humanoid Robot Planner with Vision-Language Model (Half-Open-Loop Mode)

    Key features:
    - VLM planner receives images directly (not text descriptions)
    - Autonomous execution loop without feedback waiting (half-open-loop)
    - All actions assumed successful
    - Human interaction only for: task requests, clarifications, completions
    """

    def __init__(self,
                 api_key: Optional[str] = None,
                 model_name: str = os.getenv("DEFAULT_VLM_MODEL", "gemini-2.0-flash-exp"),
                 simulation_mode: bool = True,
                 verbose: bool = True,
                 volume: float = 0.1,
                 voice: str = "female",
                 asr: Optional['FunASRManager'] = None):
        """
        Initialize VLM-based autonomous planner

        Args:
            api_key: Google GenAI API key
            model_name: VLM model name (default: from env DEFAULT_VLM_MODEL or gemini-2.0-flash-exp)
            simulation_mode: Use simulation images (True) or real camera (False)
            verbose: Print detailed logs
            volume: TTS playback volume (0.0 to 1.0)
            voice: Voice tone (default: 'female')
            asr: Optional FunASRManager for voice command detection during execution
        """
        # Store ASR reference for checking commands during execution
        self.asr = asr
        # Get API key
        self.api_key = api_key or os.getenv("GENAI_API_KEY")
        if not self.api_key:
            # Fallback
            self.api_key = os.getenv("DASHSCOPE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "API key not found. Set GENAI_API_KEY environment variable or provide api_key parameter."
            )

        # Configuration
        self.config = get_humanoid_vlm_config(model_name)
        self.verbose = verbose

        # System prompt
        self.system_prompt = get_humanoid_vlm_system_prompt()

        # Initialize VLM client
        self.vlm_client = GeminiVLMClient(
            api_key=self.api_key,
            model_name=model_name,
            verbose=verbose
        )

        # Initialize vision-enabled executor
        self.executor = VisionEnabledExecutor(
            simulation_mode=simulation_mode,
            verbose=verbose,
            enable_vision=True,
            vlm_model=model_name,
            volume=volume,
            voice=voice
        )

        # Conversation state
        self.reset_conversation()

        if verbose:
            print(f"✅ AutonomousVLMPlanner initialized with model: {self.config['model']}")
            print(f"   Simulation mode: {simulation_mode}")
            print(f"   Volume: {int(volume * 100)}%")

    def reset_conversation(self):
        """Reset conversation state for new task"""
        self.original_request = None
        self.conversation_history = []  # List of message dicts
        self.execution_history = []  # List of executed steps
        self.step_count = 0
        self.is_task_complete = False
        self.task_start_time = None
        self.waiting_for_human = False
        self.human_question = None
        self.pending_interrupt_command = None  # New command that interrupted current task
        self.current_location = "Room 01"  # Track location as state (always starts at Room 01)

        # Reset executor vision state to default
        if hasattr(self, 'executor'):
            self.executor.reset_vision_state()

    def start_new_task(self, human_request: str, run_autonomously: bool = True) -> Dict:
        """
        Start a new task from human request

        Args:
            human_request: Natural language task from human
            run_autonomously: If True, run full autonomous loop; if False, return first step only

        Returns:
            dict: Task result (completion summary or next step if not autonomous)
        """
        self.reset_conversation()
        self.original_request = human_request
        self.task_start_time = datetime.now()

        if self.verbose:
            print("\n" + "="*70)
            print("🤖 NEW TASK STARTED (VLM Half-Open-Loop Mode)")
            print("="*70)
            print(f"📝 Request: {human_request}")
            print(f"🕐 Started at: {self.task_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⚙️  Mode: All actions assumed successful")
            print("="*70 + "\n")

        if run_autonomously:
            return self._run_autonomous_loop()
        else:
            return self.plan_next_step()

    def set_asr(self, asr: 'FunASRManager'):
        """Set ASR manager for voice command detection during execution"""
        self.asr = asr

    def _run_autonomous_loop(self) -> Dict:
        """
        Run fully autonomous execution loop until completion or human input needed (Half-Open-Loop Mode)
        All actions are assumed to execute successfully without waiting for feedback.

        Returns:
            dict: Task completion summary or request for human input
        """
        while not self.is_task_complete and not self.waiting_for_human:
            # Check for new voice command (interrupt)
            if self.asr:
                new_cmd = self.asr.get_command()
                if new_cmd:
                    if self.verbose:
                        print(f"\n🎙️ New voice command detected during execution: {new_cmd}")
                    self.pending_interrupt_command = new_cmd
                    return {
                        "status": "interrupted",
                        "new_command": new_cmd,
                        "step_count": self.step_count
                    }

            # Get current observation image (reflects assumed state after last action)
            current_observation = self.executor.get_current_observation()
            current_image = current_observation['image_path']
            
            # Update location from observation state (Ground Truth)
            if 'state' in current_observation and 'location' in current_observation['state']:
                sim_location = current_observation['state']['location']
                # Map simulation location to Room ID
                location_map = {
                    "home": "Room 01",
                    "store": "Room 02"
                }
                self.current_location = location_map.get(sim_location, "Room 01")
                
            if self.verbose:
                print(f"\n📸 Current observation image: {current_image}")
                print(f"📍 Current location (from state): {self.current_location}")

            # Plan next step with image
            step_plan = self.plan_next_step_with_image(current_image)

            if step_plan.get("error"):
                # Error in planning
                if self.verbose:
                    print(f"❌ Planning error: {step_plan['error']}")
                return step_plan

            if step_plan.get("needs_human_input"):
                # If there is an action associated (e.g. speak), execute it first so the user hears/sees it
                just_spoke = False
                if step_plan.get("next_step"):
                     self._execute_step(step_plan)
                     # Check if we just executed a speak action
                     action = step_plan.get("next_step", {}).get("action")
                     if action in ["speak", "talk_with_human"]:
                        just_spoke = True

                # Pause for human clarification
                self.waiting_for_human = True
                self.human_question = step_plan.get("human_question")

                if self.verbose:
                    print(f"\n⏸️  PAUSED: Waiting for human input")
                    print(f"❓ Question: {self.human_question}")

                # Speak the question if we haven't just spoken it via an action
                if not just_spoke and self.human_question and hasattr(self.executor, 'tts_manager') and self.executor.tts_manager:
                     self.executor.tts_manager.speak(self.human_question, model="cosyvoice-v3-flash", block=False, voice="female", volume=0.1)

                return step_plan

            if step_plan.get("next_step") is None:
                # Task complete
                self.is_task_complete = True
                if self.verbose:
                    print("\n✅ TASK COMPLETE (Half-Open-Loop Mode)")
                return step_plan

            # Execute step (assumed successful, no verification)
            self._execute_step(step_plan)

        # Loop complete
        if self.is_task_complete:
            return self.get_task_summary()
        elif self.waiting_for_human:
            return {
                "status": "waiting_for_human",
                "question": self.human_question,
                "step_count": self.step_count
            }

    def provide_human_response(self, response: str) -> Dict:
        """
        Provide human's response to clarification question

        Args:
            response: Human's answer

        Returns:
            dict: Task result after continuing with response
        """
        if not self.waiting_for_human:
            return {"error": "Not waiting for human input"}

        if self.verbose:
            print(f"\n💬 Human response received: {response}")

        # Add response to conversation
        self.conversation_history.append({
            "role": "user",
            "content": f"[HUMAN RESPONSE]: {response}"
        })

        # Resume autonomous loop
        self.waiting_for_human = False
        self.human_question = None

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
            # Fallback: describe that image is not available
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

            response = self.vlm_client.create_chat_completion(
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
                # Task complete
                self.is_task_complete = True

            return step_plan

        except Exception as e:
            error_msg = f"Planning failed: {str(e)}"
            if self.verbose:
                print(f"❌ Error: {error_msg}")
            return {"error": error_msg}

    def plan_next_step(self) -> Dict:
        """
        Plan next step (convenience method, gets current observation automatically)

        Returns:
            dict: Next step plan
        """
        obs = self.executor.get_current_observation()
        return self.plan_next_step_with_image(obs['image_path'])

    def _execute_step(self, step_plan: Dict):
        """
        Execute a planned step (Half-Open-Loop: Assumed Successful)

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

        # Execute using vision-enabled executor
        execution_result = self.executor.execute_action(
            next_step.get("action_type"),
            next_step.get("action"),
            next_step.get("parameters", {})
        )

        # Add result to history (for logging purposes)
        self.execution_history[-1]["execution_result"] = execution_result.to_dict()
        # Mark as assumed successful
        self.execution_history[-1]["assumed_successful"] = True

        # Update location state after navigate_to action
        if next_step.get("action") == "navigate_to":
            target_location = next_step.get("parameters", {}).get("target_location")
            if target_location:
                old_location = self.current_location
                self.current_location = target_location
                if self.verbose:
                    print(f"📍 Location updated: {old_location} → {target_location}")

        # Display result (simplified - no verification)
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"📊 EXECUTION STATUS:")
            print(f"{'='*70}")
            print(f"✓ Action assumed successful (half-open-loop mode)")
            print(f"→ Continuing to next step...")
            print(f"{'='*70}\n")

    def _build_context_message(self) -> str:
        """Build context message for VLM planner"""
        context_parts = []

        # Original request
        context_parts.append(f"[ORIGINAL TASK REQUEST]: {self.original_request}")

        # Current location (state-based, automatically tracked)
        context_parts.append(f"\n[CURRENT LOCATION]: {self.current_location} (tracked automatically via navigate_to actions)")

        # Execution history
        if self.execution_history:
            context_parts.append(f"\n[STEPS EXECUTED SO FAR]: {len(self.execution_history)}")
            for exec_step in self.execution_history[-3:]:  # Last 3 steps
                result = exec_step.get("execution_result", {})
                context_parts.append(
                    f"  - Step {exec_step['step_number']}: {exec_step['action']} "
                    f"→ {'SUCCESS' if result.get('success') else 'FAILED'}"
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

        is_valid, message = validate_vlm_response(response_text)
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
            "execution_history": self.execution_history,
            "vision_statistics": self.executor.get_vision_statistics()
        }

    def register_observation_image(self, state_key: str, image_path: str):
        """
        Register custom observation image for simulation

        Args:
            state_key: State identifier (e.g., "home_ac_on_22")
            image_path: Path to image file
        """
        self.executor.register_custom_observation_image(state_key, image_path)

    def switch_model(self, model_name: str):
        """Switch to different VLM model"""
        old_model = self.config['model']
        self.config = get_humanoid_vlm_config(model_name)
        self.vlm_client.switch_model(model_name)
        if self.verbose:
            print(f"🔄 Switched VLM model: {old_model} → {model_name}")


def main():
    """Main function for testing"""
    import select
    
    print("🤖 Autonomous VLM-Based Humanoid Robot Planner")
    print("="*70)

    # Check API key
    if not os.getenv("GENAI_API_KEY"):
        print("⚠️  GENAI_API_KEY environment variable not set")
        print("Set it using: export GENAI_API_KEY='your-key'")
        print("\n💡 You can still test in simulation mode without VLM API,")
        print("   but planning will fail without the API key.")
        return

    try:
        # Initialize planner
        planner = AutonomousVLMPlanner(
            model_name=os.getenv("DEFAULT_VLM_MODEL", "gemini-2.0-flash-exp"),
            simulation_mode=False,
            verbose=True
        )

        # Initialize Always-On ASR
        asr = FunASRManager(verbose=True)
        asr.start()

        # Connect ASR to planner for interrupt detection
        planner.set_asr(asr)

        print("\n" + "="*70)
        print("🎮 AUTONOMOUS VLM PLANNER - Ready")
        print("="*70)
        print("📋 Commands:")
        print("  - Speak wake word (e.g. '你好机器人') followed by your request")
        print("  - Type task in natural language directly")
        print("  - 'models' or 'm': List available VLM models")
        print("  - 'switch <model>': Switch VLM model")
        print("  - 'status' or 's': Show task status")
        print("  - 'reset' or 'r': Reset for new task")
        print("  - 'quit' or 'q': Exit")
        print("="*70)

        while True:
            user_input = None
            
            # 1. Check for voice command (non-blocking)
            voice_cmd = asr.get_command()
            if voice_cmd:
                print(f"\n🎙️  Voice command detected: {voice_cmd}")
                user_input = voice_cmd
            
            # 2. Check for keyboard input (non-blocking)
            if not user_input:
                # Use select to check if stdin has data
                # Timeout of 0.1 seconds to keep the loop responsive
                rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                if rlist:
                    user_input = sys.stdin.readline().strip()
            
            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ['quit', 'q']:
                print("👋 Goodbye!")
                asr.stop()
                break

            elif user_input.lower() in ['models', 'm']:
                list_available_vlm_models()
                continue

            elif user_input.lower().startswith('switch '):
                model_name = user_input[7:].strip()
                planner.switch_model(model_name)
                continue

            elif user_input.lower().startswith('register '):
                parts = user_input[9:].split()
                if len(parts) == 2:
                    planner.register_observation_image(parts[0], parts[1])
                else:
                    print("Usage: register <state_key> <image_path>")
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

            # Handle interruption - start new task with the interrupting command
            while result.get("status") == "interrupted":
                new_cmd = result.get("new_command")
                print(f"\n🔄 Switching to new task: {new_cmd}")
                result = planner.start_new_task(new_cmd, run_autonomously=True)

            # Handle result (Clarification)
            while result.get("status") == "waiting_for_human":
                # For clarification, we also want to allow voice/keyboard
                print(f"\n❓ {result['question']}")
                print("Your response (speak or type) > ", end="", flush=True)
                
                response = None
                while not response:
                    # Check voice (use get_speech for responses - no wake word needed)
                    v_res = asr.get_speech()
                    if v_res:
                        print(f"{v_res} (voice)")
                        response = v_res

                    # Check keyboard
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if rlist:
                        response = sys.stdin.readline().strip()
                
                result = planner.provide_human_response(response)

            # Show final result
            if result.get("is_complete"):
                print("\n✅ Task completed successfully!")
            elif result.get("error"):
                print(f"\n❌ Error: {result['error']}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted. Goodbye!")
        if 'asr' in locals(): asr.stop()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if 'asr' in locals(): asr.stop()


if __name__ == "__main__":
    main()
