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
import time
from datetime import datetime
from typing import Dict, Optional

from executor.arm_executor_vision import VisionEnabledArmExecutor
# from utils.realsense_manager import RealSenseCameraManager # Removed: Handled by Executor
from template.arm_prompt_template_vlm import (
    get_arm_vlm_system_prompt,
    get_arm_legacy_system_prompt,
    get_arm_vlm_config,
    validate_arm_vlm_response,
    clean_json_response,
    list_available_vlm_models
)
from planner.base_vlm_planner import BaseVLMPlanner
from utils.funasr_manager import FunASRManager


class AutonomousArmVLMPlanner(BaseVLMPlanner):
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
                 model_name: str = os.getenv("DEFAULT_VLM_MODEL", "gemini-2.0-flash-exp"),
                 simulation_mode: bool = False,
                 verbose: bool = True,
                 volume: float = 1.0,
                 voice: str = "male"):
        """
        Initialize VLM-based autonomous arm planner

        Args:
            api_key: Google GenAI API key
            model_name: VLM model name (default: from env DEFAULT_VLM_MODEL or gemini-2.0-flash-exp)
            simulation_mode: Use simulation images (True) or real camera (False)
            verbose: Print detailed logs
            volume: TTS playback volume (0.0 to 1.0)
            voice: Voice tone (default: 'male')
        """
        self.verbose = verbose
        self.simulation_mode = simulation_mode
        super().__init__(
            profile_name="ur5e",
            api_key=api_key,
            model_name=model_name,
            verbose=verbose,
            config_getter=get_arm_vlm_config,
            system_prompt_getter=get_arm_vlm_system_prompt,
            legacy_prompt_getter=get_arm_legacy_system_prompt,
        )
        
        # Initialize Executor
        # Note: Executor handles Camera initialization internally. 
        # We must not initialize RealSenseCameraManager here to avoid "Device busy" errors.
        self.executor = VisionEnabledArmExecutor(
            simulation_mode=self.simulation_mode,
            verbose=verbose,
            enable_vision=True,
            vlm_model=model_name,
            volume=volume,
            voice=voice
        )
        self.reset_conversation()

        if verbose:
            print(f"✅ AutonomousArmVLMPlanner initialized with model: {self.config['model']}")
            print(f"   Simulation mode: {simulation_mode}")
            print(f"   Volume: {int(volume * 100)}%")

    def reset_conversation(self):
        super().reset_conversation()

    def _reset_runtime_state(self):
        self.waiting_for_humanoid = False
        self.humanoid_question = None

    def start_new_task(self, request: str, run_autonomously: bool = True,
                      observation_image: Optional[str] = None) -> Dict:
        self.current_observation_image = observation_image or self._get_default_observation_image()
        return super().start_new_task(request, run_autonomously)

    def _print_task_started(self, request: str):
        if self.verbose:
            print("\n" + "="*70)
            print("🦾 NEW ARM TASK STARTED (VLM Half-Open-Loop Mode)")
            print("="*70)
            print(f"📝 Request: {request}")
            print(f"🕐 Started at: {self.task_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⚙️  Mode: All actions assumed successful")
            print("="*70 + "\n")

    def _get_default_observation_image(self) -> str:
        """Get default observation image for store workspace"""
        # Delegate to executor
        if hasattr(self, 'executor'):
             obs = self.executor.get_current_observation()
             if obs.get('image_path'):
                 return obs['image_path']

        # Fallback for simulation or failure
        return "simulation_images/store/default.jpg"

    def provide_humanoid_response(self, response: str) -> Dict:
        return self.provide_input_response(response)

    def plan_next_step_with_image(self, image_path: str) -> Dict:
        return super().plan_next_step_with_image(image_path)

    def plan_next_step(self) -> Dict:
        return super().plan_next_step()

    def _prepare_current_observation(self) -> str:
        if self.verbose:
            print(f"\n📸 Current observation: {self.current_observation_image}")
        return self.current_observation_image

    def _get_current_image_for_planning(self) -> str:
        return self.current_observation_image

    def _handle_pause_prompt(self, just_spoke: bool, question: Optional[str]):
        if self.verbose:
            print(f"\n⏸️  PAUSED: Waiting for humanoid input")
            print(f"❓ Question: {question}")

    def _execute_step(self, step_plan: Dict):
        """
        Execute step, dispatching based on simulation mode.

        Args:
            step_plan: Step plan from VLM planner
        """
        if self.simulation_mode:
            self._execute_step_simulation(step_plan)
        else:
            self._execute_step_real(step_plan)

    def _execute_step_real(self, step_plan: Dict):
        """Execute step using Executor on real hardware"""
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

        if self.verbose:
            print("\n" + "="*70)
            print(f"⚡ EXECUTING STEP {next_step.get('step_number')} (Real Hardware)")
            print("="*70)
            print(f"🎯 Action: {next_step.get('action')}")
            print(f"📦 Parameters: {json.dumps(next_step.get('parameters', {}), indent=2)}")
            print("="*70 + "\n")

        # Execute via Executor
        result = self.executor.execute_action(
            next_step.get("action_type"),
            next_step.get("action"),
            next_step.get("parameters", {})
        )
        
        # Update observation image based on result
        if result.data.get("observation_image"):
             self.current_observation_image = result.data.get("observation_image")

        # Mark as assumed successful (half-open loop)
        self.execution_history[-1]["assumed_successful"] = True
        self.execution_history[-1]["execution_result"] = result.to_dict()

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"📊 EXECUTION STATUS:")
            print(f"{'='*70}")
            print(f"✓ Action executed on hardware")
            print(f"→ Continuing to next step...")
            print(f"{'='*70}\n")

    def _execute_step_simulation(self, step_plan: Dict):
        """
        Simulate step execution (Internal Planner Simulation Logic)

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
            print(f"⚡ EXECUTING STEP {next_step.get('step_number')} (Simulation Mode)")
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
        if action == "pick_and_place":
            item = params.get("item_name", "item")
            target = params.get("target", "counter")
            if target == "counter":
                self.current_observation_image = f"simulation_images/store/counter_{item}.jpg"
            else:
                self.current_observation_image = f"simulation_images/store/{target}_{item}.jpg"
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

    def _is_waiting_for_input(self) -> bool:
        return self.waiting_for_humanoid

    def _set_waiting_state(self, question: Optional[str]):
        self.waiting_for_humanoid = True
        self.humanoid_question = question

    def _clear_waiting_state(self):
        self.waiting_for_humanoid = False
        self.humanoid_question = None

    def _get_pending_question(self) -> Optional[str]:
        return self.humanoid_question

    def _get_waiting_status(self) -> str:
        return "waiting_for_humanoid"

    def _get_question_field(self) -> str:
        return "humanoid_question"

    def _get_input_role_name(self) -> str:
        return "humanoid"

    def _get_response_prefix(self) -> str:
        return "HUMANOID RESPONSE"

    def switch_model(self, model_name: str):
        """Switch to different VLM model"""
        old_model = self.config['model']
        self.config = get_arm_vlm_config(model_name)
        self.vlm_client.switch_model(model_name)
        if self.verbose:
            print(f"🔄 Switched VLM model: {old_model} → {model_name}")


def main():
    """Main function for testing"""
    import argparse
    import select

    parser = argparse.ArgumentParser(description="Autonomous arm VLM planner")
    parser.add_argument("--log", action="store_true", help="Show detailed planner/executor logs")
    args = parser.parse_args()
    verbose = args.log

    print("🦾 Autonomous VLM-Based Robotic Arm Planner")
    print("="*70)

    # Check API key
    if not os.getenv("GENAI_API_KEY"):
        print("⚠️  GENAI_API_KEY environment variable not set")
        print("Set it using: export GENAI_API_KEY='your-key'")
        return

    try:
        # Initialize planner
        planner = AutonomousArmVLMPlanner(
            model_name=os.getenv("DEFAULT_VLM_MODEL", "gemini-2.0-flash-exp"),
            simulation_mode=False,
            verbose=verbose
        )

        # Initialize Always-On ASR
        asr = FunASRManager(verbose=verbose)
        asr.start()

        print("\n" + "="*70)
        print("🎮 AUTONOMOUS ARM VLM PLANNER - Ready")
        print("="*70)
        print("📋 Commands:")
        print("  - Speak wake word (e.g. '你好机器人') followed by your request")
        print("  - Type request from humanoid robot directly")
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

            elif user_input.lower() in ['status', 's']:
                summary = planner.get_task_summary()
                print("\n📊 TASK STATUS:")
                if verbose:
                    print(json.dumps(summary, indent=2))
                else:
                    print(f"  request: {summary.get('original_request')}")
                    print(f"  steps: {summary.get('steps_executed')}")
                    print(f"  complete: {summary.get('is_complete')}")
                continue

            elif user_input.lower() in ['reset', 'r']:
                planner.reset_conversation()
                print("🔄 Reset complete. Ready for new task.")
                continue

            # Start new task
            result = planner.start_new_task(user_input, run_autonomously=True)

            # Handle result (Clarification)
            if result.get("status") == "waiting_for_humanoid":
                print(f"\n❓ {result['question']}")
                print("Your response (speak or type) > ", end="", flush=True)
                
                response = None
                while not response:
                    # Check voice
                    v_res = asr.get_command()
                    if v_res:
                        print(f"{v_res} (voice)")
                        response = v_res
                    
                    # Check keyboard
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if rlist:
                        response = sys.stdin.readline().strip()
                
                result = planner.provide_humanoid_response(response)

            # Show final result
            if result.get("is_complete"):
                print("\n✅ Task completed successfully!")
                if not verbose:
                    print(f"   steps: {result.get('steps_executed')}")
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
