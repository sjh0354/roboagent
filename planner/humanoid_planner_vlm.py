# humanoid_planner_vlm.py

"""
Vision-Based Autonomous Humanoid Robot Planner
Uses VLM (Vision-Language Model) with direct visual observations for planning
"""

import os
import sys
import time
# Add parent directory to path to allow imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from datetime import datetime
from typing import Dict, Optional, List, Any

from template.humanoid_prompt_template_vlm import (
    get_humanoid_vlm_system_prompt,
    get_humanoid_legacy_system_prompt,
    get_humanoid_vlm_config,
    validate_vlm_response,
    clean_json_response,
    list_available_vlm_models
)
from planner.base_vlm_planner import BaseVLMPlanner
from executor.humanoid_executor_vision import VisionEnabledExecutor
from utils.funasr_manager import FunASRManager
from utils.message_transport import MessageTransport, VoiceTransport, create_message_transport


class AutonomousVLMPlanner(BaseVLMPlanner):
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
                 asr: Optional['FunASRManager'] = None,
                 transport: Optional[MessageTransport] = None):
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
        self.asr = asr
        self.transport = transport
        self.verbose = verbose
        super().__init__(
            profile_name="humanoid_g1",
            api_key=api_key,
            model_name=model_name,
            verbose=verbose,
            config_getter=get_humanoid_vlm_config,
            system_prompt_getter=get_humanoid_vlm_system_prompt,
            legacy_prompt_getter=get_humanoid_legacy_system_prompt,
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
        self.executor.set_message_transport(self.transport)
        self.reset_conversation()

        if verbose:
            print(f"✅ AutonomousVLMPlanner initialized with model: {self.config['model']}")
            print(f"   Simulation mode: {simulation_mode}")
            print(f"   Volume: {int(volume * 100)}%")

    def reset_conversation(self):
        super().reset_conversation()

    def _reset_runtime_state(self):
        self.waiting_for_human = False
        self.human_question = None
        self.pending_interrupt_command = None
        self.current_location = "Room 01"
        if hasattr(self, 'executor'):
            self.executor.reset_vision_state()

    def _print_task_started(self, request: str):
        if self.verbose:
            print("\n" + "="*70)
            print("🤖 NEW TASK STARTED (VLM Half-Open-Loop Mode)")
            print("="*70)
            print(f"📝 Request: {request}")
            print(f"🕐 Started at: {self.task_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"⚙️  Mode: All actions assumed successful")
            print("="*70 + "\n")

    def set_asr(self, asr: 'FunASRManager'):
        """Set ASR manager for voice command detection during execution"""
        self.asr = asr
        if self.transport is None:
            self.transport = VoiceTransport(asr=asr, verbose=self.verbose)
            self.executor.set_message_transport(self.transport)

    def set_transport(self, transport: Optional[MessageTransport]):
        self.transport = transport
        self.executor.set_message_transport(transport)

    def provide_human_response(self, response: str) -> Dict:
        return self.provide_input_response(response)

    def _check_interrupt(self) -> Optional[Dict]:
        new_cmd = None
        if self.transport:
            new_cmd = self.transport.get_command()
        elif self.asr:
            new_cmd = self.asr.get_command()

        if new_cmd:
            if self.verbose:
                print(f"\n🎙️ New command detected during execution: {new_cmd}")
            self.pending_interrupt_command = new_cmd
            return {
                "status": "interrupted",
                "new_command": new_cmd,
                "step_count": self.step_count
            }
        return None

    def _prepare_current_observation(self):
        current_observation = self.executor.get_current_observation()
        current_image = current_observation['image_path']
        if 'state' in current_observation and 'location' in current_observation['state']:
            sim_location = current_observation['state']['location']
            location_map = {
                "home": "Room 01",
                "store": "Room 02"
            }
            self.current_location = location_map.get(sim_location, "Room 01")

        if self.verbose:
            print(f"\n📸 Current observation image: {current_image}")
            print(f"📍 Current location (from state): {self.current_location}")
        if self.transient_memory_packet:
            packet = dict(self.transient_memory_packet)
            packet["primary_image"] = packet.get("primary_image") or current_image
            return packet
        return current_image

    def _get_current_image_for_planning(self) -> str:
        obs = self.executor.get_current_observation()
        return obs['image_path']

    def _handle_pause_prompt(self, just_spoke: bool, question: Optional[str]):
        if self.verbose:
            print(f"\n⏸️  PAUSED: Waiting for human input")
            print(f"❓ Question: {question}")

        if not just_spoke and question and hasattr(self.executor, 'tts_manager') and self.executor.tts_manager:
            self.executor.tts_manager.speak(question, model="cosyvoice-v3-flash", block=False, voice="female", volume=0.1)

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

        step_started_at = time.time()
        # Execute using vision-enabled executor
        execution_result = self.executor.execute_action(
            next_step.get("action_type"),
            next_step.get("action"),
            next_step.get("parameters", {})
        )
        step_finished_at = time.time()

        # Add result to history (for logging purposes)
        self.execution_history[-1]["execution_result"] = execution_result.to_dict()
        # Mark as assumed successful
        self.execution_history[-1]["assumed_successful"] = True
        self.execution_history[-1]["duration_seconds"] = round(step_finished_at - step_started_at, 2)

        self.transient_memory_packet = self._build_transient_memory_packet(
            action_name=next_step.get("action", "unknown"),
            start_ts=step_started_at,
            end_ts=step_finished_at,
            primary_image=execution_result.data.get("observation_image"),
        )

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

        transient_memory_context = self._build_transient_memory_context()
        if transient_memory_context:
            context_parts.append(f"\n{transient_memory_context}")

        # Instruction
        context_parts.append(
            "\n[INSTRUCTION]: Based on the visual observation (image(s) provided above) and execution history, "
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

    def _is_waiting_for_input(self) -> bool:
        return self.waiting_for_human

    def _set_waiting_state(self, question: Optional[str]):
        self.waiting_for_human = True
        self.human_question = question

    def _clear_waiting_state(self):
        self.waiting_for_human = False
        self.human_question = None

    def _get_pending_question(self) -> Optional[str]:
        return self.human_question

    def _get_waiting_status(self) -> str:
        return "waiting_for_human"

    def _get_question_field(self) -> str:
        return "human_question"

    def _get_input_role_name(self) -> str:
        return "human"

    def _get_response_prefix(self) -> str:
        return "HUMAN RESPONSE"

    def _get_current_location_for_prompt(self) -> Optional[str]:
        return self.current_location

    def _get_transient_memory_frames(self, start_ts: float, end_ts: float):
        if hasattr(self.executor, "get_buffered_observations_between"):
            return self.executor.get_buffered_observations_between(start_ts, end_ts, max_frames=6)
        return []

    def _get_summary_extras(self) -> Dict:
        return {"vision_statistics": self.executor.get_vision_statistics()}

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
    import argparse
    import select

    parser = argparse.ArgumentParser(description="Autonomous humanoid VLM planner")
    parser.add_argument("--simulation", action="store_true", help="Run in simulation mode using local simulation images")
    parser.add_argument("--log", action="store_true", help="Show detailed planner/executor logs")
    parser.add_argument(
        "--transport",
        default=os.getenv("INTERACTION_TRANSPORT", "voice"),
        help="Interaction transport: voice | lark | openclaw_lark | none",
    )
    parser.add_argument("--lark-target", default=os.getenv("LARK_TARGET") or os.getenv("OPENCLAW_LARK_TARGET"), help="Lark/Feishu target chat id")
    parser.add_argument("--lark-account", default=os.getenv("LARK_ACCOUNT_ID") or os.getenv("OPENCLAW_LARK_ACCOUNT"), help="Lark bot account id")
    args = parser.parse_args()
    verbose = args.log
    simulation_mode = args.simulation

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
        transport = create_message_transport(
            args.transport,
            verbose=verbose,
            target=args.lark_target,
            account_id=args.lark_account,
        )
        if transport:
            transport.start()

        planner = AutonomousVLMPlanner(
            model_name=os.getenv("DEFAULT_VLM_MODEL", "gemini-2.0-flash-exp"),
            simulation_mode=simulation_mode,
            verbose=verbose,
            transport=transport,
        )

        print("\n" + "="*70)
        print("🎮 AUTONOMOUS VLM PLANNER - Ready")
        print("="*70)
        print("📋 Commands:")
        print(f"  - Mode: {'simulation' if simulation_mode else 'real'}")
        print(f"  - Transport: {args.transport}")
        print("  - Speak wake word (e.g. '你好机器人') followed by your request when using voice transport")
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
            voice_cmd = transport.get_command() if transport else None
            if voice_cmd:
                print(f"\n🎙️  Transport command detected: {voice_cmd}")
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
                if transport:
                    transport.stop()
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
                if verbose:
                    print(json.dumps(summary, indent=2))
                else:
                    print(f"  request: {summary.get('original_request')}")
                    print(f"  steps: {summary.get('steps_executed')}")
                    print(f"  complete: {summary.get('is_complete')}")
                    print(f"  location: {planner.current_location}")
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
                    v_res = transport.get_response() if transport else None
                    if v_res:
                        print(f"{v_res} (transport)")
                        response = v_res

                    # Check keyboard
                    rlist, _, _ = select.select([sys.stdin], [], [], 0.1)
                    if rlist:
                        response = sys.stdin.readline().strip()
                
                result = planner.provide_human_response(response)

            # Show final result
            if result.get("is_complete"):
                print("\n✅ Task completed successfully!")
                if not verbose:
                    print(f"   steps: {result.get('steps_executed')}")
            elif result.get("error"):
                print(f"\n❌ Error: {result['error']}")

    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted. Goodbye!")
        if 'transport' in locals() and transport:
            transport.stop()
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        if 'transport' in locals() and transport:
            transport.stop()


if __name__ == "__main__":
    main()
