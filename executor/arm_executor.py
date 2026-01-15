# arm_executor.py

"""
Robotic Arm Executor - Real Hardware Integration
Executes planned actions on real UR5e robot or simulation
"""

import time
from datetime import datetime
from typing import Dict, Any, Optional
import json

try:
    from utils.tts_manager import TTSManager
except ImportError:
    TTSManager = None

class ExecutionResult:
    """Structured result from action execution"""

    def __init__(self, success: bool, feedback: str, data: Optional[Dict] = None, error: Optional[str] = None):
        self.success = success
        self.feedback = feedback
        self.data = data or {}
        self.error = error
        self.timestamp = datetime.now().isoformat()

    def to_dict(self):
        return {
            "success": self.success,
            "feedback": self.feedback,
            "data": self.data,
            "error": self.error,
            "timestamp": self.timestamp
        }

    def get_feedback_message(self):
        """Get human-readable feedback for the planner"""
        if self.success:
            return self.feedback
        else:
            return f"Error: {self.error}. {self.feedback}"


class ArmExecutor:
    """
    Executor for UR5e Robotic Arm

    Handles real-world execution of planned actions:
    - Communication (talk)
    - Manipulation (act: pick, place)
    - Perception (sense)
    """

    def __init__(self, simulation_mode=True, verbose=True, volume=1.0):
        """
        Initialize arm executor

        Args:
            simulation_mode: If True, simulate actions; if False, execute on real hardware
            verbose: Print execution details
            volume: TTS playback volume (0.0 to 1.0)
        """
        self.simulation_mode = simulation_mode
        self.verbose = verbose
        self.volume = volume

        # Hardware/API clients would be initialized here
        self.robot_controller = None

        # Execution statistics
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0

        # Initialize TTS Manager
        self.tts_manager = None
        if TTSManager:
            try:
                self.tts_manager = TTSManager(verbose=verbose, volume=volume)
            except Exception as e:
                if verbose:
                    print(f"⚠️  TTS Manager initialization failed: {e}")

        if not simulation_mode:
            self._initialize_hardware()

        if verbose:
            mode = "SIMULATION" if simulation_mode else "REAL HARDWARE"
            print(f"✅ ArmExecutor initialized in {mode} mode")

    def _initialize_hardware(self):
        """Initialize connections to real hardware/APIs"""
        # TODO: Initialize real hardware connections (e.g., RTDE for UR5e)
        # print("⚠️  Arm hardware initialization not yet implemented")

    def execute_action(self, action_type: str, action_name: str, parameters: Dict[str, Any]) -> ExecutionResult:
        """
        Execute an action and return structured result

        Args:
            action_type: Type of action (talk, act, sense)
            action_name: Specific action to execute
            parameters: Action parameters

        Returns:
            ExecutionResult with success status and feedback
        """
        self.execution_count += 1

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"🔧 ARM EXECUTOR: Executing {action_type}.{action_name}")
            print(f"📋 Parameters: {json.dumps(parameters, indent=2)}")
            print(f"{'='*70}\n")

        try:
            # Route to appropriate handler
            if action_type == "talk":
                result = self._execute_talk(action_name, parameters)
            elif action_type == "act":
                result = self._execute_act(action_name, parameters)
            elif action_type == "sense":
                result = self._execute_sense(action_name, parameters)
            else:
                result = ExecutionResult(
                    success=False,
                    feedback="Unknown action type",
                    error=f"Action type '{action_type}' not recognized"
                )

            # Update statistics
            if result.success:
                self.success_count += 1
            else:
                self.failure_count += 1

            if self.verbose:
                status = "✅ SUCCESS" if result.success else "❌ FAILED"
                print(f"{status}: {result.feedback}\n")

            return result

        except Exception as e:
            self.failure_count += 1
            error_result = ExecutionResult(
                success=False,
                feedback="Execution exception occurred",
                error=str(e)
            )
            if self.verbose:
                print(f"❌ EXCEPTION: {str(e)}\n")
            return error_result

    # ==================== TALK Actions ====================

    def _execute_talk(self, action: str, params: Dict) -> ExecutionResult:
        """Execute communication actions"""
        if action == "speak":
            return self._speak(params.get("message", ""))
        else:
            return ExecutionResult(
                success=False,
                feedback=f"Unknown talk action: {action}",
                error=f"Action '{action}' not implemented"
            )

    def _speak(self, message: str) -> ExecutionResult:
        """Speak/communicate (send status to humanoid)"""
        
        # Trigger TTS
        if self.tts_manager:
            self.tts_manager.speak(message, model="cosyvoice-v1", block=False)
            
        if self.simulation_mode:
            print(f"🦾 Arm says: \"{message}\"")
            return ExecutionResult(
                success=True,
                feedback=f"Message sent to humanoid: '{message}'",
                data={"message": message, "recipient": "humanoid"}
            )
        else:
            # Real implementation
            print(f"🗣️  Arm Speaking: \"{message}\"")
            return ExecutionResult(
                success=True,
                feedback=f"Message spoken: '{message}'",
                data={"message": message}
            )

    # ==================== ACT Actions ====================

    def _execute_act(self, action: str, params: Dict) -> ExecutionResult:
        """Execute physical robot actions"""

        if action == "pick_from_shelf":
            return self._pick_from_shelf(params.get("item_name", ""))

        elif action == "place_on_counter":
            return self._place_on_counter(params.get("item_name", ""))

        else:
            return ExecutionResult(
                success=False,
                feedback=f"Unknown act action: {action}",
                error=f"Action '{action}' not implemented"
            )

    def _pick_from_shelf(self, item_name: str) -> ExecutionResult:
        """Pick item from shelf"""
        if self.simulation_mode:
            print(f"🦾 Picking {item_name} from shelf...")
            time.sleep(1.0)  # Simulate movement
            return ExecutionResult(
                success=True,
                feedback=f"Successfully picked {item_name} from shelf",
                data={"action": "pick", "item": item_name}
            )
        else:
            # TODO: Real implementation (IK solver, motion planning)
            raise NotImplementedError("Real pick not yet implemented")

    def _place_on_counter(self, item_name: str) -> ExecutionResult:
        """Place item on counter"""
        if self.simulation_mode:
            print(f"🦾 Placing {item_name} on counter...")
            time.sleep(1.0)  # Simulate movement
            return ExecutionResult(
                success=True,
                feedback=f"Successfully placed {item_name} on counter",
                data={"action": "place", "item": item_name}
            )
        else:
            # TODO: Real implementation
            raise NotImplementedError("Real place not yet implemented")

    # ==================== SENSE Actions ====================

    def _execute_sense(self, action: str, params: Dict) -> ExecutionResult:
        """Execute perception actions"""
        _ = params

        if action == "get_observation":
            return self._get_observation()
        else:
            return ExecutionResult(
                success=False,
                feedback=f"Unknown sense action: {action}",
                error=f"Action '{action}' not implemented"
            )

    def _get_observation(self) -> ExecutionResult:
        """Capture visual observation"""
        if self.simulation_mode:
            print(f"📸 Capturing workspace observation...")
            time.sleep(0.5)
            # This would typically return an image path or description
            return ExecutionResult(
                success=True,
                feedback="Observation captured",
                data={"timestamp": datetime.now().isoformat()}
            )
        else:
            # Real implementation handles camera capture
            raise NotImplementedError("Real observation not implemented in base class")
