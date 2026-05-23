# arm_executor.py

"""
Robotic Arm Executor - Real Hardware Integration
Executes planned actions on real UR5e robot or simulation
"""

import time
from datetime import datetime
from typing import Dict, Any, Optional
import json
from utils.memory_manager import MemoryManager
from utils.mock_music_api import MockMusicAPI

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

    def __init__(self, simulation_mode=True, verbose=True, volume=1.0, voice="male"):
        """
        Initialize arm executor

        Args:
            simulation_mode: If True, simulate actions; if False, execute on real hardware
            verbose: Print execution details
            volume: TTS playback volume (0.0 to 1.0)
            voice: Voice tone (e.g., 'male', 'female')
        """
        self.simulation_mode = simulation_mode
        self.verbose = verbose
        self.volume = volume
        self.voice = voice
        self.message_transport = None
        self.memory_manager = MemoryManager(profile_name="ur5e", verbose=verbose)
        self.music_api = MockMusicAPI()

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
                self.tts_manager = TTSManager(verbose=verbose, volume=volume, voice=voice)
            except Exception as e:
                if verbose:
                    print(f"⚠️  TTS Manager initialization failed: {e}")

        if not simulation_mode:
            self._initialize_hardware()

        if verbose:
            mode = "SIMULATION" if simulation_mode else "REAL HARDWARE"
            print(f"✅ ArmExecutor initialized in {mode} mode (voice={voice})")

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
            elif action_type == "tool":
                result = self._execute_tool(action_name, parameters)
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
        elif action == "send_agent_message":
            return self._send_agent_message(
                params.get("message", ""),
                recipient=params.get("recipient"),
            )
        else:
            return ExecutionResult(
                success=False,
                feedback=f"Unknown talk action: {action}",
                error=f"Action '{action}' not implemented"
            )

    def set_message_transport(self, transport) -> None:
        self.message_transport = transport

    def _speak(self, message: str) -> ExecutionResult:
        """Speak/communicate (send status to humanoid)"""
        
        # Trigger TTS
        if self.tts_manager:
            self.tts_manager.speak(
                message,
                model="cosyvoice-v3-flash",
                block=False,
                voice=self.voice,
                volume=self.volume,
            )
        mirrored = self._mirror_speak_to_transport(message)
            
        if self.simulation_mode:
            print(f"🦾 Arm says: \"{message}\"")
            return ExecutionResult(
                success=True,
                feedback=f"Message sent to humanoid: '{message}'",
                data={"message": message, "recipient": "humanoid", "mirrored": mirrored}
            )
        else:
            # Real implementation
            print(f"🗣️  Arm Speaking: \"{message}\"")
            return ExecutionResult(
                success=True,
                feedback=f"Message spoken: '{message}'",
                data={"message": message, "mirrored": mirrored}
            )

    def _send_agent_message(self, message: str, recipient: Optional[str] = None) -> ExecutionResult:
        """Send a remote inter-agent message through the configured transport."""
        if not message:
            return ExecutionResult(
                success=False,
                feedback="Message is empty",
                error="empty_message",
            )

        if self.message_transport and getattr(self.message_transport, "can_send_messages", lambda: False)():
            result = self.message_transport.send_message(message, recipient=recipient)
            return ExecutionResult(
                success=result.success,
                feedback=result.feedback,
                data={
                    "message": message,
                    "recipient": recipient or "default",
                    "transport_message_id": result.message_id,
                    "transport": type(self.message_transport).__name__,
                    "raw": result.raw,
                },
                error=result.error,
            )

        return self._speak(message)

    def _mirror_speak_to_transport(self, message: str) -> bool:
        if not self.message_transport:
            return False
        if not getattr(self.message_transport, "should_mirror_local_speech", lambda: False)():
            return False
        result = self.message_transport.send_message(message)
        return bool(result.success)

    # ==================== TOOL Actions ====================

    def _execute_tool(self, action: str, params: Dict) -> ExecutionResult:
        """Execute tool/utility actions"""
        if action == "web_search":
            return self._web_search(
                params.get("URL", ""),
                params.get("query", "")
            )
        elif action == "store_memory":
            return self._store_memory(
                params.get("content", ""),
                scope=params.get("scope", "global"),
                category=params.get("category", "note"),
            )
        elif action == "control_light":
            return self._control_light(
                params.get("action", ""),
                device=params.get("device"),
                brightness=params.get("brightness"),
            )
        elif action == "play_audio":
            return self._play_audio(params)
        else:
            return ExecutionResult(
                success=False,
                feedback=f"Unknown tool action: {action}",
                error=f"Action '{action}' not implemented"
            )

    def _web_search(self, url: str, query: str) -> ExecutionResult:
        """Perform web search"""
        if self.simulation_mode:
            print(f"🔍 Searching '{query}' on {url}")
            # Simulate search results
            mock_results = f"Found information about '{query}': [Simulated search results]"
            return ExecutionResult(
                success=True,
                feedback=f"Web search completed for '{query}'",
                data={"query": query, "url": url, "results": mock_results}
            )
        else:
            # Base implementation doesn't support real search
            raise NotImplementedError("Real web search not yet implemented")

    def _store_memory(self, content: str, scope: str = "global", category: str = "note") -> ExecutionResult:
        """Persist a durable memory entry."""
        if not content:
            return ExecutionResult(
                success=False,
                feedback="Memory content is empty",
                error="empty_memory_content",
            )

        path = self.memory_manager.store_memory(content=content, scope=scope, category=category)
        if not path:
            return ExecutionResult(
                success=False,
                feedback="Memory write failed",
                error="memory_write_failed",
            )
        if not self.memory_manager.is_memory_entry_visible(path, content=content, category=category):
            return ExecutionResult(
                success=False,
                feedback="Memory write returned a path but the entry is not visible on disk",
                data={"content": content, "scope": scope, "category": category, "path": path},
                error="memory_write_not_visible",
            )

        return ExecutionResult(
            success=True,
            feedback=f"Stored {category} memory in {scope} scope",
            data={"content": content, "scope": scope, "category": category, "path": path},
        )

    def _control_light(
        self,
        action: str,
        device: Optional[str] = None,
        brightness: Optional[int] = None,
    ) -> ExecutionResult:
        """Control reading-related lighting through the experiment smart-home adapter."""
        target_device = device or "room_light"
        if action == "turn_on":
            if self.verbose:
                print(f"💡 Turning on {target_device}")
            return ExecutionResult(
                success=True,
                feedback=f"{target_device} turned on successfully",
                data={"light_status": "on", "device": target_device, "brightness": brightness},
            )
        if action == "turn_off":
            if self.verbose:
                print(f"💡 Turning off {target_device}")
            return ExecutionResult(
                success=True,
                feedback=f"{target_device} turned off successfully",
                data={"light_status": "off", "device": target_device, "brightness": brightness},
            )
        if action == "set_brightness":
            if not isinstance(brightness, int) or brightness < 0 or brightness > 100:
                return ExecutionResult(
                    success=False,
                    feedback="Invalid light brightness",
                    data={"device": target_device, "brightness": brightness},
                    error="Brightness must be an integer from 0 to 100",
                )
            if self.verbose:
                print(f"💡 Setting {target_device} brightness to {brightness}%")
            return ExecutionResult(
                success=True,
                feedback=f"{target_device} brightness set to {brightness}%",
                data={"light_status": "on", "device": target_device, "brightness": brightness},
            )
        return ExecutionResult(
            success=False,
            feedback=f"Invalid light action: {action}",
            error="Action must be 'turn_on', 'turn_off', or 'set_brightness'",
        )

    def _play_audio(self, payload: Dict[str, Any]) -> ExecutionResult:
        """Play background audio through the experiment audio API."""
        response = self.music_api.play(payload)
        if response.success:
            return ExecutionResult(
                success=True,
                feedback="Audio API request succeeded",
                data={"audio": response.data, "request_payload": payload},
            )
        return ExecutionResult(
            success=False,
            feedback="Audio API request failed",
            data={
                "request_payload": payload,
                "error_code": response.error_code,
                "error_message": response.error_message,
            },
            error=response.error_message,
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

        if action in {"get_observation", "observe_workspace"}:
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
