# humanoid_executor.py

"""
Humanoid Robot Executor - Real Hardware Integration
Executes planned actions on real Unitree-G1 robot
"""

import time
import subprocess
from datetime import datetime
from typing import Dict, Any, Optional
import json
import os

from utils.memory_manager import MemoryManager
from utils.mock_weather_api import MockWeatherAPI

# Dummy placeholder for SmartHomeAPI
class SmartHomeAPI:
    def control_ac(self, action: str, temperature: Optional[int]):
        print(f"⚠️  Dummy SmartHomeAPI: AC action '{action}' with temperature '{temperature}' called.")

    def control_light(self, action: str):
        print(f"⚠️  Dummy SmartHomeAPI: Light action '{action}' called.")

try:
    from utils.tts_manager import TTSManager
except ImportError:
    TTSManager = None

try:
    from utils.dabai_camera_manager import DaBaiCameraManager
except ImportError:
    DaBaiCameraManager = None

try:
    from utils.gemini_vlm_client import GeminiVLMClient
except ImportError:
    GeminiVLMClient = None


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


class HumanoidExecutor:
    """
    Executor for Unitree-G1 Humanoid Robot

    Handles real-world execution of planned actions:
    - Communication (talk)
    - Device control (tool)
    - Physical actions (act)
    - Perception (sense)
    """

    def __init__(self, simulation_mode=True, verbose=True, volume=1.0, voice="female"):
        """
        Initialize humanoid executor

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
        self.memory_manager = MemoryManager(profile_name="humanoid_g1", verbose=verbose)

        # Hardware/API clients will be initialized here
        self.robot_controller = None
        self.vision_system = None
        self.smart_home_controller = None

        # Mock weather backend for simulation experiments
        weather_api_version = os.getenv("WEATHER_API_VERSION", "v2")
        self.weather_api = MockWeatherAPI(api_version=weather_api_version)

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
            print(f"✅ HumanoidExecutor initialized in {mode} mode (voice={voice})")

    def _initialize_hardware(self):
        """Initialize connections to real hardware/APIs"""
        # TODO: Initialize real hardware connections
        # self.robot_controller = UnitreeG1Controller()
        # self.vision_system = VisionLanguageModel()
        if self.smart_home_controller is None: # Only initialize if not already set (e.g., by a mock in tests)
            self.smart_home_controller = SmartHomeAPI() # Instantiate the dummy SmartHomeAPI
        print("⚠️  Hardware initialization not yet implemented, using dummy SmartHomeAPI.")

    def execute_action(self, action_type: str, action_name: str, parameters: Dict[str, Any]) -> ExecutionResult:
        """
        Execute an action and return structured result

        Args:
            action_type: Type of action (talk, tool, act, sense)
            action_name: Specific action to execute
            parameters: Action parameters

        Returns:
            ExecutionResult with success status and feedback
        """
        self.execution_count += 1

        if self.verbose:
            print(f"\n{'='*70}")
            print(f"🔧 EXECUTOR: Executing {action_type}.{action_name}")
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

        # Legacy support for old action names (backward compatibility)
        elif action == "talk_with_human":
            return self._speak(params.get("message", ""))

        elif action == "request_item_from_store":
            item = params.get("item", "")
            return self._speak(f"Please get {item} for me")

        else:
            return ExecutionResult(
                success=False,
                feedback=f"Unknown talk action: {action}",
                error=f"Action '{action}' not implemented"
            )

    def set_message_transport(self, transport) -> None:
        self.message_transport = transport

    def _speak(self, message: str) -> ExecutionResult:
        """Speak/communicate (unified action for all communication)"""
        
        if self.simulation_mode:
            # Simulation: just display the message
            print(f"🤖 Robot says: \"{message}\"")
            
            # Optional: Play TTS in simulation too (non-blocking)
            if self.tts_manager:
                self.tts_manager.speak(message, model="cosyvoice-v3-flash", block=False)

            mirrored = self._mirror_speak_to_transport(message)

            # Detect if this is a store request based on message content
            is_store_request = any(keyword in message.lower() for keyword in ['get', 'please', 'request', 'need'])

            if is_store_request:
                time.sleep(0.5)  # Simulate network delay for store communication
                return ExecutionResult(
                    success=True,
                    feedback=f"Message sent: '{message}'. Store acknowledged.",
                    data={"message": message, "recipient": "store", "mirrored": mirrored}
                )
            else:
                return ExecutionResult(
                    success=True,
                    feedback=f"Message delivered: '{message}'",
                    data={"message": message, "recipient": "human", "mirrored": mirrored}
                )
        else:
            # Real implementation
            print(f"🗣️  Speaking: \"{message}\"")
            if self.tts_manager:
                self.tts_manager.speak(message, model="cosyvoice-v3-flash", block=True, voice=self.voice, volume=self.volume)
            mirrored = self._mirror_speak_to_transport(message)
            
            return ExecutionResult(
                success=True,
                feedback=f"Spoke message: '{message}'",
                data={"message": message, "recipient": "human", "mirrored": mirrored}
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

        # Backward-compatible fallback: verbalize when no remote transport exists.
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
        """Execute device control actions"""

        if action == "control_air_conditioner":
            return self._control_air_conditioner(
                params.get("action", ""),
                params.get("temperature")
            )

        elif action == "control_light":
            return self._control_light(params.get("action", ""))

        elif action == "web_search":
            return self._web_search(
                params.get("URL", ""),
                params.get("query", "")
            )

        elif action == "query_weather_api":
            return self._query_weather_api(params)

        elif action == "store_memory":
            return self._store_memory(
                params.get("content", ""),
                scope=params.get("scope", "global"),
                category=params.get("category", "note"),
            )

        else:
            return ExecutionResult(
                success=False,
                feedback=f"Unknown tool action: {action}",
                error=f"Action '{action}' not implemented"
            )

    def _control_air_conditioner(self, action: str, temperature: Optional[int]) -> ExecutionResult:
        """Control air conditioner"""
        if self.simulation_mode:
            if action == "turn_on":
                print(f"❄️  Turning on AC at {temperature}°C")
                return ExecutionResult(
                    success=True,
                    feedback=f"AC turned on successfully, temperature set to {temperature}°C",
                    data={"ac_status": "on", "temperature": temperature}
                )
            elif action == "turn_off":
                print(f"❄️  Turning off AC")
                return ExecutionResult(
                    success=True,
                    feedback="AC turned off successfully",
                    data={"ac_status": "off"}
                )
            else:
                return ExecutionResult(
                    success=False,
                    feedback=f"Invalid AC action: {action}",
                    error="Action must be 'turn_on' or 'turn_off'"
                )
        else:
            # TODO: Real implementation
            # - Smart home API (HomeKit, Google Home, custom protocol)
            self.smart_home_controller.control_ac(action, temperature)
            #raise NotImplementedError("Real AC control not yet implemented")
            return ExecutionResult(
                success=True,
                feedback=f"Real AC control executed: {action} {temperature if temperature else ''}",
                data={"ac_status": action, "temperature": temperature}
            )

    def _control_light(self, action: str) -> ExecutionResult:
        """Control lighting"""
        if self.simulation_mode:
            if action == "turn_on":
                print(f"💡 Turning on lights")
                return ExecutionResult(
                    success=True,
                    feedback="Lights turned on successfully",
                    data={"light_status": "on"}
                )
            elif action == "turn_off":
                print(f"💡 Turning off lights")
                return ExecutionResult(
                    success=True,
                    feedback="Lights turned off successfully",
                    data={"light_status": "off"}
                )
            else:
                return ExecutionResult(
                    success=False,
                    feedback=f"Invalid light action: {action}",
                    error="Action must be 'turn_on' or 'turn_off'"
                )
        else:
            # TODO: Real implementation
            # self.smart_home_controller.control_light(action)
            self.smart_home_controller.control_light(action)
            return ExecutionResult(
                success=True,
                feedback=f"Real light control executed: {action}",
                data={"light_status": action}
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
            # TODO: Real implementation
            # - Use requests library or browser automation
            # - Parse search results
            raise NotImplementedError("Real web search not yet implemented")

    def _query_weather_api(self, payload: Dict[str, Any]) -> ExecutionResult:
        """Call mock weather backend with payload and return structured result."""
        response = self.weather_api.query_weather(payload)
        if response.success:
            return ExecutionResult(
                success=True,
                feedback="Weather API query succeeded",
                data={
                    "weather": response.data,
                    "api_version": self.weather_api.api_version,
                    "request_payload": payload,
                },
            )

        return ExecutionResult(
            success=False,
            feedback="Weather API query failed",
            data={
                "api_version": self.weather_api.api_version,
                "request_payload": payload,
                "error_code": response.error_code,
                "error_message": response.error_message,
            },
            error=response.error_message,
        )

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

        return ExecutionResult(
            success=True,
            feedback=f"Stored {category} memory in {scope} scope",
            data={"content": content, "scope": scope, "category": category, "path": path},
        )

    # ==================== ACT Actions ====================

    def _execute_act(self, action: str, params: Dict) -> ExecutionResult:
        """Execute physical robot actions"""

        if action == "navigate_to":
            return self._navigate_to(
                params.get("target_location", ""),
                params.get("with_item", "none")
            )

        elif action == "pick":
            return self._pick(params.get("object_description", ""))

        elif action == "place":
            return self._place(
                params.get("receptacle_description", ""),
                params.get("spatial_relationship", "")
            )

        elif action == "wait_for":
            return self._wait_for(
                params.get("estimated_time", 30),
                params.get("reason", "")
            )

        # Legacy support for old action names (backward compatibility)
        elif action == "navigate_to_store":
            return self._navigate_to("Room 02", "none")

        elif action == "return_home_with_item":
            return self._navigate_to("Room 01", params.get("item", ""))

        elif action == "wait_for_item":
            return self._wait_for(params.get("estimated_time", 30), "Waiting for store to prepare item")

        else:
            return ExecutionResult(
                success=False,
                feedback=f"Unknown act action: {action}",
                error=f"Action '{action}' not implemented"
            )

    def _pick(self, object_description: str) -> ExecutionResult:
        """
        Pick up an object

        Args:
            object_description: Natural language description of the object
        """
        if self.simulation_mode:
            print(f"🦾 Picking up: {object_description}...")
            time.sleep(1.0)  # Simulate pick time
            return ExecutionResult(
                success=True,
                feedback=f"Successfully picked up {object_description}",
                data={"picked_object": object_description}
            )
        else:
            # Real implementation using ROS2 commands from guide
            if self.verbose:
                print(f"🚀 Executing REAL pick for: {object_description}")
            
            try:
                # Commands from humaniod_execution_guide.md:
                # cd /home/peanut/sc_ros_hzz
                # source install/setup.bash
                # ros2 run sc_ros2 pick
                cmd = "source install/setup.bash && ros2 run sc_ros2 pick"
                
                subprocess.run(cmd, shell=True, executable='/bin/bash', cwd='/home/peanut/sc_ros_hzz', check=True)
                
                return ExecutionResult(
                    success=True,
                    feedback=f"Real pick executed successfully for {object_description}",
                    data={"picked_object": object_description}
                )
            except subprocess.CalledProcessError as e:
                return ExecutionResult(
                    success=False,
                    feedback="Real pick failed",
                    error=str(e)
                )

    def _place(self, receptacle_description: str, spatial_relationship: str) -> ExecutionResult:
        """
        Place an object

        Args:
            receptacle_description: Description of the container or plane
            spatial_relationship: Relationship between object and container
        """
        if self.simulation_mode:
            print(f"🦾 Placing object {spatial_relationship} {receptacle_description}...")
            time.sleep(1.0)  # Simulate place time
            return ExecutionResult(
                success=True,
                feedback=f"Successfully placed object {spatial_relationship} {receptacle_description}",
                data={
                    "receptacle": receptacle_description,
                    "relationship": spatial_relationship
                }
            )
        else:
            # Real implementation using ROS2 commands from guide
            if self.verbose:
                print(f"🚀 Executing REAL place: {spatial_relationship} {receptacle_description}")
            
            try:
                # Commands from humaniod_execution_guide.md:
                # cd /home/peanut/sc_ros
                # source install/setup.bash
                # ros2 run sc_ros2 place
                cmd = "source install/setup.bash && ros2 run sc_ros2 place"
                
                subprocess.run(cmd, shell=True, executable='/bin/bash', cwd='/home/peanut/sc_ros_hzz', check=True)
                
                return ExecutionResult(
                    success=True,
                    feedback=f"Real place executed successfully {spatial_relationship} {receptacle_description}",
                    data={
                        "receptacle": receptacle_description,
                        "relationship": spatial_relationship
                    }
                )
            except subprocess.CalledProcessError as e:
                return ExecutionResult(
                    success=False,
                    feedback="Real place failed",
                    error=str(e)
                )

    def _navigate_to(self, target_location: str, with_item: str) -> ExecutionResult:
        """
        Navigate robot to specified location with or without item

        Args:
            target_location: "Room 01" (home) or "Room 02" (store)
            with_item: Item name being carried, or "none" if empty-handed
        """
        if self.simulation_mode:
            # Determine destination name
            location_map = {
                "Room 01": "home",
                "Room 02": "store"
            }
            destination = location_map.get(target_location, target_location)

            # Build feedback message
            if with_item and with_item != "none":
                print(f"🚶 Navigating to {destination} ({target_location}) carrying {with_item}...")
                feedback = f"Successfully arrived at {destination} carrying {with_item}"
                data = {"current_location": destination, "carrying_item": with_item, "travel_time": 1.0}
            else:
                print(f"🚶 Navigating to {destination} ({target_location})...")
                feedback = f"Successfully arrived at {destination}"
                data = {"current_location": destination, "carrying_item": "none", "travel_time": 1.0}

            time.sleep(1.0)  # Simulate travel time

            return ExecutionResult(
                success=True,
                feedback=feedback,
                data=data
            )
        else:
            # TODO: Real implementation
            # - SLAM/Navigation system
            # - Path planning
            # - Obstacle avoidance
            # - Grip stability monitoring if carrying item
            # self.robot_controller.navigate_to(target_location, carrying=with_item)
            time.sleep(40)  # Simulate travel time
            raise NotImplementedError("Real navigation not yet implemented")
            

    def _wait_for(self, estimated_time: int, reason: str) -> ExecutionResult:
        """
        Wait for specified duration with a reason

        Args:
            estimated_time: Duration in seconds
            reason: Explanation of why waiting
        """
        if self.simulation_mode:
            print(f"⏳ Waiting for {estimated_time}s... Reason: {reason}")
            time.sleep(min(estimated_time, 2))  # Simulate wait (capped for testing)
            return ExecutionResult(
                success=True,
                feedback=f"Wait completed ({estimated_time}s). {reason}",
                data={"wait_time": estimated_time, "reason": reason}
            )
        else:
            # TODO: Real implementation
            # - Just wait, maybe monitor robot state
            # - Could add timeout handling
            time.sleep(estimated_time)
            return ExecutionResult(
                success=True,
                feedback=f"Waited {estimated_time} seconds. {reason}",
                data={"wait_time": estimated_time, "reason": reason}
            )

    # ==================== Legacy Methods (Backward Compatibility) ====================

    def _navigate_to_store(self) -> ExecutionResult:
        """Legacy: Navigate to store (use navigate_to instead)"""
        return self._navigate_to("Room 02", "none")

    def _return_home_with_item(self, item: str) -> ExecutionResult:
        """Legacy: Return home with item (use navigate_to instead)"""
        return self._navigate_to("Room 01", item)

    def _wait_for_item(self, estimated_time: int) -> ExecutionResult:
        """Legacy: Wait for item (use wait_for instead)"""
        return self._wait_for(estimated_time, "Waiting for store to prepare item")

    # ==================== SENSE Actions ====================

    def _execute_sense(self, action: str, params: Dict) -> ExecutionResult:
        """Execute perception actions"""
        _ = params  # params not currently used for sense actions

        if action == "get_observation":
            return self._get_observation()

        else:
            return ExecutionResult(
                success=False,
                feedback=f"Unknown sense action: {action}",
                error=f"Action '{action}' not implemented"
            )

    def _get_observation(self) -> ExecutionResult:
        """Capture and analyze visual scene"""
        if self.simulation_mode:
            print(f"📸 Capturing scene observation...")
            time.sleep(0.5)  # Simulate camera + VLM processing

            # Mock observation based on context
            mock_observation = (
                "Living room scene: Human is sitting on couch. "
                "Room lighting is moderate. AC unit visible on wall (status unclear). "
                "No items visible on nearby tables."
            )

            return ExecutionResult(
                success=True,
                feedback=f"Observation: {mock_observation}",
                data={"observation": mock_observation, "processing_time": 0.5}
            )
        else:
            # TODO: Real implementation
            # - Capture image from robot camera
            # - Send to Vision-Language Model (VLM)
            # - Parse VLM response
            # image = self.robot_controller.capture_image()
            # observation = self.vision_system.analyze(image)
            raise NotImplementedError("Real vision system not yet implemented")

    # ==================== Utility Methods ====================

    def get_statistics(self) -> Dict:
        """Get execution statistics"""
        return {
            "total_executions": self.execution_count,
            "successful": self.success_count,
            "failed": self.failure_count,
            "success_rate": self.success_count / self.execution_count if self.execution_count > 0 else 0
        }

    def reset_statistics(self):
        """Reset execution statistics"""
        self.execution_count = 0
        self.success_count = 0
        self.failure_count = 0

    def switch_mode(self, simulation_mode: bool):
        """Switch between simulation and real hardware mode"""
        old_mode = "SIMULATION" if self.simulation_mode else "REAL HARDWARE"
        new_mode = "SIMULATION" if simulation_mode else "REAL HARDWARE"

        self.simulation_mode = simulation_mode

        if not simulation_mode and self.robot_controller is None:
            self._initialize_hardware()

        print(f"🔄 Executor mode switched: {old_mode} → {new_mode}")


# Example usage
if __name__ == "__main__":
    print("🤖 Humanoid Robot Executor - Standalone Test\n")

    # Create executor in simulation mode
    executor = HumanoidExecutor(simulation_mode=True, verbose=True)

    # Test different action types
    print("\n" + "="*70)
    print("Testing SENSE action")
    print("="*70)
    result = executor.execute_action("sense", "get_observation", {})
    print(f"Result: {result.get_feedback_message()}\n")

    print("\n" + "="*70)
    print("Testing TOOL action")
    print("="*70)
    result = executor.execute_action("tool", "control_air_conditioner", {
        "action": "turn_on",
        "temperature": 24
    })
    print(f"Result: {result.get_feedback_message()}\n")

    print("\n" + "="*70)
    print("Testing TALK action")
    print("="*70)
    result = executor.execute_action("talk", "talk_with_human", {
        "message": "The AC is now on at 24°C"
    })
    print(f"Result: {result.get_feedback_message()}\n")

    print("\n" + "="*70)
    print("Testing ACT action")
    print("="*70)
    result = executor.execute_action("act", "navigate_to_store", {})
    print(f"Result: {result.get_feedback_message()}\n")

    # Show statistics
    print("\n" + "="*70)
    print("Execution Statistics")
    print("="*70)
    stats = executor.get_statistics()
    print(json.dumps(stats, indent=2))
