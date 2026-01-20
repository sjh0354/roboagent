# humanoid_executor_vision.py

"""
Vision-Enabled Humanoid Robot Executor
Extends HumanoidExecutor with vision-based observation capabilities
"""

import os
import time
from typing import Dict, Any, Optional
from executor.humanoid_executor import HumanoidExecutor, ExecutionResult
from utils.gemini_vlm_client import GeminiVLMClient
from utils.simulation_image_manager import SimulationImageManager
from utils.dabai_camera_manager import DaBaiCameraManager


class VisionEnabledExecutor(HumanoidExecutor):
    """
    Vision-enabled executor for Unitree-G1 Humanoid Robot

    Extends base executor with:
    - Automatic visual observation capture
    - VLM-based action verification
    - State-aware image management
    """

    def __init__(self,
                 simulation_mode: bool = True,
                 verbose: bool = True,
                 enable_vision: bool = True,
                 vlm_model: str = os.getenv("DEFAULT_VLM_MODEL", "gemini-2.0-flash-exp"),
                 volume: float = 1.0):
        """
        Initialize vision-enabled executor

        Args:
            simulation_mode: If True, use simulation; if False, use real hardware
            verbose: Print execution details
            enable_vision: Enable vision-based observation
            vlm_model: VLM model to use (default: from env DEFAULT_VLM_MODEL or gemini-2.0-flash-exp)
            volume: TTS playback volume (0.0 to 1.0)
        """
        # Initialize base executor
        super().__init__(simulation_mode=simulation_mode, verbose=verbose, volume=volume)

        self.enable_vision = enable_vision
        self.vlm_model = vlm_model

        # Vision components
        self.vlm_client = None
        self.image_manager = None
        self.camera_manager = None

        if enable_vision:
            self._initialize_vision_system()

    def _initialize_vision_system(self):
        """Initialize vision components"""
        try:
            # Initialize VLM client
            api_key = os.getenv("GENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
            
            if api_key:
                self.vlm_client = GeminiVLMClient(
                    api_key=api_key,
                    model_name=self.vlm_model,
                    verbose=self.verbose
                )
                if self.verbose:
                    print(f"✅ VLM client initialized: {self.vlm_model}")
            else:
                if self.verbose:
                    print("⚠️  GENAI_API_KEY not set. VLM disabled (will use text descriptions)")
                self.enable_vision = False

            # Initialize DaBai camera if in real robot mode
            if not self.simulation_mode:
                try:
                    self.camera_manager = DaBaiCameraManager(device_id=6, verbose=self.verbose)
                except Exception as e:
                    print(f"⚠️  DaBai camera initialization failed: {str(e)}")
                    print("   Falling back to simulation image manager")

            # Initialize simulation image manager (always available as fallback/simulation)
            self.image_manager = SimulationImageManager(
                image_directory="simulation_images",
                verbose=self.verbose
            )

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Vision system initialization failed: {str(e)}")
                print("   Falling back to text-only mode")
            self.enable_vision = False

    def _get_observation(self) -> ExecutionResult:
        """Capture and analyze visual scene using real camera/VLM"""
        # Get raw observation data (image path, vlm description)
        obs_data = self.get_current_observation()
        
        image_path = obs_data.get('image_path')
        vlm_desc = obs_data.get('vlm_description')
        
        if not image_path:
             return ExecutionResult(
                success=False,
                feedback="Failed to capture observation image",
                error="Camera capture returned None"
            )

        # Construct feedback
        feedback = f"Observation captured: {image_path}"
        if vlm_desc:
            feedback += f"\nScene Description: {vlm_desc}"
        else:
            feedback += "\n(No VLM description available)"

        return ExecutionResult(
            success=True,
            feedback=feedback,
            data={
                "observation_image": image_path,
                "vlm_observation": vlm_desc,
                "observation": vlm_desc if vlm_desc else "Image captured but VLM analysis unavailable."
            }
        )

    def execute_action(self, action_type: str, action_name: str, parameters: Dict[str, Any]) -> ExecutionResult:
        """
        Execute action with vision-based observation

        Flow:
        1. Capture observation BEFORE action (if using vision)
        2. Execute action using base executor
        3. Capture observation AFTER action
        4. Use VLM to analyze result (if enabled)
        5. Return enhanced ExecutionResult

        Args:
            action_type: Type of action (talk, tool, act, sense)
            action_name: Specific action name
            parameters: Action parameters

        Returns:
            ExecutionResult with visual observation data
        """
        # Execute using base executor
        base_result = super().execute_action(action_type, action_name, parameters)

        # Handle observation capture
        observation_image = None
        
        # Case 1: Real robot mode with camera
        if not self.simulation_mode and self.camera_manager:
            # Wait for physical action to complete/settle
            time.sleep(1.0)
            observation_image = self.camera_manager.capture_image()
            
            # Still update simulation state tracker if available (for consistent state tracking)
            if self.image_manager:
                # We update state blindly since we can't infer state from image without VLM yet
                # This keeps the internal state machine roughly synchronized
                self.image_manager.get_observation_after_action(action_type, action_name, parameters)
        
        # Case 2: Simulation mode
        elif self.image_manager:
            observation_image = self.image_manager.get_observation_after_action(
                action_type,
                action_name,
                parameters
            )

        # Add vision data to result
        if observation_image:
            base_result.data['observation_image'] = observation_image
            if self.image_manager:
                base_result.data['state'] = self.image_manager.state.copy()

            # Optionally, use VLM to get observation if API is available
            if self.vlm_client and os.path.exists(observation_image):
                try:
                    vlm_observation = self.vlm_client.get_observation_description(observation_image)
                    # Store VLM observation in data for planner to use
                    base_result.data['vlm_observation'] = vlm_observation
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  VLM observation failed: {str(e)}")

        return base_result

    def get_current_observation(self) -> Dict[str, Any]:
        """
        Get current visual observation

        Returns:
            dict: {
                'image_path': str,
                'state': dict,
                'vlm_description': str (if VLM enabled)
            }
        """
        # Case 1: Real robot mode with camera
        if not self.simulation_mode and self.camera_manager:
            observation_image = self.camera_manager.capture_image()
            
            # If capture failed, fallback or return error state
            if not observation_image:
                 if self.verbose:
                    print("⚠️  RealSense capture failed, using previous state or None")
                 observation_image = None
        
        # Case 2: Simulation mode or fallback
        elif self.image_manager:
            observation_image = self.image_manager.get_current_observation_image()
            
        else:
            return {
                'image_path': None,
                'state': {},
                'vlm_description': 'Vision system not initialized'
            }

        observation_data = {
            'image_path': observation_image,
            'state': self.image_manager.state.copy() if self.image_manager else {},
            'vlm_description': None
        }

        # Get VLM description if available
        if self.vlm_client and observation_image and os.path.exists(observation_image):
            try:
                vlm_desc = self.vlm_client.get_observation_description(observation_image)
                observation_data['vlm_description'] = vlm_desc
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  VLM observation failed: {str(e)}")

        return observation_data

    def register_custom_observation_image(self, state_key: str, image_path: str):
        """
        Register a custom observation image for a specific state

        Args:
            state_key: State identifier (e.g., "home_ac_on_22")
            image_path: Path to image file
        """
        if self.image_manager:
            self.image_manager.register_custom_image(state_key, image_path)
        else:
            print("⚠️  Image manager not initialized")

    def reset_vision_state(self):
        """Reset vision system state"""
        if self.image_manager:
            self.image_manager.reset_state()

    def get_vision_statistics(self) -> Dict:
        """Get vision system statistics"""
        stats = {
            'vision_enabled': self.enable_vision,
            'vlm_model': self.vlm_model if self.vlm_client else None,
            'vlm_available': self.vlm_client is not None,
        }

        if self.image_manager:
            stats['state_summary'] = self.image_manager.get_state_summary()

        return stats


# Example usage
if __name__ == "__main__":
    print("🤖 Vision-Enabled Humanoid Executor - Test Mode\n")

    # Test with simulation mode
    print("="*70)
    print("Initializing Vision-Enabled Executor (Simulation Mode)")
    print("="*70)

    executor = VisionEnabledExecutor(
        simulation_mode=True,
        verbose=True,
        enable_vision=True
    )

    # Test action execution with vision
    print("\n" + "="*70)
    print("Test 1: Get Current Observation")
    print("="*70)
    obs = executor.get_current_observation()
    print(f"Observation Image: {obs['image_path']}")
    print(f"Current State: {obs['state']}")

    print("\n" + "="*70)
    print("Test 2: Execute Action (Turn on AC)")
    print("="*70)
    result = executor.execute_action("tool", "control_air_conditioner", {
        "action": "turn_on",
        "temperature": 22
    })
    print(f"Success: {result.success}")
    print(f"Feedback: {result.feedback}")
    print(f"Observation Image: {result.data.get('observation_image')}")
    print(f"New State: {result.data.get('state')}")

    print("\n" + "="*70)
    print("Test 3: Get Observation After Action")
    print("="*70)
    obs_after = executor.get_current_observation()
    print(f"Observation Image: {obs_after['image_path']}")
    print(f"Current State: {obs_after['state']}")

    # Show statistics
    print("\n" + "="*70)
    print("Vision System Statistics")
    print("="*70)
    import json
    stats = executor.get_vision_statistics()
    print(json.dumps(stats, indent=2))

    print("\n" + "="*70)
    print("Execution Complete")
    print("="*70)
    print("\nTo use with real VLM:")
    print("  1. Set GENAI_API_KEY environment variable")
    print("  2. Provide real observation images in simulation_images/")
    print("  3. VLM will automatically analyze images and enhance feedback")
    print("="*70)
