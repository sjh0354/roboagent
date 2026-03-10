# humanoid_executor_vision.py

"""
Vision-Enabled Humanoid Robot Executor
Extends HumanoidExecutor with vision-based observation capabilities
"""

import os
from typing import Dict, Any
from executor.humanoid_executor import HumanoidExecutor, ExecutionResult
from executor.vision_enabled_mixin import VisionEnabledMixin

try:
    from utils.dabai_camera_manager import DaBaiCameraManager
except ImportError:
    DaBaiCameraManager = None


class VisionEnabledExecutor(VisionEnabledMixin, HumanoidExecutor):
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
                 volume: float = 1.0,
                 voice: str = "female"):
        """
        Initialize vision-enabled executor

        Args:
            simulation_mode: If True, use simulation; if False, use real hardware
            verbose: Print execution details
            enable_vision: Enable vision-based observation
            vlm_model: VLM model to use (default: from env DEFAULT_VLM_MODEL or gemini-2.0-flash-exp)
            volume: TTS playback volume (0.0 to 1.0)
            voice: Voice tone (e.g., 'male', 'female')
        """
        # Initialize base executor
        super().__init__(simulation_mode=simulation_mode, verbose=verbose, volume=volume, voice=voice)

        self.enable_vision = enable_vision
        self.vlm_model = vlm_model
        self.vlm_client = None
        self.image_manager = None
        self.camera_manager = None

        if enable_vision:
            self._initialize_vision_components()

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
        base_result = super().execute_action(action_type, action_name, parameters)
        return self._augment_result_with_observation(base_result, action_type, action_name, parameters)

    def _create_camera_manager(self):
        if DaBaiCameraManager is None:
            raise ImportError("DaBaiCameraManager dependencies are not available")
        return DaBaiCameraManager(device_id=6, verbose=self.verbose)

    def _build_execution_result(self, success: bool, feedback: str, data=None, error: str = None):
        return ExecutionResult(success=success, feedback=feedback, data=data, error=error)

    def _web_search(self, url: str, query: str) -> ExecutionResult:
        """
        Perform web search using VLM client (Gemini Grounding).
        Overrides base HumanoidExecutor._web_search.
        """
        # If VLM client is available, use it for search
        if self.vlm_client:
            if self.verbose:
                print(f"🌐 VisionEnabledExecutor: Delegating web search '{query}' to VLM client...")
            
            try:
                # Use the new perform_web_search method
                search_result = self.vlm_client.perform_web_search(query)
                
                return ExecutionResult(
                    success=True,
                    feedback=f"Web search result for '{query}': {search_result}",
                    data={"query": query, "result": search_result, "source": "Gemini Grounding"}
                )
            except Exception as e:
                return ExecutionResult(
                    success=False,
                    feedback=f"Web search failed: {str(e)}",
                    error=str(e)
                )
        
        # Fallback to simulation mode behavior from base class if VLM not available
        return super()._web_search(url, query)


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
