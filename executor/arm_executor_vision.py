# arm_executor_vision.py

"""
Vision-Enabled Robotic Arm Executor
Extends ArmExecutor with vision-based observation capabilities
"""

import os
import time
import subprocess
from typing import Dict, Any, Optional
from executor.arm_executor import ArmExecutor, ExecutionResult
from utils.qwen_vlm_client import QwenVLMClient
from utils.simulation_image_manager import SimulationImageManager
from utils.realsense_manager import RealSenseCameraManager


class VisionEnabledArmExecutor(ArmExecutor):
    """
    Vision-enabled executor for UR5e Robotic Arm

    Extends base executor with:
    - RealSense camera integration for real-world observations
    - Simulation image management for testing
    - VLM-based verification support
    """

    def __init__(self,
                 simulation_mode: bool = True,
                 verbose: bool = True,
                 enable_vision: bool = True,
                 vlm_model: str = "qwen-vl-plus"):
        """
        Initialize vision-enabled arm executor

        Args:
            simulation_mode: If True, use simulation; if False, use real hardware
            verbose: Print execution details
            enable_vision: Enable vision-based observation
            vlm_model: VLM model to use (qwen-vl-plus, qwen-vl-max)
        """
        # Initialize base executor
        super().__init__(simulation_mode=simulation_mode, verbose=verbose)

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
            if os.getenv("DASHSCOPE_API_KEY"):
                self.vlm_client = QwenVLMClient(
                    model_name=self.vlm_model,
                    verbose=self.verbose
                )
                if self.verbose:
                    print(f"✅ VLM client initialized: {self.vlm_model}")
            else:
                if self.verbose:
                    print("⚠️  DASHSCOPE_API_KEY not set. VLM disabled (will use text descriptions)")
                self.enable_vision = False

            # Initialize RealSense camera if in real robot mode
            if not self.simulation_mode:
                try:
                    self.camera_manager = RealSenseCameraManager(verbose=self.verbose)
                except Exception as e:
                    print(f"⚠️  RealSense camera initialization failed: {str(e)}")
                    print("   Falling back to simulation image manager")

            # Initialize simulation image manager (always available as fallback/simulation)
            # Note: For arm, we primarily use the 'store' location context
            self.image_manager = SimulationImageManager(
                image_directory="simulation_images",
                verbose=self.verbose
            )
            # Set initial state for arm context
            self.image_manager.update_state(location="store")

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Vision system initialization failed: {str(e)}")
                print("   Falling back to text-only mode")
            self.enable_vision = False

    def execute_action(self, action_type: str, action_name: str, parameters: Dict[str, Any]) -> ExecutionResult:
        """
        Execute action with vision-based observation

        Flow:
        1. Execute action (using base executor or PI0 script for real hardware 'act')
        2. Capture observation AFTER action (RealSense or Simulation)
        3. Return enhanced ExecutionResult
        """
        
        # Intercept 'act' actions for real hardware execution using PI0 script
        if not self.simulation_mode and action_type == "act":
            instruction = ""
            if action_name == "pick_and_place":
                item_name = parameters.get("item_name", "item")
                source = parameters.get("source", "shelf")
                target = parameters.get("target", "counter")
                instruction = f"Move the {item_name} into the {target}."
            elif action_name == "pick_from_shelf":
                 item_name = parameters.get("item_name", "item")
                 instruction = f"Pick the {item_name} from the shelf."
            elif action_name == "place_on_counter":
                 item_name = parameters.get("item_name", "item")
                 instruction = f"Place the {item_name} on the counter."
            
            if instruction:
                base_result = self._execute_pi0_script(instruction)
            else:
                # Fallback to base execution if action unknown or no instruction
                base_result = super().execute_action(action_type, action_name, parameters)
        else:
            # Execute using base executor for simulation or non-act actions
            base_result = super().execute_action(action_type, action_name, parameters)

        # Handle observation capture
        observation_image = None
        
        # Case 1: Real robot mode with camera
        if not self.simulation_mode and self.camera_manager:
            # Wait for physical action to complete/settle
            time.sleep(1.0)
            observation_image = self.camera_manager.capture_image()
            
            # Still update simulation state tracker if available (for consistency)
            if self.image_manager:
                self.image_manager.get_observation_after_action(action_type, action_name, parameters)
        
        # Case 2: Simulation mode
        elif self.image_manager:
            # Note: SimulationImageManager might need updates to handle specific arm actions better,
            # but we use its generic state tracking for now.
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

            # Optionally, use VLM to get observation description
            if self.vlm_client and os.path.exists(observation_image):
                try:
                    vlm_observation = self.vlm_client.get_observation_description(observation_image)
                    base_result.data['vlm_observation'] = vlm_observation
                except Exception as e:
                    if self.verbose:
                        print(f"⚠️  VLM observation failed: {str(e)}")

        return base_result
    
    def _execute_pi0_script(self, instruction: str) -> ExecutionResult:
        """
        Execute PI0 inference script with instruction using subprocess inside Docker
        """
        base_dir = "/home/ef/projects/ur5e-arm-teleoperation"
        docker_dir = os.path.join(base_dir, "docker")
        container_name = "exp_ef_ur5e-arm-teleopration"
        script_in_docker = "/home/ef/projects/ur5e-arm-teleoperation/run_pi0_inference.sh"
        kill_script_in_docker = "/home/ef/projects/ur5e-arm-teleoperation/kill_project.sh"

        try:
            if self.verbose:
                print(f"🚀 Launching PI0 inference in Docker...")
                print(f"   Instruction: {instruction}")

            # Prepare environment variables for make command
            env = os.environ.copy()
            env["INSTRUCTION"] = instruction
            
            # 1. Run make in docker dir to ensure container is running
            if self.verbose:
                print(f"   Starting/Checking Docker container in {docker_dir}...")
            subprocess.run(["make", "_instantiate_container"], cwd=docker_dir, env=env, check=True)
            
            # 2. Run inference script INSIDE Docker
            # Command: docker exec -u 1002 -e INSTRUCTION="..." -w ... <container> ./run_pi0_inference.sh
            cmd = [
                "docker", "exec",
                "-e", f"INSTRUCTION={instruction}",
                "-u", "1002",
                "-w", "/home/ef/projects/ur5e-arm-teleoperation",
                container_name,
                "./run_pi0_inference.sh"
            ]
            
            if self.verbose:
                print(f"   Running inside Docker: {' '.join(cmd)}")
            
            subprocess.run(cmd, check=True)
            
            # 3. Wait for action to complete (since script is non-blocking tmux)
            wait_time = 60
            if self.verbose:
                print(f"   ⏳ Waiting {wait_time}s for robot action to complete...")
            time.sleep(wait_time)
            
            return ExecutionResult(
                success=True,
                feedback=f"Executed PI0 inference with instruction: '{instruction}'",
                data={"instruction": instruction}
            )

        except subprocess.CalledProcessError as e:
            return ExecutionResult(False, f"Script execution failed", error=str(e))
        except Exception as e:
            return ExecutionResult(False, f"Failed to execute script", error=str(e))
        finally:
            # 4. Cleanup INSIDE Docker
            if self.verbose:
                print(f"   Cleaning up processes inside Docker...")
            
            cleanup_cmd = [
                "docker", "exec",
                "-u", "1002",
                "-w", "/home/ef/projects/ur5e-arm-teleoperation",
                container_name,
                "./kill_project.sh"
            ]
            
            try:
                subprocess.run(cleanup_cmd, check=False)
            except Exception as e:
                print(f"Error during cleanup: {e}")
            
            # 5. Stop Docker container (optional, maybe keep it running for speed?)
            # Keeping it running is faster for subsequent commands. 
            # If we want to strictly stop it:
            # subprocess.run(["make", "kill"], cwd=docker_dir, check=False)

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

    def close(self):
        """Cleanup resources"""
        if self.camera_manager:
            self.camera_manager.close()


# Example usage
if __name__ == "__main__":
    print("🦾 Vision-Enabled Arm Executor - Test Mode\n")

    # Test with simulation mode
    executor = VisionEnabledArmExecutor(
        simulation_mode=True,
        verbose=True,
        enable_vision=True
    )

    # Test observation
    obs = executor.get_current_observation()
    print(f"Observation: {obs['image_path']}")

    # Test pick action
    result = executor.execute_action("act", "pick_from_shelf", {"item_name": "water"})
    print(f"Pick Result: {result.success}")
    print(f"Image after action: {result.data.get('observation_image')}")
