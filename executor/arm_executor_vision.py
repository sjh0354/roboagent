# arm_executor_vision.py

"""
Vision-Enabled Robotic Arm Executor
Extends ArmExecutor with vision-based observation capabilities
"""

import os
import time
import subprocess
import glob
import json
import shlex
from typing import Dict, Any
from executor.arm_executor import ArmExecutor, ExecutionResult
from executor.vision_enabled_mixin import VisionEnabledMixin

try:
    from utils.realsense_manager import RealSenseCameraManager
except ImportError:
    RealSenseCameraManager = None


class VisionEnabledArmExecutor(VisionEnabledMixin, ArmExecutor):
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
                 vlm_model: str = os.getenv("DEFAULT_VLM_MODEL", "gemini-2.0-flash-exp"),
                 volume: float = 1.0,
                 voice: str = "male"):
        """
        Initialize vision-enabled arm executor

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
        self.act_backend = os.getenv("ARM_ACT_BACKEND", "pi0").strip().lower()

    def execute_action(self, action_type: str, action_name: str, parameters: Dict[str, Any]) -> ExecutionResult:
        """
        Execute action with vision-based observation

        Flow:
        1. Execute action (using base executor or PI0 script for real hardware 'act')
        2. Capture observation AFTER action (RealSense or Simulation)
        3. Return enhanced ExecutionResult
        """
        
        # Intercept 'act' actions for real hardware execution using selected backend
        if not self.simulation_mode and action_type == "act":
            if self.act_backend == "anygrasp":
                base_result = self._execute_anygrasp_action(action_name, parameters)
            else:
                instruction = ""
                if action_name == "pick_and_place":
                    item_name = parameters.get("item_name", "item")
                    source = parameters.get("source", "shelf")
                    target = parameters.get("target", "counter")
                    # instruction = f"Move the {item_name} into the {target}."
                    instruction = f"pick up the {item_name} and place into {target}."
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

        return self._augment_result_with_observation(base_result, action_type, action_name, parameters)
    
    def _execute_pi0_script(self, instruction: str) -> ExecutionResult:
        """
        Execute PI0 inference script with instruction using subprocess inside Docker
        """
        base_dir = "/home/ef/projects/ur5e-arm-teleoperation"
        docker_dir = os.path.join(base_dir, "docker")
        container_name = "exp_ef_ur5e-arm-teleopration"
        script_in_docker = "/home/ef/projects/ur5e-arm-teleoperation/run_pi0_inference.sh"
        kill_script_in_docker = "/home/ef/projects/ur5e-arm-teleoperation/kill_project.sh"
        keep_session_on_fail = os.getenv("PI0_KEEP_SESSION_ON_FAIL", "0") == "1"
        should_cleanup = True

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

            # Ensure no stale tmux session from previous run remains inside container.
            # A leftover `tmp` session causes `duplicate session: tmp` and launch failure.
            stale_cleanup_cmd = [
                "docker", "exec",
                "-u", "1002",
                "-w", "/home/ef/projects/ur5e-arm-teleoperation",
                container_name,
                "bash", "-lc",
                "tmux has-session -t tmp 2>/dev/null && ./kill_project.sh || true",
            ]
            if self.verbose:
                print("   Cleaning stale Docker tmux session (if exists)...")
            subprocess.run(stale_cleanup_cmd, check=False)
            
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

            # Optional overrides for backend/network troubleshooting.
            for env_key in ["POLICY_HOST", "POLICY_PORT", "ROBOT_IP", "USE_REAL_ROBOT"]:
                env_val = os.environ.get(env_key)
                if env_val:
                    cmd[2:2] = ["-e", f"{env_key}={env_val}"]
            
            if self.verbose:
                print(f"   Running inside Docker: {' '.join(cmd)}")
            
            subprocess.run(cmd, check=True)
            
            # 3. Wait for action to complete (since script is non-blocking tmux)
            wait_time = 120
            if self.verbose:
                print(f"   ⏳ Waiting {wait_time}s for robot action to complete...")
            time.sleep(wait_time)

            log_path, steps = self._get_latest_pi0_action_steps(
                "/home/ef/projects/ur5e-arm-teleoperation/logs"
            )
            if self.verbose and log_path:
                print(f"   PI0 action log: {log_path}")
                print(f"   PI0 steps observed: {steps}")

            if steps <= 0:
                if keep_session_on_fail:
                    should_cleanup = False
                diagnostics = self._capture_container_pi0_pane(container_name)
                return ExecutionResult(
                    success=False,
                    feedback="PI0 launched but no robot control steps were produced.",
                    error=(
                        "Detected 0 PI0 steps. Likely no camera/joint-state data or PI0 node not ready. "
                        f"log={log_path or 'N/A'}"
                    ),
                    data={
                        "instruction": instruction,
                        "pi0_action_log": log_path,
                        "pi0_steps": steps,
                        "pi0_pane_tail": diagnostics,
                        "cleanup_skipped": keep_session_on_fail,
                    },
                )
            
            return ExecutionResult(
                success=True,
                feedback=f"Executed PI0 inference with instruction: '{instruction}'",
                data={
                    "instruction": instruction,
                    "pi0_action_log": log_path,
                    "pi0_steps": steps,
                },
            )

        except subprocess.CalledProcessError as e:
            return ExecutionResult(False, f"Script execution failed", error=str(e))
        except Exception as e:
            return ExecutionResult(False, f"Failed to execute script", error=str(e))
        finally:
            if not should_cleanup:
                if self.verbose:
                    print("   Debug mode: keeping Docker tmux session alive for inspection (PI0_KEEP_SESSION_ON_FAIL=1).")
            else:
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

    def _execute_anygrasp_action(self, action_name: str, parameters: Dict[str, Any]) -> ExecutionResult:
        if action_name != "pick_and_place":
            return super().execute_action("act", action_name, parameters)

        item_name = parameters.get("item_name", "item")
        source = parameters.get("source", "shelf")
        target = parameters.get("target", "counter")
        instruction = f"pick up the {item_name} from {source} and place into {target}."

        request_payload = {
            "action": "pick_and_place",
            "item_name": item_name,
            "source": source,
            "target": target,
            "instruction": instruction,
            "observation_image": self._capture_pre_action_observation(),
            "timestamp": time.time(),
        }

        return self._execute_anygrasp_runner(request_payload)

    def _capture_pre_action_observation(self) -> str:
        try:
            snapshot = self._capture_observation_snapshot(include_vlm_description=False)
            if isinstance(snapshot, dict):
                return snapshot.get("image_path") or ""
        except Exception:
            pass
        return ""

    def _execute_anygrasp_runner(self, request_payload: Dict[str, Any]) -> ExecutionResult:
        runner_cmd = os.getenv("ANYGRASP_RUNNER_CMD", "").strip()
        if not runner_cmd:
            default_runner = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "utils",
                "anygrasp_runner.py",
            )
            if os.path.exists(default_runner):
                runner_cmd = f"python {default_runner}"
        if not runner_cmd:
            return ExecutionResult(
                success=False,
                feedback="AnyGrasp backend selected but runner command is not configured.",
                error="missing_ANYGRASP_RUNNER_CMD",
                data={
                    "backend": "anygrasp",
                    "required_env": "ANYGRASP_RUNNER_CMD",
                    "request_payload": request_payload,
                },
            )

        timeout_seconds = int(os.getenv("ANYGRASP_RUNNER_TIMEOUT", "240"))
        command = shlex.split(runner_cmd) + ["--request-json", json.dumps(request_payload, ensure_ascii=False)]
        if self.verbose:
            print("🚀 Launching AnyGrasp backend runner...")
            print(f"   Command: {' '.join(command)}")

        try:
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                feedback="AnyGrasp runner timed out.",
                error=f"runner_timeout_{timeout_seconds}s",
                data={"backend": "anygrasp", "request_payload": request_payload},
            )
        except Exception as error:
            return ExecutionResult(
                success=False,
                feedback="Failed to start AnyGrasp runner.",
                error=str(error),
                data={"backend": "anygrasp", "request_payload": request_payload},
            )

        parsed = self._parse_anygrasp_runner_output(completed.stdout)
        stderr_text = (completed.stderr or "").strip()
        if completed.returncode != 0:
            return ExecutionResult(
                success=False,
                feedback="AnyGrasp runner returned non-zero exit code.",
                error=stderr_text or f"runner_exit_{completed.returncode}",
                data={
                    "backend": "anygrasp",
                    "returncode": completed.returncode,
                    "stdout": (completed.stdout or "").strip(),
                    "stderr": stderr_text,
                    "request_payload": request_payload,
                },
            )

        if parsed is None:
            return ExecutionResult(
                success=True,
                feedback="AnyGrasp runner completed (no JSON result; treat as success).",
                data={
                    "backend": "anygrasp",
                    "stdout": (completed.stdout or "").strip(),
                    "request_payload": request_payload,
                },
            )

        success = bool(parsed.get("success", True))
        feedback = str(parsed.get("feedback") or "AnyGrasp runner completed.")
        error = parsed.get("error")
        data = parsed.get("data") if isinstance(parsed.get("data"), dict) else {}
        data = {
            "backend": "anygrasp",
            "request_payload": request_payload,
            **data,
        }
        return ExecutionResult(success=success, feedback=feedback, data=data, error=error)

    def _parse_anygrasp_runner_output(self, stdout_text: str):
        text = (stdout_text or "").strip()
        if not text:
            return None
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            return None
        try:
            parsed = json.loads(lines[-1])
        except Exception:
            return None
        if isinstance(parsed, dict):
            return parsed
        return None

    def _get_latest_pi0_action_steps(self, log_dir: str):
        pattern = os.path.join(log_dir, "pi0_actions_*.log")
        candidates = glob.glob(pattern)
        if not candidates:
            return None, 0

        latest = max(candidates, key=os.path.getmtime)
        steps = 0
        try:
            with open(latest, "r", encoding="utf-8", errors="ignore") as file:
                for line in file:
                    if line.startswith("#") or not line.strip():
                        continue
                    steps += 1
        except Exception:
            return latest, 0

        return latest, steps

    def _capture_container_pi0_pane(self, container_name: str) -> str:
        try:
            cmd = [
                "docker",
                "exec",
                "-u",
                "1002",
                container_name,
                "tmux",
                "capture-pane",
                "-pt",
                "tmp:0.2",
                "-S",
                "-80",
            ]
            result = subprocess.run(cmd, check=False, capture_output=True, text=True)
            if result.returncode == 0:
                return (result.stdout or "").strip()
            return (result.stderr or "").strip()
        except Exception as error:
            return f"failed to capture pane: {error}"

    def _create_camera_manager(self):
        if RealSenseCameraManager is None:
            raise ImportError("RealSenseCameraManager dependencies are not available")
        return RealSenseCameraManager(verbose=self.verbose)

    def _initialize_simulation_state(self):
        if self.image_manager:
            self.image_manager.update_state(location="store")

    def _build_execution_result(self, success: bool, feedback: str, data=None, error: str = None):
        return ExecutionResult(success=success, feedback=feedback, data=data, error=error)

    def _web_search(self, url: str, query: str) -> ExecutionResult:
        """
        Perform web search using VLM client (Gemini Grounding).
        Overrides base ArmExecutor._web_search.
        """
        # If VLM client is available, use it for search
        if self.vlm_client:
            if self.verbose:
                print(f"🌐 VisionEnabledArmExecutor: Delegating web search '{query}' to VLM client...")
            
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
