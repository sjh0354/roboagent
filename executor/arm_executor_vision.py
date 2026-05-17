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
import socket
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
                 voice: str = "male",
                 act_backend: str = None):
        """
        Initialize vision-enabled arm executor

        Args:
            simulation_mode: If True, use simulation; if False, use real hardware
            verbose: Print execution details
            enable_vision: Enable vision-based observation
            vlm_model: VLM model to use (default: from env DEFAULT_VLM_MODEL or gemini-2.0-flash-exp)
            volume: TTS playback volume (0.0 to 1.0)
            voice: Voice tone (e.g., 'male', 'female')
            act_backend: Real-hardware act backend: vla/pi0, replay, anygrasp, or auto
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
        self.act_backend = self._normalize_act_backend(
            act_backend or os.getenv("ARM_ACT_BACKEND") or os.getenv("ARM_CONTROL_METHOD") or "vla"
        )

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
            selected_backend = self._select_backend_for_action()
            if selected_backend == "anygrasp":
                base_result = self._execute_anygrasp_action(action_name, parameters)
            else:
                instruction = self._build_action_instruction(action_name, parameters)
                if instruction:
                    if selected_backend == "replay":
                        base_result = self._execute_trajectory_replay_script(instruction)
                    elif selected_backend == "vla":
                        base_result = self._execute_pi0_script(instruction)
                    else:
                        base_result = ExecutionResult(
                            success=False,
                            feedback="Unsupported arm act backend.",
                            error=f"unknown_ARM_ACT_BACKEND={selected_backend}",
                            data={"instruction": instruction, "backend": selected_backend},
                        )
                else:
                    # Fallback to base execution if action unknown or no instruction
                    base_result = super().execute_action(action_type, action_name, parameters)
        else:
            # Execute using base executor for simulation or non-act actions
            base_result = super().execute_action(action_type, action_name, parameters)

        return self._augment_result_with_observation(base_result, action_type, action_name, parameters)

    def _normalize_act_backend(self, backend: str) -> str:
        value = (backend or "vla").strip().lower().replace("-", "_")
        aliases = {
            "pi0": "vla",
            "openpi": "vla",
            "vla": "vla",
            "auto": "vla",
            "trajectory": "replay",
            "trajectory_replay": "replay",
            "recorded": "replay",
            "recorded_trajectory": "replay",
            "replay": "replay",
            "anygrasp": "anygrasp",
        }
        return aliases.get(value, value)

    def _select_backend_for_action(self, quiet: bool = False) -> str:
        if self.act_backend != "vla":
            return self.act_backend
        if os.getenv("ARM_VLA_FALLBACK_TO_REPLAY", "1") == "0":
            return "vla"
        if self._is_vla_policy_server_available():
            return "vla"
        if self.verbose and not quiet:
            host, port = self._get_vla_policy_endpoint()
            print(f"⚠️  VLA policy server unavailable at {host}:{port}; falling back to recorded replay.")
        return "replay"

    def _get_vla_policy_endpoint(self):
        host = os.getenv("POLICY_HOST", "10.11.18.197")
        port = int(os.getenv("POLICY_PORT", "8000"))
        return host, port

    def _is_vla_policy_server_available(self) -> bool:
        host, port = self._get_vla_policy_endpoint()
        timeout_s = float(os.getenv("ARM_VLA_PORT_CHECK_TIMEOUT", "0.5"))
        try:
            with socket.create_connection((host, port), timeout=timeout_s):
                return True
        except OSError:
            return False
    
    def _build_action_instruction(self, action_name: str, parameters: Dict[str, Any]) -> str:
        """Build the task description passed to the recorded-trajectory router."""
        for key in ("instruction", "task_description", "description"):
            value = parameters.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()

        item_name = str(parameters.get("item_name", "item")).strip() or "item"
        item_name = self._normalize_replay_item_name(item_name)

        if action_name == "pick_and_place":
            source = parameters.get("source", "shelf")
            target = parameters.get("target", "counter")
            return f"pick up the {item_name} from {source} and place into {target}."
        if action_name == "pick_from_shelf":
            source = parameters.get("source", "shelf")
            return f"Pick the {item_name} from the {source}."
        if action_name == "place_on_counter":
            target = parameters.get("target", "counter")
            return f"Place the {item_name} on the {target}."

        return ""

    def _normalize_replay_item_name(self, item_name: str) -> str:
        lowered = item_name.lower().replace("_", " ")
        if "water" in lowered:
            return "water"
        medicine_terms = ("medicine", "medication", "pill", "drug", "green box")
        if any(term in lowered for term in medicine_terms):
            return "medicine"
        return item_name

    def _execute_trajectory_replay_script(self, instruction: str) -> ExecutionResult:
        """Execute a recorded UR5E trajectory selected from a task description."""
        base_dir = os.getenv("UR5E_TELEOP_WORKSPACE", "/home/ef/projects/ur5e-arm-teleoperation")
        docker_dir = os.path.join(base_dir, "docker")
        script_path = os.getenv(
            "TRAJECTORY_REPLAY_SCRIPT",
            os.path.join(base_dir, "run_recorded_trajectory_replay.sh"),
        )
        timeout_seconds = int(os.getenv("TRAJECTORY_REPLAY_TIMEOUT", "180"))
        start_container = os.getenv("TRAJECTORY_REPLAY_START_CONTAINER", "1") != "0"

        if not os.path.exists(script_path):
            return ExecutionResult(
                success=False,
                feedback="Trajectory replay script was not found.",
                error=f"missing_script={script_path}",
                data={"instruction": instruction, "backend": "trajectory_replay"},
            )

        env = os.environ.copy()
        if start_container and os.path.isdir(docker_dir):
            if self.verbose:
                print(f"   Starting/Checking Docker container in {docker_dir}...")
            try:
                self._remove_stopped_docker_container()
                subprocess.run(["make", "_instantiate_container"], cwd=docker_dir, env=env, check=True)
            except subprocess.CalledProcessError as error:
                return ExecutionResult(
                    success=False,
                    feedback="Failed to prepare Docker container for trajectory replay.",
                    error=str(error),
                    data={"instruction": instruction, "backend": "trajectory_replay"},
                )

        command = [script_path, "--task-description", instruction]
        extra_args = os.getenv("TRAJECTORY_REPLAY_EXTRA_ARGS", "").strip()
        if extra_args:
            command.extend(shlex.split(extra_args))

        if self.verbose:
            print("🚀 Launching recorded trajectory replay...")
            print(f"   Instruction: {instruction}")
            print(f"   Command: {' '.join(shlex.quote(part) for part in command)}")

        try:
            completed = subprocess.run(
                command,
                cwd=base_dir,
                env=env,
                check=False,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired as error:
            return ExecutionResult(
                success=False,
                feedback="Trajectory replay timed out.",
                error=f"trajectory_replay_timeout_{timeout_seconds}s",
                data={
                    "instruction": instruction,
                    "backend": "trajectory_replay",
                    "stdout": (error.stdout or "") if isinstance(error.stdout, str) else "",
                    "stderr": (error.stderr or "") if isinstance(error.stderr, str) else "",
                },
            )
        except Exception as error:
            return ExecutionResult(
                success=False,
                feedback="Failed to execute trajectory replay script.",
                error=str(error),
                data={"instruction": instruction, "backend": "trajectory_replay"},
            )

        stdout_text = (completed.stdout or "").strip()
        stderr_text = (completed.stderr or "").strip()
        result_data = {
            "instruction": instruction,
            "backend": "trajectory_replay",
            "command": command,
            "returncode": completed.returncode,
            "stdout": stdout_text,
            "stderr": stderr_text,
        }
        if completed.returncode != 0:
            return ExecutionResult(
                success=False,
                feedback="Trajectory replay failed.",
                error=stderr_text or f"trajectory_replay_exit_{completed.returncode}",
                data=result_data,
            )

        return ExecutionResult(
            success=True,
            feedback=f"Executed recorded trajectory replay for instruction: '{instruction}'",
            data=result_data,
        )

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
            self._remove_stopped_docker_container(container_name)
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
                        "backend": "vla",
                        "pi0_action_log": log_path,
                        "pi0_steps": steps,
                        "pi0_pane_tail": diagnostics,
                        "cleanup_skipped": keep_session_on_fail,
                        "retryable": True,
                        "failure_kind": "pi0_no_progress",
                    },
                )
            
            return ExecutionResult(
                success=True,
                feedback=f"Executed PI0 inference with instruction: '{instruction}'",
                data={
                    "instruction": instruction,
                    "backend": "vla",
                    "pi0_action_log": log_path,
                    "pi0_steps": steps,
                },
            )

        except subprocess.CalledProcessError as e:
            return ExecutionResult(
                False,
                "PI0 script execution failed",
                data={
                    "backend": "vla",
                    "instruction": instruction,
                    "retryable": True,
                    "failure_kind": "pi0_script_error",
                },
                error=str(e),
            )
        except Exception as e:
            return ExecutionResult(
                False,
                "Failed to execute PI0 script",
                data={
                    "backend": "vla",
                    "instruction": instruction,
                    "retryable": True,
                    "failure_kind": "pi0_exception",
                },
                error=str(e),
            )
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

    def _remove_stopped_docker_container(self, container_name: str = "exp_ef_ur5e-arm-teleopration") -> None:
        inspect_cmd = [
            "docker",
            "inspect",
            "-f",
            "{{.State.Running}}",
            container_name,
        ]
        result = subprocess.run(inspect_cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            return
        if result.stdout.strip().lower() == "true":
            return

        if self.verbose:
            print(f"   Removing stopped Docker container: {container_name}")
        subprocess.run(["docker", "rm", container_name], check=False)

    def _execute_anygrasp_action(self, action_name: str, parameters: Dict[str, Any]) -> ExecutionResult:
        if action_name != "pick_and_place":
            return super().execute_action("act", action_name, parameters)

        item_name = parameters.get("item_name", "item")
        source = parameters.get("source", "shelf")
        target = parameters.get("target", "counter")
        # instruction = f"pick up the {item_name} from {source} and place it into the {target}."
        instruction = f"pick up the {item_name} and place it into the {target}."

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
