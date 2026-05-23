"""
Shared vision support for executor classes.
"""

import os
import time
from typing import Any, Dict, Optional

from utils.gemini_vlm_client import GeminiVLMClient
from utils.observation_buffer import ObservationBuffer, ObservationFrame, now_ts
from utils.simulation_image_manager import SimulationImageManager


class VisionEnabledMixin:
    """Mixin for executors that augment actions with visual observation."""

    def _initialize_vision_components(self):
        self.vlm_client = None
        self.image_manager = None
        self.camera_manager = None
        self.observation_buffer = ObservationBuffer(
            max_frames=int(os.getenv("TRANSIENT_MEMORY_BUFFER_MAX_FRAMES", "120"))
        )
        self.observation_sampling_interval = float(os.getenv("TRANSIENT_MEMORY_SAMPLE_INTERVAL", "5.0"))
        self._observation_buffer_thread = None
        self._observation_buffer_running = False

        try:
            api_key = os.getenv("GENAI_API_KEY") or os.getenv("DASHSCOPE_API_KEY")
            if api_key:
                self.vlm_client = GeminiVLMClient(
                    api_key=api_key,
                    model_name=self.vlm_model,
                    verbose=self.verbose,
                )
                if self.verbose:
                    print(f"✅ VLM client initialized: {self.vlm_model}")
            else:
                if self.verbose:
                    print("⚠️  GENAI_API_KEY not set. VLM disabled (will use text descriptions)")
                self.enable_vision = False

            if not self.simulation_mode:
                try:
                    self.camera_manager = self._create_camera_manager()
                except Exception as error:
                    print(f"⚠️  Camera initialization failed: {str(error)}")
                    print("   Falling back to simulation image manager")

            self.image_manager = SimulationImageManager(
                image_directory="simulation_images",
                verbose=self.verbose,
            )
            self._initialize_simulation_state()
            self._start_observation_buffer()
        except Exception as error:
            if self.verbose:
                print(f"⚠️  Vision system initialization failed: {str(error)}")
                print("   Falling back to text-only mode")
            self.enable_vision = False

    def _augment_result_with_observation(
        self,
        base_result,
        action_type: str,
        action_name: str,
        parameters: Dict[str, Any],
    ):
        observation_image = None

        if not self.simulation_mode and self.camera_manager:
            settle_seconds = float(os.getenv("POST_ACTION_OBSERVATION_SETTLE_SECONDS", "3.0"))
            if settle_seconds > 0:
                if getattr(self, "verbose", False):
                    print(f"   ⏳ Waiting {settle_seconds:.1f}s before post-action observation...")
                time.sleep(settle_seconds)
            observation_image = self.camera_manager.capture_image()
            if self.image_manager:
                self.image_manager.get_observation_after_action(action_type, action_name, parameters)
        elif self.image_manager:
            observation_image = self.image_manager.get_observation_after_action(
                action_type,
                action_name,
                parameters,
            )

        if observation_image:
            base_result.data["observation_image"] = observation_image
            if self.image_manager:
                base_result.data["state"] = self.image_manager.state.copy()

            describe_post_action = os.getenv("POST_ACTION_VLM_DESCRIPTION", "0") == "1"
            if describe_post_action and self.vlm_client and os.path.exists(observation_image):
                try:
                    base_result.data["vlm_observation"] = self.vlm_client.get_observation_description(observation_image)
                except Exception as error:
                    if self.verbose:
                        print(f"⚠️  VLM observation failed: {str(error)}")

        return base_result

    def get_current_observation(self, include_vlm_description: bool = False) -> Dict[str, Any]:
        """Get current visual observation."""
        observation_data = self._capture_observation_snapshot(
            include_vlm_description=include_vlm_description
        )
        if observation_data is None:
            return {
                "image_path": None,
                "state": {},
                "vlm_description": "Vision system not initialized",
            }
        return observation_data

    def get_buffered_observations_between(self, start_ts: float, end_ts: float, max_frames: int = 6):
        if not getattr(self, "observation_buffer", None):
            return []
        return self.observation_buffer.get_frames_between(start_ts, end_ts, max_frames=max_frames)

    def _get_observation(self):
        """Capture and analyze visual scene using camera/VLM."""
        obs_data = self.get_current_observation()
        image_path = obs_data.get("image_path")
        vlm_desc = obs_data.get("vlm_description")

        if not image_path:
            return self._build_execution_result(
                success=False,
                feedback="Failed to capture observation image",
                error="Camera capture returned None",
            )

        feedback = f"Observation captured: {image_path}"
        feedback += f"\nScene Description: {vlm_desc}" if vlm_desc else "\n(No VLM description available)"
        return self._build_execution_result(
            success=True,
            feedback=feedback,
            data={
                "observation_image": image_path,
                "vlm_observation": vlm_desc,
                "observation": vlm_desc if vlm_desc else "Image captured but VLM analysis unavailable.",
            },
        )

    def register_custom_observation_image(self, state_key: str, image_path: str):
        if self.image_manager:
            self.image_manager.register_custom_image(state_key, image_path)
        else:
            print("⚠️  Image manager not initialized")

    def reset_vision_state(self):
        if self.image_manager:
            self.image_manager.reset_state()

    def get_vision_statistics(self) -> Dict:
        stats = {
            "vision_enabled": self.enable_vision,
            "vlm_model": self.vlm_model if self.vlm_client else None,
            "vlm_available": self.vlm_client is not None,
        }
        if self.image_manager:
            stats["state_summary"] = self.image_manager.get_state_summary()
        return stats

    def close(self):
        self._stop_observation_buffer()
        if self.camera_manager:
            self.camera_manager.close()

    def _initialize_simulation_state(self):
        """Optional hook for subclasses to initialize simulation state."""
        return None

    def _create_camera_manager(self):
        raise NotImplementedError

    def _build_execution_result(
        self,
        success: bool,
        feedback: str,
        data: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ):
        raise NotImplementedError

    def _start_observation_buffer(self):
        if self._observation_buffer_running:
            return
        self._observation_buffer_running = True
        self._capture_buffer_frame(source="initial")
        import threading

        self._observation_buffer_thread = threading.Thread(
            target=self._observation_buffer_loop,
            daemon=True,
        )
        self._observation_buffer_thread.start()
        if self.verbose:
            print(
                f"🧠 Transient observation buffer started "
                f"(interval={self.observation_sampling_interval:.1f}s)"
            )

    def _stop_observation_buffer(self):
        self._observation_buffer_running = False
        if self._observation_buffer_thread:
            self._observation_buffer_thread.join(timeout=1.0)
            self._observation_buffer_thread = None

    def _observation_buffer_loop(self):
        while self._observation_buffer_running:
            time.sleep(self.observation_sampling_interval)
            if not self._observation_buffer_running:
                break
            self._capture_buffer_frame(source="interval")

    def _capture_buffer_frame(self, source: str = "buffer") -> Optional[ObservationFrame]:
        observation = self._capture_observation_snapshot(include_vlm_description=False, quiet=True)
        if not observation:
            return None
        frame = ObservationFrame(
            timestamp=now_ts(),
            image_path=observation.get("image_path"),
            state=observation.get("state") or {},
            source=source,
        )
        self.observation_buffer.add_frame(frame)
        return frame

    def _capture_observation_snapshot(
        self,
        include_vlm_description: bool = False,
        quiet: bool = False,
    ) -> Optional[Dict[str, Any]]:
        if not self.simulation_mode and self.camera_manager:
            observation_image = self.camera_manager.capture_image(quiet=quiet)
            if not observation_image:
                if self.verbose:
                    print("⚠️  Camera capture failed, using previous state or None")
                observation_image = None
        elif self.image_manager:
            observation_image = self.image_manager.get_current_observation_image()
        else:
            return None

        observation_data = {
            "image_path": observation_image,
            "state": self.image_manager.state.copy() if self.image_manager else {},
            "vlm_description": None,
        }

        if include_vlm_description and self.vlm_client and observation_image and os.path.exists(observation_image):
            try:
                observation_data["vlm_description"] = self.vlm_client.get_observation_description(observation_image)
            except Exception as error:
                if self.verbose:
                    print(f"⚠️  VLM observation failed: {str(error)}")

        return observation_data
