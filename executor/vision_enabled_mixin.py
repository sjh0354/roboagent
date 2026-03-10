"""
Shared vision support for executor classes.
"""

import os
import time
from typing import Any, Dict, Optional

from utils.gemini_vlm_client import GeminiVLMClient
from utils.simulation_image_manager import SimulationImageManager


class VisionEnabledMixin:
    """Mixin for executors that augment actions with visual observation."""

    def _initialize_vision_components(self):
        self.vlm_client = None
        self.image_manager = None
        self.camera_manager = None

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
            time.sleep(1.0)
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

            if self.vlm_client and os.path.exists(observation_image):
                try:
                    base_result.data["vlm_observation"] = self.vlm_client.get_observation_description(observation_image)
                except Exception as error:
                    if self.verbose:
                        print(f"⚠️  VLM observation failed: {str(error)}")

        return base_result

    def get_current_observation(self) -> Dict[str, Any]:
        """Get current visual observation."""
        if not self.simulation_mode and self.camera_manager:
            observation_image = self.camera_manager.capture_image()
            if not observation_image:
                if self.verbose:
                    print("⚠️  Camera capture failed, using previous state or None")
                observation_image = None
        elif self.image_manager:
            observation_image = self.image_manager.get_current_observation_image()
        else:
            return {
                "image_path": None,
                "state": {},
                "vlm_description": "Vision system not initialized",
            }

        observation_data = {
            "image_path": observation_image,
            "state": self.image_manager.state.copy() if self.image_manager else {},
            "vlm_description": None,
        }

        if self.vlm_client and observation_image and os.path.exists(observation_image):
            try:
                observation_data["vlm_description"] = self.vlm_client.get_observation_description(observation_image)
            except Exception as error:
                if self.verbose:
                    print(f"⚠️  VLM observation failed: {str(error)}")

        return observation_data

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
