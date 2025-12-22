# simulation_image_manager.py

"""
Simulation Image Manager
Manages mock observation images for different robot states in simulation mode
"""

import os
from typing import Dict, Optional, Any
from datetime import datetime
import json


class SimulationImageManager:
    """
    Manages observation images for simulation mode

    Supports:
    - State-based image selection (location, device states, etc.)
    - User-provided custom images for key states
    - Automatic state tracking
    - Fallback to default images
    """

    def __init__(self, image_directory: str = "simulation_images", verbose: bool = True):
        """
        Initialize simulation image manager

        Args:
            image_directory: Directory containing simulation images
            verbose: Print debug information
        """
        self.image_directory = image_directory
        self.verbose = verbose

        # Current robot state
        self.state = {
            "location": "home",  # home, store, in_transit
            "ac_status": "off",
            "ac_temperature": None,
            "light_status": "off",
            "carrying_item": None,
            "human_present": True
        }

        # Image mapping: state -> image path
        self.image_map = {}

        # History of state changes
        self.state_history = []

        # Initialize image directory
        self._initialize_image_directory()

        # Load default image mappings
        self._load_default_mappings()

        if verbose:
            print(f"✅ SimulationImageManager initialized")
            print(f"   Image directory: {self.image_directory}")

    def _initialize_image_directory(self):
        """Create image directory if it doesn't exist"""
        if not os.path.exists(self.image_directory):
            os.makedirs(self.image_directory)
            if self.verbose:
                print(f"📁 Created image directory: {self.image_directory}")

            # Create README
            readme_path = os.path.join(self.image_directory, "README.md")
            with open(readme_path, "w") as f:
                f.write("""# Simulation Images

This directory contains observation images for different robot states.

## Directory Structure

```
simulation_images/
├── home/
│   ├── default.jpg              # Default home view
│   ├── ac_on_22.jpg            # AC on at 22°C
│   ├── ac_on_24.jpg            # AC on at 24°C
│   ├── ac_off.jpg              # AC off
│   ├── light_on.jpg            # Light on
│   └── light_off.jpg           # Light off
├── store/
│   ├── default.jpg              # Default store view
│   ├── robot_present.jpg       # Humanoid at store
│   └── item_ready.jpg          # Item on counter
├── in_transit/
│   ├── to_store.jpg            # Navigating to store
│   └── to_home.jpg             # Returning home
└── README.md                    # This file
```

## Adding Custom Images

1. Place images in appropriate subdirectories
2. Use descriptive names matching states
3. Supported formats: jpg, jpeg, png

## Image Naming Convention

Format: `{state_key}_{state_value}.jpg`

Examples:
- `ac_on_22.jpg` → AC on at 22°C
- `light_on.jpg` → Light on
- `robot_present.jpg` → Robot at location
""")

            # Create subdirectories
            for subdir in ["home", "store", "in_transit"]:
                subdir_path = os.path.join(self.image_directory, subdir)
                os.makedirs(subdir_path, exist_ok=True)

    def _load_default_mappings(self):
        """Load default image mappings"""
        # Default images for each location
        self.image_map = {
            "home_default": os.path.join(self.image_directory, "home", "default.jpg"),
            "home_ac_on_22": os.path.join(self.image_directory, "home", "ac_on_22.jpg"),
            "home_ac_on_24": os.path.join(self.image_directory, "home", "ac_on_24.jpg"),
            "home_ac_off": os.path.join(self.image_directory, "home", "ac_off.jpg"),
            "home_light_on": os.path.join(self.image_directory, "home", "light_on.jpg"),
            "home_light_off": os.path.join(self.image_directory, "home", "light_off.jpg"),
            "store_default": os.path.join(self.image_directory, "store", "default.jpg"),
            "store_robot_present": os.path.join(self.image_directory, "store", "robot_present.jpg"),
            "store_item_ready": os.path.join(self.image_directory, "store", "item_ready.jpg"),
            "in_transit_to_store": os.path.join(self.image_directory, "in_transit", "to_store.jpg"),
            "in_transit_to_home": os.path.join(self.image_directory, "in_transit", "to_home.jpg"),
        }

    def register_custom_image(self, state_key: str, image_path: str):
        """
        Register a custom image for a specific state

        Args:
            state_key: State identifier (e.g., "home_ac_on_22")
            image_path: Path to image file
        """
        if not os.path.exists(image_path):
            print(f"⚠️  Warning: Image not found: {image_path}")
            return

        self.image_map[state_key] = image_path

        if self.verbose:
            print(f"✅ Registered custom image: {state_key} → {image_path}")

    def update_state(self, **kwargs):
        """
        Update robot state

        Args:
            **kwargs: State updates (location, ac_status, light_status, etc.)
        """
        old_state = self.state.copy()

        for key, value in kwargs.items():
            if key in self.state:
                self.state[key] = value

        # Record state change
        self.state_history.append({
            "timestamp": datetime.now().isoformat(),
            "old_state": old_state,
            "new_state": self.state.copy(),
            "changes": kwargs
        })

        if self.verbose:
            print(f"📊 State updated: {kwargs}")

    def get_current_observation_image(self) -> str:
        """
        Get observation image for current state

        Returns:
            str: Path to appropriate observation image
        """
        # Build state key based on current state
        location = self.state["location"]

        # Try to find specific state match
        if location == "home":
            # Priority 0: If no state changes yet (initial observation), use default
            if not self.state_history:
                state_key = "home_default"
                if state_key in self.image_map and os.path.exists(self.image_map[state_key]):
                    return self.image_map[state_key]

            # Priority 1: Try combined state images (AC + light)
            if self.state["ac_status"] == "on" and self.state["ac_temperature"]:
                if self.state["light_status"] == "on":
                    # Both AC and light on
                    state_key = f"home_ac_on_{self.state['ac_temperature']}_light_on"
                    if state_key in self.image_map and os.path.exists(self.image_map[state_key]):
                        return self.image_map[state_key]
                else:
                    # AC on, light off
                    state_key = f"home_ac_on_{self.state['ac_temperature']}_light_off"
                    if state_key in self.image_map and os.path.exists(self.image_map[state_key]):
                        return self.image_map[state_key]

            # Priority 2: Check individual device states based on what changed most recently
            if self.state_history:
                # Get most recent change
                last_change = self.state_history[-1]["changes"]

                # If light was just changed, prioritize light state
                if "light_status" in last_change:
                    if self.state["light_status"] == "on":
                        state_key = "home_light_on"
                        if state_key in self.image_map and os.path.exists(self.image_map[state_key]):
                            return self.image_map[state_key]
                    else:
                        state_key = "home_light_off"
                        if state_key in self.image_map and os.path.exists(self.image_map[state_key]):
                            return self.image_map[state_key]

                # If AC was just changed, prioritize AC state
                if "ac_status" in last_change or "ac_temperature" in last_change:
                    if self.state["ac_status"] == "on" and self.state["ac_temperature"]:
                        state_key = f"home_ac_on_{self.state['ac_temperature']}"
                        if state_key in self.image_map and os.path.exists(self.image_map[state_key]):
                            return self.image_map[state_key]
                    else:
                        state_key = "home_ac_off"
                        if state_key in self.image_map and os.path.exists(self.image_map[state_key]):
                            return self.image_map[state_key]

            # Priority 3: Fall back to individual device states (no recent change info)
            if self.state["ac_status"] == "on" and self.state["ac_temperature"]:
                state_key = f"home_ac_on_{self.state['ac_temperature']}"
                if state_key in self.image_map and os.path.exists(self.image_map[state_key]):
                    return self.image_map[state_key]

            if self.state["light_status"] == "on":
                state_key = "home_light_on"
                if state_key in self.image_map and os.path.exists(self.image_map[state_key]):
                    return self.image_map[state_key]
            elif self.state["light_status"] == "off":
                state_key = "home_light_off"
                if state_key in self.image_map and os.path.exists(self.image_map[state_key]):
                    return self.image_map[state_key]

            # Fallback to default home
            return self._get_fallback_image("home")

        elif location == "store":
            # Use default if no state changes yet
            if not self.state_history:
                state_key = "store_default"
                if state_key in self.image_map and os.path.exists(self.image_map[state_key]):
                    return self.image_map[state_key]

            state_key = "store_robot_present"
            if state_key in self.image_map and os.path.exists(self.image_map[state_key]):
                return self.image_map[state_key]

            return self._get_fallback_image("store")

        elif location == "in_transit":
            # Use default if no state changes yet
            if not self.state_history:
                return self._get_fallback_image("in_transit")

            # Determine direction
            # (This would need more context - simplified here)
            return self._get_fallback_image("in_transit")

        # Default fallback
        return self._get_fallback_image("home")

    def _get_fallback_image(self, location: str) -> str:
        """
        Get fallback image for location

        Args:
            location: Location name

        Returns:
            str: Path to fallback image (generates placeholder if needed)
        """
        state_key = f"{location}_default"

        if state_key in self.image_map and os.path.exists(self.image_map[state_key]):
            return self.image_map[state_key]

        # Generate placeholder image path
        placeholder_path = os.path.join(self.image_directory, location, "placeholder.jpg")

        if not os.path.exists(placeholder_path):
            # Create a text-based placeholder
            self._create_placeholder_image(placeholder_path, location)

        return placeholder_path

    def _create_placeholder_image(self, path: str, location: str):
        """
        Create a placeholder image (simple text file for now)

        In real implementation, would generate actual image with PIL
        """
        # For simulation, just create a text description file
        # User will replace with actual images
        placeholder_text = f"""
PLACEHOLDER IMAGE: {location}

This is a placeholder for the observation image.
Please replace this with an actual image file:

Expected path: {path}

State context:
{json.dumps(self.state, indent=2)}

To use real images:
1. Provide a JPG/PNG image of the {location} scene
2. Name it appropriately based on state
3. Place it in the correct directory

For now, the VLM planner will receive a text description instead.
"""
        text_path = path.replace(".jpg", ".txt")
        with open(text_path, "w") as f:
            f.write(placeholder_text)

        if self.verbose:
            print(f"📝 Created placeholder description: {text_path}")
            print(f"   ⚠️  Replace with actual image: {path}")

    def get_observation_after_action(self, action_type: str, action_name: str, parameters: Dict) -> str:
        """
        Get observation image after executing an action

        Args:
            action_type: Type of action (talk, tool, act, sense)
            action_name: Specific action name
            parameters: Action parameters

        Returns:
            str: Path to observation image reflecting action result
        """
        # Update state based on action
        if action_name == "control_air_conditioner":
            if parameters.get("action") == "turn_on":
                self.update_state(
                    ac_status="on",
                    ac_temperature=parameters.get("temperature", 24)
                )
            else:
                self.update_state(ac_status="off", ac_temperature=None)

        elif action_name == "control_light":
            if parameters.get("action") == "turn_on":
                self.update_state(light_status="on")
            else:
                self.update_state(light_status="off")

        elif action_name == "navigate_to":
            # Handle unified navigation action
            target_location = parameters.get("target_location", "")
            with_item = parameters.get("with_item", "none")

            # Map Room 01/Room 02 to home/store
            location_map = {
                "Room 01": "home",
                "Room 02": "store"
            }

            internal_location = location_map.get(target_location, "home")

            # Update location and item carrying state
            update_dict = {"location": internal_location}

            if with_item and with_item != "none":
                update_dict["carrying_item"] = with_item
            else:
                update_dict["carrying_item"] = None

            self.update_state(**update_dict)

        elif action_name == "pick":
            self.update_state(carrying_item=parameters.get("object_description"))

        elif action_name == "place":
            self.update_state(carrying_item=None)

        # Legacy support for old action names
        elif action_name == "navigate_to_store":
            self.update_state(location="store")

        elif action_name == "return_home_with_item":
            self.update_state(
                location="home",
                carrying_item=parameters.get("item")
            )

        elif action_name == "wait_for":
            # No state change for waiting
            pass

        # Return appropriate observation image
        return self.get_current_observation_image()

    def get_state_summary(self) -> Dict:
        """Get current state summary"""
        return {
            "current_state": self.state.copy(),
            "state_changes": len(self.state_history),
            "current_image": self.get_current_observation_image()
        }

    def reset_state(self):
        """Reset to initial state"""
        self.state = {
            "location": "home",
            "ac_status": "off",
            "ac_temperature": None,
            "light_status": "off",
            "carrying_item": None,
            "human_present": True
        }
        self.state_history = []

        if self.verbose:
            print("🔄 State reset to initial configuration")


# Example usage
if __name__ == "__main__":
    print("🖼️  Simulation Image Manager - Test Mode\n")

    # Initialize manager
    manager = SimulationImageManager(verbose=True)

    print("\n" + "="*70)
    print("Initial State")
    print("="*70)
    print(json.dumps(manager.get_state_summary(), indent=2))

    print("\n" + "="*70)
    print("Simulating Actions")
    print("="*70)

    # Simulate turning on AC
    print("\n1. Turn on AC at 22°C")
    image = manager.get_observation_after_action(
        "tool",
        "control_air_conditioner",
        {"action": "turn_on", "temperature": 22}
    )
    print(f"   Observation image: {image}")

    # Simulate navigation to store
    print("\n2. Navigate to store (Room 02)")
    image = manager.get_observation_after_action(
        "act",
        "navigate_to",
        {"target_location": "Room 02", "with_item": "none"}
    )
    print(f"   Observation image: {image}")
    print(f"   Current location: {manager.state['location']}")

    # Simulate returning home with item
    print("\n3. Return home (Room 01) with water")
    image = manager.get_observation_after_action(
        "act",
        "navigate_to",
        {"target_location": "Room 01", "with_item": "water"}
    )
    print(f"   Observation image: {image}")
    print(f"   Current location: {manager.state['location']}")
    print(f"   Carrying item: {manager.state['carrying_item']}")

    print("\n" + "="*70)
    print("Final State")
    print("="*70)
    print(json.dumps(manager.get_state_summary(), indent=2))

    print("\n" + "="*70)
    print("To use with real images:")
    print("  1. Place images in simulation_images/ subdirectories")
    print("  2. Use manager.register_custom_image(state_key, image_path)")
    print("  3. Images will be automatically selected based on state")
    print("="*70)
