# Hardware Integration Guide for Unitree-G1 Humanoid Robot

This guide explains how to integrate real hardware APIs into the Interactive Humanoid Planner system.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface / CLI                     │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│          InteractiveHumanoidPlanner (Planner)               │
│  - Plans next step based on LLM reasoning                   │
│  - Manages conversation history                             │
│  - Coordinates task execution flow                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│            HumanoidExecutor (Executor)                      │
│  - Executes planned actions on real/simulated robot        │
│  - Handles hardware communication                           │
│  - Returns structured feedback                              │
└─────────────────────┬───────────────────────────────────────┘
                      │
          ┌───────────┴───────────┬──────────────┐
          ▼                       ▼              ▼
┌──────────────────┐    ┌─────────────┐   ┌─────────────┐
│ Unitree-G1 API   │    │ Vision/VLM  │   │ Smart Home  │
│ (Navigation/Act) │    │   System    │   │   APIs      │
└──────────────────┘    └─────────────┘   └─────────────┘
```

## Integration Steps

### 1. Navigation & Locomotion (ACT Actions)

**File**: `humanoid_executor.py` → `_execute_act()` methods

**Actions to implement**:
- `navigate_to_store()`: Move robot to store location
- `return_home_with_item()`: Navigate back while carrying object
- `wait_for_item()`: Idle state management

**Required APIs**:
```python
# Example Unitree-G1 SDK integration
from unitree_sdk import UnitreeG1Controller

def _initialize_hardware(self):
    self.robot_controller = UnitreeG1Controller(
        ip_address="192.168.1.100",  # Your robot IP
        port=8080
    )

def _navigate_to_store(self):
    # Real implementation
    target_position = {"x": 5.0, "y": 2.0, "theta": 0.0}
    result = self.robot_controller.navigate_to(target_position)

    if result.success:
        return ExecutionResult(
            success=True,
            feedback=f"Arrived at store location",
            data={"position": result.final_position}
        )
    else:
        return ExecutionResult(
            success=False,
            feedback="Navigation failed",
            error=result.error_message
        )
```

**APIs you need from Unitree**:
- `navigate_to(x, y, theta)`: Path planning and execution
- `get_current_position()`: Localization
- `check_collision()`: Safety monitoring
- `set_velocity(vx, vy, w)`: Low-level control
- `get_battery_level()`: Status monitoring

### 2. Vision & Perception (SENSE Actions)

**File**: `humanoid_executor.py` → `_execute_sense()` methods

**Actions to implement**:
- `get_observation()`: Capture and analyze scene

**Required APIs**:
```python
from unitree_sdk import Camera
from your_vlm_service import VisionLanguageModel

def _initialize_hardware(self):
    self.camera = Camera(device_id=0)
    self.vision_system = VisionLanguageModel(
        api_key="your_vlm_api_key",
        model="gpt-4-vision" or "qwen-vl-max"
    )

def _get_observation(self):
    # Capture image
    image = self.camera.capture()

    # Analyze with VLM
    prompt = "Describe this scene in detail. Include: people, objects, room state, and device status."
    observation = self.vision_system.analyze(image, prompt)

    return ExecutionResult(
        success=True,
        feedback=f"Observation: {observation}",
        data={"image_path": image.save_path, "observation": observation}
    )
```

**VLM Options**:
- **OpenAI GPT-4V**: Best quality, requires internet
- **Qwen-VL-Max**: Alibaba's VLM, good for Chinese context
- **Local VLM**: LLaVA, CogVLM (runs on edge GPU)

### 3. Communication (TALK Actions)

**File**: `humanoid_executor.py` → `_execute_talk()` methods

**Actions to implement**:
- `talk_with_human()`: Verbal/visual communication
- `request_item_from_store()`: Inter-robot communication

**Required APIs**:
```python
from unitree_sdk import SpeechSynthesizer
import requests  # For inter-robot communication

def _initialize_hardware(self):
    self.tts = SpeechSynthesizer(language="en")
    self.store_robot_url = "http://192.168.1.200:5000"

def _talk_with_human(self, message):
    # Text-to-speech
    self.tts.speak(message)

    # Optional: Display on robot screen
    # self.robot_controller.display_message(message)

    return ExecutionResult(
        success=True,
        feedback=f"Message delivered: '{message}'"
    )

def _request_item_from_store(self, item):
    # REST API call to store robot
    response = requests.post(
        f"{self.store_robot_url}/api/request_item",
        json={"item": item, "requester": "humanoid_robot"}
    )

    if response.status_code == 200:
        data = response.json()
        return ExecutionResult(
            success=True,
            feedback=f"Store robot acknowledged: {data['message']}",
            data=data
        )
    else:
        return ExecutionResult(
            success=False,
            feedback="Failed to contact store robot",
            error=f"HTTP {response.status_code}"
        )
```

### 4. Device Control (TOOL Actions)

**File**: `humanoid_executor.py` → `_execute_tool()` methods

**Actions to implement**:
- `control_air_conditioner()`: HVAC control
- `control_light()`: Lighting control
- `web_search()`: Information retrieval

**Required APIs**:
```python
from homeassistant_api import Client as HomeAssistant
import requests

def _initialize_hardware(self):
    self.smart_home_controller = HomeAssistant(
        url="http://homeassistant.local:8123",
        token="your_ha_token"
    )

def _control_air_conditioner(self, action, temperature):
    if action == "turn_on":
        self.smart_home_controller.services.climate.set_temperature(
            entity_id="climate.living_room_ac",
            temperature=temperature
        )
    elif action == "turn_off":
        self.smart_home_controller.services.climate.turn_off(
            entity_id="climate.living_room_ac"
        )

    return ExecutionResult(
        success=True,
        feedback=f"AC {action} at {temperature}°C"
    )
```

**Smart Home Integration Options**:
- **Home Assistant**: Open-source, supports many devices
- **Direct APIs**: AC brand-specific APIs (Daikin, Mitsubishi, etc.)
- **MQTT**: IoT messaging protocol
- **Zigbee/Z-Wave**: Direct wireless device control

## Safety Considerations

### Emergency Stop System
```python
class HumanoidExecutor:
    def __init__(self, ...):
        self.emergency_stop = False

    def execute_action(self, action_type, action_name, parameters):
        # Check emergency stop before execution
        if self.emergency_stop:
            return ExecutionResult(
                success=False,
                feedback="Emergency stop activated",
                error="System halted for safety"
            )

        # Check human proximity
        if self._human_too_close():
            return ExecutionResult(
                success=False,
                feedback="Human detected in safety zone",
                error="Waiting for clear path"
            )

        # Proceed with execution
        ...
```

### Collision Avoidance
```python
def _navigate_to_store(self):
    # Enable collision detection
    self.robot_controller.enable_obstacle_avoidance(True)

    # Set safety parameters
    self.robot_controller.set_safety_distance(0.5)  # 50cm margin

    # Navigate with monitoring
    result = self.robot_controller.navigate_to(target,
                                                 max_speed=0.3)  # Slow for safety
```

### Error Recovery
```python
def execute_action(self, action_type, action_name, parameters):
    max_retries = 3

    for attempt in range(max_retries):
        try:
            result = self._execute_action_internal(action_type, action_name, parameters)

            if result.success:
                return result

            # Retry logic for recoverable errors
            if "timeout" in result.error.lower():
                time.sleep(2)
                continue

        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(1)

    return ExecutionResult(success=False, feedback="Max retries exceeded")
```

## Testing Strategy

### 1. Unit Testing (Simulation)
```bash
# Test executor in simulation mode
python humanoid_executor.py

# Run example tasks
python humanoid_usage_example.py
```

### 2. Integration Testing (Mock Hardware)
Create mock hardware classes that simulate real APIs:
```python
class MockUnitreeG1Controller:
    def navigate_to(self, position):
        time.sleep(1)  # Simulate navigation time
        return MockNavigationResult(success=True, final_position=position)
```

### 3. Hardware-in-Loop Testing
Test with real hardware in controlled environment:
- Small enclosed area
- Soft obstacles
- Emergency stop within reach
- Human supervisor present

### 4. Full Deployment
Only after thorough testing:
```python
planner = InteractiveHumanoidPlanner(
    simulation_mode=False,  # REAL HARDWARE
    verbose=True
)
```

## API Checklist for Unitree-G1

- [ ] **Navigation APIs**
  - [ ] `navigate_to(x, y, theta)`
  - [ ] `get_current_position()`
  - [ ] `stop_navigation()`
  - [ ] `set_safety_parameters()`

- [ ] **Manipulation APIs** (if needed)
  - [ ] `grasp_object(object_id)`
  - [ ] `release_object()`
  - [ ] `get_gripper_state()`

- [ ] **Perception APIs**
  - [ ] `capture_image(camera_id)`
  - [ ] `get_depth_map()`
  - [ ] VLM integration (cloud or local)

- [ ] **Communication APIs**
  - [ ] Text-to-speech library
  - [ ] Display interface (if robot has screen)
  - [ ] Network communication for inter-robot

- [ ] **Smart Home APIs**
  - [ ] AC control interface
  - [ ] Light control interface
  - [ ] Device status query

- [ ] **Safety Systems**
  - [ ] Emergency stop mechanism
  - [ ] Collision detection
  - [ ] Human detection
  - [ ] Battery monitoring
  - [ ] Error logging

## Next Steps

1. **Contact Unitree**: Get official SDK documentation and API references
2. **Set up development environment**: Install Unitree SDK, test connectivity
3. **Implement one action category at a time**: Start with perception (easiest to test)
4. **Test incrementally**: Simulation → Mock → Real hardware
5. **Add safety layers**: Never skip safety features
6. **Document your APIs**: Keep track of what works and what doesn't

## Support Resources

- Unitree Official Docs: [unitree.com/support](https://www.unitree.com/support)
- ROS Integration: If using ROS, check for `unitree_ros` packages
- Community: Unitree forums, GitHub issues, Discord servers
- VLM Options: OpenAI, Qwen-VL, LLaVA

---

**Remember**: Safety first! Always test in simulation before deploying on real hardware.
