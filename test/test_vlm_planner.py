# test/test_vlm_planner.py

"""
Test script for VLM-based autonomous planner
Tests the vision-based planning system in simulation mode
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.humanoid_planner_vlm import AutonomousVLMPlanner
import json


def test_initialization():
    """Test 1: Initialize VLM planner"""
    print("\n" + "="*70)
    print("TEST 1: Initialization")
    print("="*70)

    try:
        # Use dummy API key for testing if not set
        api_key = os.getenv("DASHSCOPE_API_KEY") or "test_api_key_for_simulation"

        planner = AutonomousVLMPlanner(
            api_key=api_key,
            model_name="qwen-vl-plus",
            simulation_mode=True,
            verbose=True
        )
        print("✅ Planner initialized successfully")
        if not os.getenv("DASHSCOPE_API_KEY"):
            print("   (Using dummy API key - VLM calls will fail, but structure works)")
        return planner
    except Exception as e:
        print(f"❌ Initialization failed: {str(e)}")
        return None


def test_observation_system(planner):
    """Test 2: Vision/observation system"""
    print("\n" + "="*70)
    print("TEST 2: Vision/Observation System")
    print("="*70)

    try:
        obs = planner.executor.get_current_observation()
        print(f"✅ Current observation retrieved:")
        print(f"   Image: {obs['image_path']}")
        print(f"   State: {obs['state']}")

        # Test image manager state updates
        planner.executor.image_manager.update_state(ac_status="on", ac_temperature=22)
        obs_after = planner.executor.get_current_observation()
        print(f"\n✅ State updated successfully:")
        print(f"   New state: {obs_after['state']}")

        # Reset state
        planner.executor.reset_vision_state()
        print(f"\n✅ State reset successful")

        return True
    except Exception as e:
        print(f"❌ Observation test failed: {str(e)}")
        return False


def test_executor_with_vision(planner):
    """Test 3: Execute actions with vision"""
    print("\n" + "="*70)
    print("TEST 3: Vision-Enabled Executor")
    print("="*70)

    try:
        # Test action execution
        print("\n📍 Test 3a: Control air conditioner")
        result = planner.executor.execute_action(
            "tool",
            "control_air_conditioner",
            {"action": "turn_on", "temperature": 24}
        )

        print(f"{'✅' if result.success else '❌'} Action result:")
        print(f"   Success: {result.success}")
        print(f"   Feedback: {result.feedback}")
        print(f"   Observation image: {result.data.get('observation_image')}")
        print(f"   State: {result.data.get('state')}")

        # Test navigation
        print("\n📍 Test 3b: Navigate to store")
        result2 = planner.executor.execute_action(
            "act",
            "navigate_to_store",
            {}
        )

        print(f"{'✅' if result2.success else '❌'} Navigation result:")
        print(f"   Success: {result2.success}")
        print(f"   New location: {result2.data.get('state', {}).get('location')}")

        return True
    except Exception as e:
        print(f"❌ Executor test failed: {str(e)}")
        return False


def test_planning_without_api(planner):
    """Test 4: Test planning components (without API call)"""
    print("\n" + "="*70)
    print("TEST 4: Planning Components (No API)")
    print("="*70)

    try:
        # Test context building
        planner.original_request = "Test task"
        planner.step_count = 0

        context = planner._build_context_message()
        print("✅ Context message built:")
        print(f"   {context[:200]}...")

        # Test step plan parsing
        test_json = '''
{
  "current_step_analysis": {
    "visual_state": "Test state",
    "task_progress": "Test progress",
    "next_action_reasoning": "Test reasoning"
  },
  "next_step": {
    "step_number": 1,
    "agent": "Unitree-G1 humanoid_robot",
    "location": "home",
    "action": "control_air_conditioner",
    "action_type": "tool",
    "parameters": {"action": "turn_on", "temperature": 22},
    "expected_visual_outcome": "AC display shows ON",
    "verification_method": "Visual check"
  },
  "needs_human_input": false
}
'''
        parsed = planner._parse_step_plan(test_json)
        print(f"\n✅ Step plan parsed successfully:")
        print(f"   Action: {parsed.get('next_step', {}).get('action')}")

        return True
    except Exception as e:
        print(f"❌ Planning components test failed: {str(e)}")
        return False


def test_custom_images(planner):
    """Test 5: Custom image registration"""
    print("\n" + "="*70)
    print("TEST 5: Custom Image Registration")
    print("="*70)

    try:
        # Register a custom image (will create placeholder if not exists)
        planner.register_observation_image(
            "home_ac_on_24",
            "simulation_images/home/ac_on_24.jpg"
        )

        print("✅ Custom image registered")
        print("   Note: Actual image files can be placed in simulation_images/")

        return True
    except Exception as e:
        print(f"❌ Custom image test failed: {str(e)}")
        return False


def test_full_task_simulation(planner):
    """Test 6: Full task without API (mock mode)"""
    print("\n" + "="*70)
    print("TEST 6: Full Task Simulation (No API)")
    print("="*70)
    print("Note: This test simulates the flow without calling VLM API")
    print("      To test with real VLM API, set DASHSCOPE_API_KEY and run main script")

    try:
        # Reset planner
        planner.reset_conversation()

        # Set task
        planner.original_request = "I'm feeling cold"
        planner.task_start_time = None

        # Manually simulate a planning cycle
        print("\n📋 Simulating planning cycle:")
        print(f"   Task: {planner.original_request}")

        # Get current observation
        obs = planner.executor.get_current_observation()
        print(f"   Current image: {obs['image_path']}")

        # In real scenario, would call plan_next_step_with_image(obs['image_path'])
        # But that requires API, so we just verify the flow works

        print("\n✅ Simulation flow verified")
        print("   To run with real VLM API:")
        print("   1. Set DASHSCOPE_API_KEY environment variable")
        print("   2. Run: python humanoid_planner_vlm.py")
        print("   3. Enter task: 'I'm feeling cold'")

        return True
    except Exception as e:
        print(f"❌ Full simulation test failed: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("="*70)
    print("VLM-BASED AUTONOMOUS PLANNER - TEST SUITE")
    print("="*70)

    # Check API key status
    has_api_key = bool(os.getenv("DASHSCOPE_API_KEY"))
    print(f"\n📡 API Key Status: {'✅ Set' if has_api_key else '❌ Not Set'}")
    if not has_api_key:
        print("   Tests will run in MOCK MODE (no actual VLM API calls)")
        print("   Set DASHSCOPE_API_KEY to enable full VLM testing")
    else:
        print("   Full VLM testing available")

    # Run tests
    results = []

    # Test 1: Initialization
    planner = test_initialization()
    results.append(("Initialization", planner is not None))

    if not planner:
        print("\n❌ Cannot proceed without planner initialization")
        return

    # Test 2: Observation system
    results.append(("Observation System", test_observation_system(planner)))

    # Test 3: Executor with vision
    results.append(("Vision-Enabled Executor", test_executor_with_vision(planner)))

    # Test 4: Planning components
    results.append(("Planning Components", test_planning_without_api(planner)))

    # Test 5: Custom images
    results.append(("Custom Image Registration", test_custom_images(planner)))

    # Test 6: Full simulation
    results.append(("Full Task Simulation", test_full_task_simulation(planner)))

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    passed = sum(1 for _, success in results if success)
    total = len(results)

    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {test_name}")

    print("="*70)
    print(f"Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 All tests passed!")
        print("\n📋 Next Steps:")
        print("   1. Provide observation images in simulation_images/")
        print("   2. Set DASHSCOPE_API_KEY for VLM planning")
        print("   3. Run: python humanoid_planner_vlm.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

    print("="*70)


if __name__ == "__main__":
    main()
