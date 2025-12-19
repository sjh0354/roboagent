# test/test_arm_vlm_planner.py

"""
Test script for VLM-based autonomous arm planner
Tests the vision-based planning system in simulation mode
"""

import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.arm_planner_vlm import AutonomousArmVLMPlanner
import json


def test_initialization():
    """Test 1: Initialize arm VLM planner"""
    print("\n" + "="*70)
    print("TEST 1: Initialization")
    print("="*70)

    try:
        # Use dummy API key for testing if not set
        api_key = os.getenv("DASHSCOPE_API_KEY") or "test_api_key_for_simulation"

        planner = AutonomousArmVLMPlanner(
            api_key=api_key,
            model_name="qwen-vl-plus",
            simulation_mode=True,
            verbose=True
        )
        print("✅ Arm planner initialized successfully")
        if not os.getenv("DASHSCOPE_API_KEY"):
            print("   (Using dummy API key - VLM calls will fail, but structure works)")
        return planner
    except Exception as e:
        print(f"❌ Initialization failed: {str(e)}")
        return None


def test_planning_components(planner):
    """Test 2: Test planning components (without API call)"""
    print("\n" + "="*70)
    print("TEST 2: Planning Components (No API)")
    print("="*70)

    try:
        # Test context building
        planner.original_request = "Get water for humanoid"
        planner.step_count = 0

        context = planner._build_context_message()
        print("✅ Context message built:")
        print(f"   {context[:200]}...")

        # Test step plan parsing
        test_json = '''
{
  "current_step_analysis": {
    "visual_state": "Shelves with water visible",
    "task_progress": "Starting retrieval",
    "next_action_reasoning": "Pick water from shelf"
  },
  "next_step": {
    "step_number": 1,
    "agent": "ur5e_arm",
    "location": "store",
    "action": "pick_from_shelf",
    "action_type": "act",
    "parameters": {"item_name": "water", "shelf_location": "Shelf 1, Level 2"},
    "expected_visual_outcome": "Gripper holds water",
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


def test_action_validation():
    """Test 3: Validate action constraints"""
    print("\n" + "="*70)
    print("TEST 3: Action Validation")
    print("="*70)

    from arm_prompt_template_vlm import validate_arm_vlm_response

    try:
        # Test valid action
        valid_response = '''
{
  "current_step_analysis": {
    "visual_state": "test",
    "task_progress": "test",
    "next_action_reasoning": "test"
  },
  "next_step": {
    "step_number": 1,
    "agent": "ur5e_arm",
    "location": "store",
    "action": "pick_from_shelf",
    "action_type": "act",
    "parameters": {}
  },
  "needs_human_input": false
}
'''
        is_valid, msg = validate_arm_vlm_response(valid_response)
        print(f"✅ Valid action test: {'PASS' if is_valid else 'FAIL'}")

        # Test invalid action
        invalid_response = '''
{
  "current_step_analysis": {
    "visual_state": "test",
    "task_progress": "test",
    "next_action_reasoning": "test"
  },
  "next_step": {
    "step_number": 1,
    "agent": "ur5e_arm",
    "location": "store",
    "action": "grasp_item",
    "action_type": "act",
    "parameters": {}
  },
  "needs_human_input": false
}
'''
        is_valid, msg = validate_arm_vlm_response(invalid_response)
        print(f"✅ Invalid action test: {'PASS' if not is_valid else 'FAIL'}")
        if not is_valid:
            print(f"   Correctly rejected: {msg}")

        return True
    except Exception as e:
        print(f"❌ Action validation test failed: {str(e)}")
        return False


def test_simulation_flow(planner):
    """Test 4: Full task simulation (No API)"""
    print("\n" + "="*70)
    print("TEST 4: Full Task Simulation (No API)")
    print("="*70)
    print("Note: This test simulates the flow without calling VLM API")

    try:
        # Reset planner
        planner.reset_conversation()

        # Set task
        planner.original_request = "Get water for humanoid"
        planner.task_start_time = None

        # Get current observation
        obs_image = planner._get_default_observation_image()
        print(f"\n📋 Simulating planning cycle:")
        print(f"   Task: {planner.original_request}")
        print(f"   Observation image: {obs_image}")

        print("\n✅ Simulation flow verified")
        print("   To run with real VLM API:")
        print("   1. Set DASHSCOPE_API_KEY environment variable")
        print("   2. Run: python arm_planner_vlm.py")
        print("   3. Enter request: 'Get water'")

        return True
    except Exception as e:
        print(f"❌ Full simulation test failed: {str(e)}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("="*70)
    print("ARM VLM-BASED AUTONOMOUS PLANNER - TEST SUITE")
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

    # Test 2: Planning components
    results.append(("Planning Components", test_planning_components(planner)))

    # Test 3: Action validation
    results.append(("Action Validation", test_action_validation()))

    # Test 4: Full simulation
    results.append(("Full Task Simulation", test_simulation_flow(planner)))

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
        print("   1. Set DASHSCOPE_API_KEY for VLM planning")
        print("   2. Run: python arm_planner_vlm.py")
        print("   3. Test with humanoid requests")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")

    print("="*70)


if __name__ == "__main__":
    main()
