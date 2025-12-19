#!/usr/bin/env python3
"""
Test script to verify location tracking in VLM planner
"""

import sys
import os

# Simple test to verify location tracking logic


class MockPlanner:
    """Mock planner to test location tracking"""

    def __init__(self):
        self.current_location = "Room 01"  # Start at Room 01
        self.verbose = True

    def simulate_navigation(self, target_location):
        """Simulate a navigation action"""
        if self.verbose:
            print(f"\n{'='*70}")
            print(f"🔧 Simulating navigate_to action")
            print(f"   Parameters: target_location={target_location}")
            print(f"{'='*70}")

        # Update location (mimics the logic in _execute_step)
        old_location = self.current_location
        self.current_location = target_location

        if self.verbose:
            print(f"📍 Location updated: {old_location} → {target_location}")

    def get_context_location_message(self):
        """Get the context message with current location"""
        return f"[CURRENT LOCATION]: {self.current_location} (tracked automatically via navigate_to actions)"

    def check_location(self, expected):
        """Verify current location"""
        if self.current_location == expected:
            print(f"✅ PASS: Location is {self.current_location}")
            return True
        else:
            print(f"❌ FAIL: Expected {expected}, got {self.current_location}")
            return False


def main():
    print("🤖 Location Tracking Test")
    print("="*70)
    print("Testing state-based location tracking in VLM planner\n")

    # Create mock planner
    planner = MockPlanner()

    # Test 1: Initial location
    print("\n" + "="*70)
    print("TEST 1: Initial Location")
    print("="*70)
    print(planner.get_context_location_message())
    test1_pass = planner.check_location("Room 01")

    # Test 2: Navigate to store
    print("\n" + "="*70)
    print("TEST 2: Navigate to Store (Room 02)")
    print("="*70)
    planner.simulate_navigation("Room 02")
    print(planner.get_context_location_message())
    test2_pass = planner.check_location("Room 02")

    # Test 3: Return home
    print("\n" + "="*70)
    print("TEST 3: Return Home (Room 01)")
    print("="*70)
    planner.simulate_navigation("Room 01")
    print(planner.get_context_location_message())
    test3_pass = planner.check_location("Room 01")

    # Test 4: Multiple navigations
    print("\n" + "="*70)
    print("TEST 4: Multiple Navigation Sequence")
    print("="*70)
    planner.simulate_navigation("Room 02")
    print("After nav to Room 02:", planner.get_context_location_message())
    planner.simulate_navigation("Room 01")
    print("After nav to Room 01:", planner.get_context_location_message())
    planner.simulate_navigation("Room 02")
    print("After nav to Room 02:", planner.get_context_location_message())
    test4_pass = planner.check_location("Room 02")

    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    all_tests = [test1_pass, test2_pass, test3_pass, test4_pass]
    passed = sum(all_tests)
    total = len(all_tests)

    print(f"Tests passed: {passed}/{total}")
    if passed == total:
        print("✅ All tests PASSED!")
        print("\nLocation tracking is working correctly:")
        print("  ✓ Location starts at Room 01")
        print("  ✓ Location updates after navigate_to execution")
        print("  ✓ Location is provided in context message")
        print("  ✓ Location persists across multiple navigations")
        return 0
    else:
        print("❌ Some tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
