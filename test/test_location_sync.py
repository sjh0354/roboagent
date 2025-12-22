
import unittest
from unittest.mock import MagicMock, patch
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from planner.humanoid_planner_vlm import AutonomousVLMPlanner

class TestLocationSync(unittest.TestCase):
    @patch('planner.humanoid_planner_vlm.QwenVLMClient')
    @patch('planner.humanoid_planner_vlm.OpenAI')
    @patch('planner.humanoid_planner_vlm.VisionEnabledExecutor')
    def test_location_update_from_observation(self, MockExecutor, MockOpenAI, MockVLMClient):
        # Setup mocks
        mock_executor_instance = MockExecutor.return_value
        
        # Initialize planner (with dummy key to bypass check)
        planner = AutonomousVLMPlanner(api_key="dummy_key", verbose=False)
        
        # 1. Test Initial State
        self.assertEqual(planner.current_location, "Room 01")
        
        # 2. Mock observation returning "store" location
        mock_executor_instance.get_current_observation.return_value = {
            "image_path": "dummy_path.jpg",
            "state": {
                "location": "store",
                "other_info": "foo"
            }
        }
        
        # Mock VLM response to avoid network calls
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = '{"next_step": null}'
        planner.client.chat.completions.create.return_value = mock_completion
        
        # Trigger the loop (which calls get_current_observation and updates state)
        # We'll mock plan_next_step_with_image to stop the loop immediately
        with patch.object(planner, 'plan_next_step_with_image') as mock_plan:
            mock_plan.return_value = {"next_step": None} # Stop loop
            
            planner._run_autonomous_loop()
            
            # Verify location was updated to Room 02 (mapped from "store")
            self.assertEqual(planner.current_location, "Room 02")
            print("✅ Location successfully updated from 'store' to 'Room 02'")

        # 3. Test update back to "home"
        mock_executor_instance.get_current_observation.return_value = {
            "image_path": "dummy_path.jpg",
            "state": {
                "location": "home"
            }
        }
        
        with patch.object(planner, 'plan_next_step_with_image') as mock_plan:
            mock_plan.return_value = {"next_step": None}
            planner.is_task_complete = False # Reset flag
            
            planner._run_autonomous_loop()
            
            # Verify location was updated to Room 01 (mapped from "home")
            self.assertEqual(planner.current_location, "Room 01")
            print("✅ Location successfully updated from 'home' to 'Room 01'")

if __name__ == '__main__':
    unittest.main()
