# dabai_camera_manager.py

"""
DaBai Camera Manager
Handles image capture from Orbbec DaBai Max Pro (via OpenCV/V4L2)
"""

import os
import time
from datetime import datetime
import cv2
import numpy as np

class DaBaiCameraManager:
    """
    Manages connection and capture from Orbbec DaBai Max Pro camera
    """

    def __init__(self, device_id=6, width=1280, height=960, verbose=True):
        """
        Initialize Camera
        
        Args:
            device_id: Video device ID (e.g., 6 for /dev/video6)
            width: Image width
            height: Image height
            verbose: Print status messages
        """
        self.verbose = verbose
        self.device_id = device_id
        self.width = width
        self.height = height
        self.cap = None
        
        # Create output directory (relative to interactive_planner root)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(base_dir, "captured_images")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self._start_camera()

    def _start_camera(self):
        """Start the camera capture"""
        if self.verbose:
            print(f"📷 Opening DaBai Camera on /dev/video{self.device_id}...")
        
        # Prefer V4L2 backend on Linux
        self.cap = cv2.VideoCapture(self.device_id, cv2.CAP_V4L2)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video device /dev/video{self.device_id}")

        # Set properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        # Verify settings
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        if self.verbose:
            print(f"✅ Camera initialized. Resolution: {int(actual_width)}x{int(actual_height)}")
            print("📷 Warming up (2s)...")
        
        # Warmup: Read and discard frames to allow auto-exposure/white-balance to settle
        start_time = time.time()
        while time.time() - start_time < 2.0:
            ret, _ = self.cap.read()
            if not ret:
                time.sleep(0.1)

    def capture_image(self, filename=None, quiet=False):
        """
        Capture a single color frame and save to disk
        
        Args:
            filename: Optional filename (default: timestamped)
            quiet: If True, suppress successful capture logging
            
        Returns:
            str: Path to saved image file
        """
        if not self.cap or not self.cap.isOpened():
            if self.verbose:
                print("⚠️  Camera not open, attempting to restart...")
            self._start_camera()

        max_retries = 3
        for attempt in range(max_retries):
            # Read a frame
            ret, frame = self.cap.read()
            
            if not ret or frame is None:
                if self.verbose:
                    print(f"⚠️  Failed to grab frame (attempt {attempt+1}/{max_retries}).")
                time.sleep(0.2)
                continue

            # Generate filename if not provided
            if filename is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                filename = f"observation_dabai_{timestamp}.jpg"
            
            if not filename.endswith(('.jpg', '.png', '.jpeg')):
                filename += ".jpg"

            save_path = os.path.join(self.output_dir, filename)

            # Save image
            try:
                cv2.imwrite(save_path, frame)
                if self.verbose and not quiet:
                    print(f"📸 Image captured: {save_path}")
                return save_path
            except Exception as e:
                if self.verbose:
                    print(f"❌ Failed to save image: {e}")
                return None
        
        if self.verbose:
            print("❌ Failed to capture image after multiple attempts")
        return None

    def close(self):
        """Release camera resources"""
        if self.cap and self.cap.isOpened():
            self.cap.release()
            if self.verbose:
                print("📷 DaBai camera released")

# Example usage
if __name__ == "__main__":
    try:
        # Defaulting to /dev/video6 as identified
        camera = DaBaiCameraManager(device_id=6)
        img_path = camera.capture_image()
        print(f"Test capture saved to: {img_path}")
        camera.close()
    except Exception as e:
        print(f"Test failed: {str(e)}")
