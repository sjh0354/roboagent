# realsense_manager.py

"""
RealSense Camera Manager
Handles image capture from Intel RealSense D435 for real-world robot vision
"""

import os
import time
from datetime import datetime
import numpy as np
import cv2

try:
    import pyrealsense2 as rs
except ImportError:
    rs = None

class RealSenseCameraManager:
    """
    Manages connection and capture from RealSense D435 camera
    """

    def __init__(self, serial_no="346222072679", width=640, height=480, fps=30, verbose=True):
        """
        Initialize RealSense pipeline
        
        Args:
            serial_no: Serial number of the camera to use (optional)
            width: Image width
            height: Image height
            fps: Frames per second
            verbose: Print status messages
        """
        self.verbose = verbose
        self.serial_no = serial_no
        self.output_dir = "captured_images"
        
        if rs is None:
            raise ImportError("pyrealsense2 not installed. Please install with: pip install pyrealsense2")

        self.pipeline = rs.pipeline()
        self.config = rs.config()

        if self.serial_no:
            if self.verbose:
                print(f"📷 Configuring RealSense with serial number: {self.serial_no}")
            self.config.enable_device(self.serial_no)

        # Enable color stream
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        # Create output directory (relative to interactive_planner root)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.output_dir = os.path.join(base_dir, "captured_images")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        try:
            self._start_pipeline()
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Initial start failed: {e}. Attempting hardware reset...")
            self._reset_device()
            time.sleep(5.0) # Wait for device to re-enumerate
            self._start_pipeline()

    def _start_pipeline(self):
        """Start the RealSense pipeline"""
        # Start streaming
        self.profile = self.pipeline.start(self.config)
        
        # Allow camera to warm up / auto-exposure to settle
        if self.verbose:
            print("📷 RealSense D435: Warming up (2s)...")
        time.sleep(2.0)
        
        if self.verbose:
            print("✅ RealSense D435 initialized successfully")

    def _reset_device(self):
        """Reset the first available RealSense device"""
        try:
            ctx = rs.context()
            devices = ctx.query_devices()
            if len(devices) > 0:
                dev = devices[0]
                if self.verbose:
                    print(f"🔄 Resetting device: {dev.get_info(rs.camera_info.name)}")
                dev.hardware_reset()
            else:
                if self.verbose:
                    print("⚠️  No device found to reset")
        except Exception as e:
            if self.verbose:
                print(f"❌ Failed to reset device: {e}")

    def capture_image(self, filename=None):
        """
        Capture a single color frame and save to disk
        
        Args:
            filename: Optional filename (default: timestamped)
            
        Returns:
            str: Path to saved image file
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Wait for a coherent pair of frames: depth and color
                # Increased timeout to 10000ms (10s)
                frames = self.pipeline.wait_for_frames(timeout_ms=10000)
                color_frame = frames.get_color_frame()
                
                if not color_frame:
                    if self.verbose:
                        print("⚠️  RealSense: No color frame received")
                    continue

                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())

                # Generate filename if not provided
                if filename is None:
                    # Use formatted timestamp: YYYYMMDD_HHMMSS_mmm
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:19]
                    filename = f"observation_{timestamp}.jpg"
                
                # Ensure extension
                if not filename.endswith(('.jpg', '.png', '.jpeg')):
                    filename += ".jpg"

                save_path = os.path.join(self.output_dir, filename)

                # Save image using OpenCV
                cv2.imwrite(save_path, color_image)
                
                if self.verbose:
                    print(f"📸 Image captured: {save_path}")
                    
                return save_path

            except RuntimeError as e:
                if "Frame didn't arrive" in str(e):
                    if self.verbose:
                        print(f"⚠️  Frame timeout (attempt {attempt+1}/{max_retries}). Retrying...")
                    time.sleep(0.5)
                else:
                    if self.verbose:
                        print(f"❌ RealSense runtime error: {str(e)}")
                    return None
            except Exception as e:
                if self.verbose:
                    print(f"❌ RealSense capture failed: {str(e)}")
                return None
        
        if self.verbose:
            print("❌ Failed to capture image after multiple attempts")
        return None

    def close(self):
        """Stop streaming and close pipeline"""
        try:
            self.pipeline.stop()
            if self.verbose:
                print("📷 RealSense pipeline stopped")
        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error stopping pipeline: {str(e)}")

# Example usage
if __name__ == "__main__":
    try:
        camera = RealSenseCameraManager()
        img_path = camera.capture_image()
        print(f"Test capture saved to: {img_path}")
        camera.close()
    except Exception as e:
        print(f"Test failed: {str(e)}")
