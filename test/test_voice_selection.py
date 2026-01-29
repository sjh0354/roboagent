# test_voice_selection.py
import sys
import os

# Add parent directory to path to allow importing utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.tts_manager import TTSManager
import time

def test_voices():
    print("Testing Voice Selection (CosyVoice v3 flash)...")
    
    # Initialize with default (female)
    tts = TTSManager(verbose=True, volume=0.5)
    
    print("\n1. Testing Female voice (longanhuan)...")
    tts.speak("你好，我是龙安欢，这是我的女声测试。", voice="female", block=True)
    
    time.sleep(1)
    
    print("\n2. Testing Male voice (longanyang)...")
    tts.speak("你好，我是龙安阳，这是我的男声测试。", voice="male", block=True)
    
    time.sleep(1)

if __name__ == "__main__":
    test_voices()
