# tts_manager.py

"""
Text-to-Speech (TTS) Manager
Handles speech synthesis using various backends:
1. DashScope (CosyVoice) - High quality, requires API key
2. System (espeak) - Low quality, offline fallback
3. Mock - Text output only
"""

import os
import subprocess
import threading
import time
from typing import Optional, Dict
from http import HTTPStatus

# Try importing DashScope SDK
try:
    import dashscope
    from dashscope.audio.tts_v2 import SpeechSynthesizer, AudioFormat
    DASHSCOPE_AVAILABLE = True
except ImportError:
    DASHSCOPE_AVAILABLE = False
    dashscope = None

class TTSManager:
    """
    Manages Text-to-Speech synthesis
    """

    def __init__(self, verbose: bool = True, preferred_backend: str = "auto"):
        """
        Initialize TTS Manager

        Args:
            verbose: Print status messages
            preferred_backend: 'dashscope', 'system', or 'auto'
        """
        self.verbose = verbose
        self.preferred_backend = preferred_backend
        
        # Check API key for DashScope
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        
        if self.verbose:
            self._print_status()

    def _print_status(self):
        """Print initialization status"""
        print(f"🎤 TTS Manager Initialized")
        print(f"   - DashScope SDK: {'✅ Available' if DASHSCOPE_AVAILABLE else '❌ Not installed (pip install dashscope)'}")
        print(f"   - API Key: {'✅ Found' if self.api_key else '❌ Not found (DASHSCOPE_API_KEY)'}")
        print(f"   - System TTS: {'✅ Available (espeak)' if self._check_espeak() else '⚠️ Not found'}")

    def _check_espeak(self) -> bool:
        """Check if espeak is available"""
        try:
            subprocess.run(["which", "espeak"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def speak(self, text: str, model: str = "cosyvoice-v1", block: bool = False):
        """
        Speak the given text

        Args:
            text: Text to speak
            model: Model name for DashScope (e.g., 'cosyvoice-v1', 'sambert-zh-v1')
            block: If True, wait for speech to finish (not fully supported for all backends)
        """
        if not text:
            return

        # Determine backend
        backend = self._choose_backend()
        
        if self.verbose:
            print(f"🗣️  Speaking ({backend}): \"{text}\"")

        if backend == "dashscope":
            self._speak_dashscope(text, model, block)
        elif backend == "system":
            self._speak_system(text, block)
        else:
            self._speak_mock(text)

    def _choose_backend(self) -> str:
        """Choose the best available backend"""
        if self.preferred_backend == "dashscope" and DASHSCOPE_AVAILABLE and self.api_key:
            return "dashscope"
        elif self.preferred_backend == "system":
            return "system"
        
        # Auto selection
        if DASHSCOPE_AVAILABLE and self.api_key:
            return "dashscope"
        elif self._check_espeak():
            return "system"
        else:
            return "mock"

    def _speak_dashscope(self, text: str, model: str, block: bool):
        """Speak using DashScope API (CosyVoice)"""
        if not DASHSCOPE_AVAILABLE:
            if self.verbose:
                print("⚠️  DashScope SDK not installed. Falling back to system.")
            self._speak_system(text, block)
            return

        def _run():
            try:
                import uuid
                filename = f"/tmp/tts_{uuid.uuid4()}.wav"
                
                # Configure synthesizer
                # Note: 'cosyvoice-v1' might be the specific model name user wants
                # format should be passed to constructor for tts_v2
                synthesizer = SpeechSynthesizer(model=model, voice='longxiaochun', format=AudioFormat.WAV_16000HZ_MONO_16BIT) 
                
                # Call API
                # The 5s timeout seems to be a hard limit for connection in some SDK versions.
                # We try once; if it fails, we fall back to system immediately to avoid long delays.
                response = synthesizer.call(text=text, timeout_millis=30000)
                
                # Check if response is bytes (success) or something else
                if isinstance(response, bytes):
                    with open(filename, 'wb') as f:
                        f.write(response)
                    
                    self._play_audio(filename)
                    
                    if os.path.exists(filename):
                        os.remove(filename)
                elif hasattr(response, 'status_code') and response.status_code == HTTPStatus.OK:
                    # Handle case where it returns a response object (legacy/different version behavior)
                    with open(filename, 'wb') as f:
                         # Try to get audio data from response object
                         if hasattr(response, 'get_audio_data'):
                             f.write(response.get_audio_data())
                         elif hasattr(response, 'output') and hasattr(response.output, 'audio'):
                             # Some dashscope responses have output.audio (though usually for async)
                             f.write(response.output.audio)
                         else:
                             # Fallback, maybe response itself is iterable?
                             f.write(response.content if hasattr(response, 'content') else b'')

                    self._play_audio(filename)
                    
                    if os.path.exists(filename):
                        os.remove(filename)
                else:
                    if self.verbose:
                         msg = response.message if hasattr(response, 'message') else str(response)
                         print(f"❌ DashScope TTS API Error: {msg}")
                    # Fallback
                    self._speak_system(text, block=True)

            except Exception as e:
                if self.verbose:
                    print(f"❌ DashScope TTS Exception: {str(e)}. Falling back to system.")
                self._speak_system(text, block=True)

        if block:
            _run()
        else:
            threading.Thread(target=_run).start()

    def _speak_system(self, text: str, block: bool):
        """Speak using system espeak"""
        def _run():
            try:
                # Use espeak
                subprocess.run(["espeak", text], check=True)
            except Exception as e:
                if self.verbose:
                    print(f"❌ System TTS failed: {e}")

        if block:
            _run()
        else:
            threading.Thread(target=_run).start()

    def _speak_mock(self, text: str):
        """Mock speak (print only)"""
        # Already printed in speak()
        pass

    def _play_audio(self, filename: str):
        """Play audio file using system player"""
        try:
            # Try aplay (linux, wav only)
            if subprocess.run(["which", "aplay"], stdout=subprocess.DEVNULL, check=False).returncode == 0:
                subprocess.run(["aplay", "-q", filename], check=True)
                return
            
            # Try ffplay
            if subprocess.run(["which", "ffplay"], stdout=subprocess.DEVNULL, check=False).returncode == 0:
                subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", filename], check=True)
                return

            if self.verbose:
                print(f"⚠️  No audio player (aplay/ffplay) found. Saved to {filename}")
        except Exception as e:
            if self.verbose:
                print(f"❌ Audio playback failed: {e}")

# Example usage
if __name__ == "__main__":
    # Test
    manager = TTSManager(verbose=True)
    manager.speak("你好，你好，可以听到我说话吗？", block=True)