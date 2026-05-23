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

    # Voice mapping for common tones (using CosyVoice v3 flash voices)
    VOICE_MAPPING = {
        "female": "longanhuan",    # Standard female
        "male": "longanyang",      # Standard male
    }

    def __init__(self, verbose: bool = True, preferred_backend: str = "auto", volume: float = 1.0, voice: str = "female"):
        """
        Initialize TTS Manager

        Args:
            verbose: Print status messages
            preferred_backend: 'dashscope', 'system', or 'auto'
            volume: Default playback volume (0.0 to 1.0)
            voice: Default voice name or mapping key ('male', 'female', etc.)
        """
        self.verbose = verbose
        self.preferred_backend = preferred_backend
        self.volume = max(0.0, min(1.0, volume)) # Clamp between 0.0 and 1.0
        
        # Resolve voice
        self.voice = self.VOICE_MAPPING.get(voice, voice)
        
        # Check API key for DashScope. GENAI_API_KEY is accepted for compatibility
        # with the Gemini/DashScope-compatible planner environment.
        self.api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("GENAI_API_KEY")
        if DASHSCOPE_AVAILABLE and self.api_key:
            dashscope.api_key = self.api_key
        
        if self.verbose:
            self._print_status()

    def _print_status(self):
        """Print initialization status"""
        print(f"🎤 TTS Manager Initialized")
        print(f"   - DashScope SDK: {'✅ Available' if DASHSCOPE_AVAILABLE else '❌ Not installed (pip install dashscope)'}")
        print(f"   - API Key: {'✅ Found' if self.api_key else '❌ Not found (DASHSCOPE_API_KEY or GENAI_API_KEY)'}")
        system_backend = self._system_tts_backend()
        print(f"   - System TTS: {f'✅ Available ({system_backend})' if system_backend else '⚠️ Not found'}")
        print(f"   - Volume: {int(self.volume * 100)}%")
        print(f"   - Default Voice: {self.voice}")

    def _check_espeak(self) -> bool:
        """Check if espeak is available"""
        return self._command_exists("espeak")

    def _command_exists(self, command: str) -> bool:
        """Check if a command is available on PATH."""
        try:
            subprocess.run(["which", command], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except subprocess.CalledProcessError:
            return False

    def _system_tts_backend(self) -> Optional[str]:
        """Return the first available offline/system TTS backend."""
        if self._command_exists("espeak"):
            return "espeak"
        if self._command_exists("spd-say"):
            return "spd-say"
        return None

    def speak(self, text: str, model: str = "cosyvoice-v3-flash", block: bool = False, volume: Optional[float] = None, voice: Optional[str] = None):
        """
        Speak the given text

        Args:
            text: Text to speak
            model: Model name for DashScope (default: 'cosyvoice-v3-flash')
            block: If True, wait for speech to finish (not fully supported for all backends)
            volume: Override default volume (0.0 to 1.0), if None use self.volume
            voice: Override default voice
        """
        if not text:
            return

        # Determine volume
        current_volume = self.volume if volume is None else max(0.0, min(1.0, volume))
        
        # Determine voice
        current_voice = self.voice if voice is None else self.VOICE_MAPPING.get(voice, voice)

        # Determine backend
        backend = self._choose_backend()
        
        if self.verbose:
            print(f"🗣️  Speaking ({backend}, vol={current_volume:.1f}, voice={current_voice}): \"{text}\"")

        if backend == "dashscope":
            self._speak_dashscope(text, model, block, current_volume, current_voice)
        elif backend == "system":
            self._speak_system(text, block, current_volume)
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
        elif self._system_tts_backend():
            return "system"
        else:
            return "mock"

    def _speak_dashscope(self, text: str, model: str, block: bool, volume: float, voice: str):
        """Speak using DashScope API (CosyVoice)"""
        if not DASHSCOPE_AVAILABLE:
            if self.verbose:
                print("⚠️  DashScope SDK not installed. Falling back to system.")
            self._speak_system(text, block, volume)
            return

        def _run():
            try:
                import uuid
                filename = f"/tmp/tts_{uuid.uuid4()}.wav"
                
                # Configure synthesizer
                # Use passed voice parameter
                synthesizer = SpeechSynthesizer(model=model, voice=voice, format=AudioFormat.WAV_16000HZ_MONO_16BIT) 
                connect_timeout = float(os.getenv("TTS_DASHSCOPE_CONNECT_TIMEOUT_SECONDS", "15"))
                if connect_timeout > 0 and hasattr(synthesizer, "_SpeechSynthesizer__connect"):
                    # DashScope SDK's normal call path hard-codes a 5s WebSocket connect
                    # wait. Preconnecting lets slow but usable networks finish the handshake.
                    synthesizer._SpeechSynthesizer__connect(connect_timeout)
                
                # Call API
                synthesis_timeout_ms = int(os.getenv("TTS_DASHSCOPE_SYNTHESIS_TIMEOUT_MS", "30000"))
                response = synthesizer.call(text=text, timeout_millis=synthesis_timeout_ms)
                
                # Check if response is bytes (success) or something else
                if isinstance(response, bytes):
                    with open(filename, 'wb') as f:
                        f.write(response)
                    
                    self._play_audio(filename, volume)
                    
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

                    self._play_audio(filename, volume)
                    
                    if os.path.exists(filename):
                        os.remove(filename)
                else:
                    if self.verbose:
                         msg = response.message if hasattr(response, 'message') else str(response)
                         print(f"❌ DashScope TTS API Error: {msg}")
                    # Fallback
                    self._speak_system(text, block=True, volume=volume)

            except Exception as e:
                if self.verbose:
                    print(f"❌ DashScope TTS Exception: {str(e)}. Falling back to system.")
                self._speak_system(text, block=True, volume=volume)

        if block:
            _run()
        else:
            threading.Thread(target=_run).start()

    def _speak_system(self, text: str, block: bool, volume: float):
        """Speak using system espeak"""
        def _run():
            try:
                backend = self._system_tts_backend()
                if backend == "espeak":
                    # espeak amplitude: 0 to 200, default 100.
                    amplitude = int(volume * 100)
                    subprocess.run(["espeak", "-a", str(amplitude), text], check=True)
                elif backend == "spd-say":
                    intensity = int((volume * 200) - 100)
                    cmd = ["spd-say", "-i", str(intensity)]
                    if block:
                        cmd.append("-w")
                    cmd.append(text)
                    subprocess.run(cmd, check=True)
                else:
                    raise FileNotFoundError("No system TTS backend found (espeak/spd-say)")
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

    def _play_audio(self, filename: str, volume: float):
        """Play audio file using system player with volume control"""
        try:
            # Try ffplay (cross-platform, requires ffmpeg, good volume control)
            if subprocess.run(["which", "ffplay"], stdout=subprocess.DEVNULL, check=False).returncode == 0:
                # -af volume=... (float)
                subprocess.run(["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", "-af", f"volume={volume}", filename], check=True)
                return

            # Try paplay (PulseAudio - common on Ubuntu Desktop)
            if subprocess.run(["which", "paplay"], stdout=subprocess.DEVNULL, check=False).returncode == 0:
                # --volume=VOLUME (0=muted, 65536=100%)
                pa_volume = int(volume * 65536)
                subprocess.run(["paplay", "--volume", str(pa_volume), filename], check=True)
                return

            # Try aplay (ALSA - linux, wav only)
            if subprocess.run(["which", "aplay"], stdout=subprocess.DEVNULL, check=False).returncode == 0:
                if self.verbose and volume != 1.0:
                    print("⚠️  aplay does not support direct volume control. Playing at system volume.")
                subprocess.run(["aplay", "-q", filename], check=True)
                return
            
            if self.verbose:
                print(f"⚠️  No audio player (ffplay/paplay/aplay) found. Saved to {filename}")
        except Exception as e:
            if self.verbose:
                print(f"❌ Audio playback failed: {e}")

# Example usage
if __name__ == "__main__":
    # Test
    manager = TTSManager(verbose=True, volume=0.1)
    manager.speak("你好，我现在的音量是百分之十。", block=True)
    #manager.speak("Checking volume fifty percent.", volume=0.5, block=True)
