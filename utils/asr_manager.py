# utils/asr_manager.py

"""
Automatic Speech Recognition (ASR) Manager
Handles speech-to-text using qwen3-asr-toolkit (QwenASR).
"""

import os
import tempfile
from typing import Optional

# Try importing sounddevice and scipy for recording
try:
    import sounddevice as sd
    import scipy.io.wavfile as wav
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    sd = None
    wav = None

# Try importing qwen3-asr-toolkit
try:
    from qwen3_asr_toolkit.qwen3asr import QwenASR
    QWEN_ASR_AVAILABLE = True
except ImportError:
    QWEN_ASR_AVAILABLE = False


class ASRManager:
    """
    Manages Automatic Speech Recognition (Speech-to-Text) using QwenASR
    """

    def __init__(self, verbose: bool = True, model: str = "qwen3-asr-flash"):
        """
        Initialize ASR Manager

        Args:
            verbose: Print status messages
            model: Qwen ASR model (default: 'qwen3-asr-flash')
        """
        self.verbose = verbose
        self.model = model
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        
        self.qwen_asr = None
        if QWEN_ASR_AVAILABLE:
            try:
                self.qwen_asr = QwenASR(model=self.model)
            except Exception as e:
                print(f"❌ Failed to initialize QwenASR: {e}")
        
        if self.verbose:
            self._print_status()

    def _print_status(self):
        """Print initialization status"""
        print(f"👂 ASR Manager Initialized")
        print(f"   - QwenASR Toolkit: {'✅ Available' if QWEN_ASR_AVAILABLE else '❌ Not installed (pip install qwen3-asr-toolkit)'}")
        print(f"   - API Key: {'✅ Found' if self.api_key else '❌ Not found (DASHSCOPE_API_KEY)'}")
        print(f"   - Audio Recording: {'✅ Available (sounddevice)' if SOUNDDEVICE_AVAILABLE else '❌ Not installed (pip install sounddevice scipy)'}")
        print(f"   - Model: {self.model}")

    def record_audio(self, duration: float = 5.0, sample_rate: int = 48000) -> Optional[str]:
        """
        Record audio from microphone

        Args:
            duration: Recording duration in seconds
            sample_rate: Sample rate (48000 is default for many USB mics)

        Returns:
            str: Path to recorded wav file, or None if failed
        """
        if not SOUNDDEVICE_AVAILABLE:
            if self.verbose:
                print("❌ sounddevice or scipy not available. Cannot record.")
            return None

        # Find input device
        device_id = None
        try:
            devices = sd.query_devices()
            for i, dev in enumerate(devices):
                if "Loostone" in dev['name'] and dev['max_input_channels'] > 0:
                    device_id = i
                    if self.verbose:
                        print(f"🎤 Using microphone: {dev['name']} (Index: {i})")
                    break
            
            if device_id is None and self.verbose:
                 print("⚠️  Loostone microphone not found. Using default device.")

        except Exception as e:
            if self.verbose:
                print(f"⚠️  Error querying devices: {e}")

        if self.verbose:
            print(f"🎙️  Recording for {duration} seconds... (Speak now)")

        try:
            # Record
            recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype='int16', device=device_id)
            sd.wait()  # Wait until recording is finished
            
            # Save to temp file
            temp_file = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            filename = temp_file.name
            temp_file.close()
            
            wav.write(filename, sample_rate, recording)
            
            if self.verbose:
                print(f"✅ Recording saved to {filename}")
                
            return filename
            
        except Exception as e:
            if self.verbose:
                print(f"❌ Recording failed: {e}")
            return None

    def transcribe_audio(self, audio_file: str) -> str:
        """
        Transcribe audio file using QwenASR

        Args:
            audio_file: Path to wav file

        Returns:
            str: Transcribed text
        """
        if not QWEN_ASR_AVAILABLE:
            return "Error: qwen3-asr-toolkit not installed."
        
        if not self.qwen_asr:
             return "Error: QwenASR not initialized."

        if not self.api_key:
            return "Error: DASHSCOPE_API_KEY not found."

        if not os.path.exists(audio_file):
            return "Error: Audio file not found."

        if self.verbose:
            print(f"🔄 Transcribing with {self.model}...")

        try:
            # Pass absolute path just in case
            abs_path = os.path.abspath(audio_file)
            
            # Call QwenASR
            # Returns: (language, text)
            language, text = self.qwen_asr.asr(abs_path)
            
            # Cleanup text
            text = str(text).strip()
            
            if self.verbose:
                print(f"✅ Transcription ({language}): \"{text}\"")
            return text

        except Exception as e:
            error_msg = f"ASR Exception: {str(e)}"
            if self.verbose:
                print(f"❌ {error_msg}")
            return f"Error: {error_msg}"

    def listen_and_transcribe(self, duration: float = 5.0) -> str:
        """
        Record and transcribe in one go (Convenience method)
        
        Args:
            duration: Duration to record in seconds

        Returns:
            str: Transcribed text
        """
        filename = self.record_audio(duration)
        if filename:
            text = self.transcribe_audio(filename)
            # Cleanup
            try:
                os.remove(filename)
            except OSError:
                pass
            return text
        return "Error: Recording failed."


if __name__ == "__main__":
    # Test
    manager = ASRManager(verbose=True)
    if SOUNDDEVICE_AVAILABLE and QWEN_ASR_AVAILABLE:
        print("\nTesting: Speak now (5 seconds)...")
        result = manager.listen_and_transcribe(5.0)
        print(f"\nFinal Result: {result}")
    else:
        print("\nCannot test: Dependencies missing.")