# utils/funasr_manager.py

"""
FunASR Remote Client Manager
Handles continuous audio capture and sends chunks to a remote FunASR server for inference.
This protects local storage from heavy model checkpoints.
"""

import os
import threading
import time
import queue
import numpy as np
import io
import requests
import scipy.io.wavfile as wav
from scipy import signal
from typing import Optional, List

# Audio recording
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False


class FunASRManager:
    """
    Lightweight client for Always-On Speech Recognition via remote server.
    """

    def __init__(self, 
                 wake_words: List[str] = ["你好机器人", "开始任务", "小智", "你好"],
                 server_url: Optional[str] = None,
                 verbose: bool = True):
        """
        Initialize FunASR Remote Manager

        Args:
            wake_words: List of wake words/phrases to trigger command mode
            server_url: URL of the remote ASR server (e.g., "http://192.168.1.100:8000").
                        If None, reads from 'ASR_SERVER_URL' environment variable.
            verbose: Print debug info
        """
        self.verbose = verbose
        self.wake_words = wake_words
        self.is_listening = False
        self.command_queue = queue.Queue()
        
        # Resolve server URL
        self.server_url = server_url or os.getenv("ASR_SERVER_URL")
        if not self.server_url:
            print("⚠️  Warning: ASR_SERVER_URL not set. Remote inference will fail.")
        elif self.verbose:
            print(f"🌐 FunASR Client configured for server: {self.server_url}")

        # Audio configuration
        self.target_sample_rate = 16000  # Server expects 16kHz
        self.device_sample_rate = 16000  # Will be updated based on device
        self.device_id = None

        if not SOUNDDEVICE_AVAILABLE:
            print("❌ Error: 'sounddevice' not installed. Cannot capture audio.")
        else:
            # Find Loostone microphone
            try:
                devices = sd.query_devices()
                for i, dev in enumerate(devices):
                    if "Loostone" in dev['name'] and dev['max_input_channels'] > 0:
                        self.device_id = i
                        self.device_sample_rate = int(dev['default_samplerate'])
                        if self.verbose:
                            print(f"🎤 Using microphone: {dev['name']} (Index: {i}, Rate: {self.device_sample_rate}Hz)")
                        break

                if self.device_id is None and self.verbose:
                    print("⚠️  Loostone microphone not found. Using default device.")
            except Exception as e:
                if self.verbose:
                    print(f"⚠️  Error querying audio devices: {e}")

    def start(self):
        """Start the continuous listening thread"""
        if not self.server_url:
            print("❌ Cannot start listening: Remote server URL is missing.")
            return
        if not SOUNDDEVICE_AVAILABLE:
            print("❌ Cannot start listening: Audio hardware/library not available.")
            return

        if self.is_listening:
            return

        self.is_listening = True
        self.listen_thread = threading.Thread(target=self._run_listen_loop, daemon=True)
        self.listen_thread.start()
        
        if self.verbose:
            print(f"👂 Remote listening started -> {self.server_url}")
            print(f"   Wake words: {self.wake_words}")

    def stop(self):
        """Stop listening"""
        self.is_listening = False
        if hasattr(self, 'listen_thread'):
            self.listen_thread.join(timeout=2.0)
        if self.verbose:
            print("🛑 Remote listening stopped")

    def _run_listen_loop(self):
        """
        Main Loop: Record Chunk -> Send to Server -> Wake Word Check -> Queue
        """
        chunk_duration = 3.0 # 3-second segments
        
        while self.is_listening:
            try:
                # 1. Record Audio at device's native sample rate
                recording = sd.rec(
                    int(chunk_duration * self.device_sample_rate),
                    samplerate=self.device_sample_rate,
                    channels=1,
                    dtype='float32',
                    device=self.device_id
                )
                sd.wait() 
                
                # Check energy to skip absolute silence
                if np.max(np.abs(recording)) < 0.005: 
                    continue

                # 2. Remote Inference
                text = self._infer_remote(recording)
                
                if not text:
                    continue
                
                if self.verbose:
                    print(f"📝 Heard: {text}")
                
                # 3. Wake Word Logic
                triggered = False
                for ww in self.wake_words:
                    if ww in text:
                        triggered = True
                        break
                
                if triggered:
                    if self.verbose:
                        print(f"🚀 Wake word detected! Command: {text}")
                    self.command_queue.put(text)
                    
            except Exception as e:
                if self.is_listening:
                    print(f"⚠️ Remote ASR Loop Error: {e}")
                time.sleep(0.5)

    def _infer_remote(self, audio_data: np.ndarray) -> str:
        """Send audio to remote server and get transcription"""
        try:
            # Resample to target rate if needed
            if self.device_sample_rate != self.target_sample_rate:
                num_samples = int(len(audio_data) * self.target_sample_rate / self.device_sample_rate)
                audio_data = signal.resample(audio_data, num_samples)

            # Convert float32 numpy to WAV bytes
            audio_int16 = (audio_data * 32767).astype(np.int16)
            byte_io = io.BytesIO()
            wav.write(byte_io, self.target_sample_rate, audio_int16)
            byte_io.seek(0)
            
            files = {'file': ('chunk.wav', byte_io, 'audio/wav')}
            
            # Prepare endpoint
            endpoint = self.server_url.rstrip("/") + "/transcribe"
            
            response = requests.post(endpoint, files=files, timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                return data.get("text", "").strip()
            else:
                if self.verbose:
                    print(f"❌ Server Error: {response.status_code}")
        except Exception as e:
            if self.verbose:
                print(f"📡 Connection Error: {e}")
        return ""

    def get_command(self) -> Optional[str]:
        """Get latest command from queue (non-blocking)"""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

if __name__ == "__main__":
    # Quick Test
    # export ASR_SERVER_URL="http://YOUR_WORKSTATION_IP:8000"
    manager = FunASRManager(verbose=True)
    manager.start()
    
    try:
        while True:
            cmd = manager.get_command()
            if cmd:
                print(f"🔔 RECEIVED: {cmd}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        manager.stop()