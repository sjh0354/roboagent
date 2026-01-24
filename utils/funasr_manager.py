# utils/funasr_manager.py

"""
FunASR Manager (SenseVoice + FSMN-VAD)
Handles continuous speech recognition, VAD, and wake word detection locally.
"""

import os
import sys
import threading
import time
import queue
import numpy as np
from typing import Optional, List, Dict, Any

# Audio recording
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

# FunASR
try:
    from funasr import AutoModel
    FUNASR_AVAILABLE = True
except ImportError:
    FUNASR_AVAILABLE = False


class FunASRManager:
    """
    Manages Always-On Speech Recognition using FunASR (SenseVoice + FSMN-VAD).
    
    Features:
    - Continuous audio streaming
    - VAD (Voice Activity Detection) to segment speech
    - Local ASR inference using SenseVoiceSmall
    - Wake word detection (text-based)
    """

    def __init__(self, 
                 model_id: str = "iic/SenseVoiceSmall",
                 vad_model_id: str = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch",
                 wake_words: List[str] = ["你好机器人", "开始任务", "小智", "你好"],
                 device: str = "cpu",
                 verbose: bool = True):
        """
        Initialize FunASR Manager

        Args:
            model_id: ModelScope model ID for ASR
            vad_model_id: ModelScope model ID for VAD
            wake_words: List of wake words/phrases to trigger command mode
            device: 'cpu' or 'cuda'
            verbose: Print debug info
        """
        self.verbose = verbose
        self.wake_words = wake_words
        self.is_listening = False
        self.command_queue = queue.Queue()
        self.device = device
        
        # Audio configuration
        self.sample_rate = 16000 # FunASR models usually expect 16k
        
        # Models
        self.model = None
        
        if FUNASR_AVAILABLE:
            if self.verbose:
                print(f"🔄 Loading FunASR models on {device} (this may take a moment)...")
            try:
                # Load SenseVoiceSmall without VAD for debugging/short chunks
                # We are feeding short chunks (3s) so we can rely on model to just output empty if silence
                self.model = AutoModel(
                    model=model_id,
                    # vad_model=vad_model_id, # Disable VAD for now to test raw ASR
                    # vad_kwargs={"max_single_segment_time": 30000},
                    trust_remote_code=True,
                    device=self.device,
                    disable_update=True
                )
                if self.verbose:
                    print("✅ FunASR models loaded successfully (VAD Disabled)")
            except Exception as e:
                print(f"❌ Failed to load FunASR models: {e}")
        else:
            print("❌ funasr not installed. Run: pip install funasr modelscope")

    def start(self):
        """Start the continuous listening thread"""
        if not self.model:
            print("❌ Cannot start listening: Model not loaded")
            return
        if not SOUNDDEVICE_AVAILABLE:
            print("❌ Cannot start listening: sounddevice not available")
            return

        if self.is_listening:
            return

        self.is_listening = True
        self.listen_thread = threading.Thread(target=self._run_listen_loop, daemon=True)
        self.listen_thread.start()
        
        if self.verbose:
            print(f"👂 Continuous listening started (Wake words: {self.wake_words})")

    def stop(self):
        """Stop listening"""
        self.is_listening = False
        if hasattr(self, 'listen_thread'):
            self.listen_thread.join(timeout=2.0)
        if self.verbose:
            print("🛑 Continuous listening stopped")

    def _run_listen_loop(self):
        """
        Simple loop: Record fixed chunks and process.
        For a more advanced version, use VAD streaming API if available in funasr.
        """
        # Chunk duration in seconds
        chunk_duration = 3.0 
        
        while self.is_listening:
            try:
                # Record a chunk
                # We use a slightly overlapping or continuous recording if possible
                # But for simplicity, we record 3s segments
                recording = sd.rec(
                    int(chunk_duration * self.sample_rate), 
                    samplerate=self.sample_rate,
                    channels=1,
                    dtype='float32'
                )
                sd.wait() 
                
                audio_data = recording.flatten()
                
                # Check energy
                # max_energy = np.max(np.abs(audio_data))
                # if self.verbose:
                #     print(f"🔊 Max Energy: {max_energy:.4f}")

                # Check if there is any sound (simple energy threshold to avoid API call if silent)
                if np.max(np.abs(audio_data)) < 0.005: 
                    continue

                # Inference
                # SenseVoiceSmall output: [{'text': '...', 'label': '...', 'emoji': '...'}]
                res = self.model.generate(
                    input=audio_data, 
                    cache={}, 
                    language="zh", 
                    use_itn=True
                )
                
                # if self.verbose:
                #     print(f"DEBUG: raw res: {res}")
                
                if not res or not isinstance(res, list):
                    continue
                
                text = res[0].get("text", "").strip()
                if not text:
                    continue
                
                if self.verbose:
                    # Clean tags like <|zh|> <|HAPPY|> etc.
                    clean_text = self._clean_text(text)
                    if clean_text:
                        print(f"📝 Heard: {clean_text}")
                
                # Wake word logic
                triggered = False
                for ww in self.wake_words:
                    if ww in text:
                        triggered = True
                        break
                
                if triggered:
                    clean_command = self._clean_text(text)
                    if self.verbose:
                        print(f"🚀 Wake word detected! Full text: {clean_command}")
                    self.command_queue.put(clean_command)
                    
            except Exception as e:
                if self.is_listening:
                    print(f"⚠️ ASR Loop Error: {e}")
                time.sleep(0.5)

    def _clean_text(self, text: str) -> str:
        """Remove SenseVoice specific tags like <|zh|>, <|HAPPY|>, etc."""
        import re
        # Remove anything between <| and |>
        # Need to escape | because it is a special regex char (OR)
        return re.sub(r'<\|.*?\|>', '', text).strip()

    def get_command(self) -> Optional[str]:
        """Get latest command from queue (non-blocking)"""
        try:
            return self.command_queue.get_nowait()
        except queue.Empty:
            return None

if __name__ == "__main__":
    # Test script
    manager = FunASRManager(verbose=True)
    manager.start()
    
    print("\nPress Ctrl+C to stop...")
    try:
        while True:
            cmd = manager.get_command()
            if cmd:
                print(f"🔔 COMMAND RECEIVED: {cmd}")
            time.sleep(0.1)
    except KeyboardInterrupt:
        manager.stop()
        print("\nExiting...")
