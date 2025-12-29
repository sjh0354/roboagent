import os
import dashscope
from dashscope.audio.tts_v2 import SpeechSynthesizer

dashscope.api_key = "sk-dummy-key"

try:
    print("Attempting to initialize SpeechSynthesizer...")
    # Trying the configuration that caused the error
    synthesizer = SpeechSynthesizer(model='cosyvoice-v1', voice='longxiaochun', format='wav')
    
    print("Synthesizer initialized. Calling...")
    response = synthesizer.call(text="Hello world")
    print("Call successful.")
    
except Exception as e:
    print(f"Caught exception: {e}")
    import traceback
    traceback.print_exc()
