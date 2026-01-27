from fastapi import FastAPI, UploadFile, File, Form
from funasr import AutoModel
import uvicorn
import os
import shutil
import tempfile
import aiofiles

# Configuration
MODEL_ID = "iic/SenseVoiceSmall"
VAD_MODEL_ID = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
HOST = "0.0.0.0"
PORT = 8000

app = FastAPI(title="FunASR Remote Server")

print("🔄 Loading FunASR models on Server...")
# Load models (use GPU if available)
model = AutoModel(
    model=MODEL_ID,
    # vad_model=VAD_MODEL_ID, # Optional: Enable VAD on server if strictly needed, but client chunks usually suffice
    trust_remote_code=True,
    device="cuda", # Default to GPU on workstation
    disable_update=True
)
print("✅ Models loaded!")

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...), language: str = Form("zh")):
    """
    Receive audio file (wav/mp3/pcm), perform ASR, return text.
    """
    temp_filename = f"temp_{file.filename}"
    
    try:
        # Save uploaded file to temp
        async with aiofiles.open(temp_filename, 'wb') as out_file:
            content = await file.read()
            await out_file.write(content)
        
        # Inference
        # SenseVoice supports input as file path
        res = model.generate(
            input=temp_filename,
            cache={},
            language=language,
            use_itn=True
        )
        
        # Parse result
        text = ""
        if res and isinstance(res, list):
            text = res[0].get("text", "")
        
        # Clean text (remove tags)
        import re
        clean_text = re.sub(r'<\|.*?\|>', '', text).strip()
        
        print(f"📝 Transcribed: {clean_text}")
        return {"text": clean_text}

    except Exception as e:
        print(f"❌ Error: {str(e)}")
        return {"error": str(e)}
        
    finally:
        # Cleanup
        if os.path.exists(temp_filename):
            os.remove(temp_filename)

if __name__ == "__main__":
    print(f"🚀 Starting FunASR Server on {HOST}:{PORT}")
    uvicorn.run(app, host=HOST, port=PORT)
