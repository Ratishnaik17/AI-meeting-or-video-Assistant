import os
import shutil
import tempfile
import whisper
from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI(title="Whisper Transcription Service")

# Load model at startup (uses model specified by environment variable or defaults to "small")
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")
print(f"Loading Whisper model: {WHISPER_MODEL} ...")
model = whisper.load_model(WHISPER_MODEL)
print("Whisper model loaded successfully.")

@app.get("/")
def health_check():
    return {"status": "ok", "model": WHISPER_MODEL}

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    # Verify file extension
    filename = file.filename
    if not filename:
        raise HTTPException(status_code=400, detail="Filename missing")
    
    # Save the uploaded file to a temporary location
    suffix = os.path.splitext(filename)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        temp_path = tmp.name
        shutil.copyfileobj(file.file, tmp)
        
    try:
        # Run Whisper transcription
        result = model.transcribe(temp_path, task="transcribe")
        return {"text": result.get("text", "").strip()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Clean up temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
