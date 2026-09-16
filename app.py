from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import scipy.io.wavfile
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import base64
import io

app = FastAPI(title="Afzal AI Music Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_NAME = "facebook/musicgen-small"
processor = None
model = None

class MusicRequest(BaseModel):
    lyrics: str = ""
    language: str = "Sindhi"
    vocal_style: str = "Male Sufi"
    music_style: str = "Sindhi Sufi Folk"
    duration_seconds: int = 15

@app.get("/")
def home():
    return {"status": "ok", "service": "Afzal AI Music Backend"}

@app.post("/generate")
def generate_music(request: MusicRequest):
    global processor, model
    try:
        if processor is None or model is None:
            processor = AutoProcessor.from_pretrained(MODEL_NAME)
            model = MusicgenForConditionalGeneration.from_pretrained(MODEL_NAME)

        duration = max(5, min(int(request.duration_seconds), 30))
        prompt = (
            f"{request.language} music, {request.vocal_style}, "
            f"{request.music_style}. Create an instrumental musical arrangement "
            f"inspired by these lyrics: {request.lyrics[:2000]}"
        )

        inputs = processor(text=[prompt], padding=True, return_tensors="pt")
        max_new_tokens = int(duration * 50)

        with torch.no_grad():
            audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)

        audio = audio_values[0, 0].cpu().numpy()
        sample_rate = model.config.audio_encoder.sampling_rate

        wav_buffer = io.BytesIO()
        scipy.io.wavfile.write(wav_buffer, rate=sample_rate, data=audio)
        audio_base64 = base64.b64encode(wav_buffer.getvalue()).decode("utf-8")

        return {
            "success": True,
            "format": "wav",
            "audio_base64": audio_base64,
            "note": "MusicGen currently creates instrumental music; singing vocals require a separate singing-voice model."
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
