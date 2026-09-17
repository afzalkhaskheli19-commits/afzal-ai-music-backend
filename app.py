from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import base64
import io

app = FastAPI(
    title="Afzal AI Music Backend",
    version="3.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_LANGUAGES = [
    "Sindhi",
    "Balochi",
    "Urdu",
    "Arabic",
    "Punjabi",
    "Marvadi",
    "English",
]

SUPPORTED_VOCAL_STYLES = [
    "Male Sufi",
    "Female",
    "Male + Female",
    "Male Folk",
    "Emotional Male",
]

class GenerateRequest(BaseModel):
    lyrics: str
    language: str
    vocal_style: str
    music_style: str
    duration_seconds: Optional[int] = 20

@app.get("/")
def home():
    return {
        "service": "Afzal AI Music Backend",
        "status": "online",
        "languages": SUPPORTED_LANGUAGES,
        "vocal_styles": SUPPORTED_VOCAL_STYLES,
        "note": "Music generation endpoint is ready."
    }

@app.get("/languages")
def languages():
    return {"languages": SUPPORTED_LANGUAGES}

@app.post("/generate")
def generate_music(request: GenerateRequest):
    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language. Choose one of: {', '.join(SUPPORTED_LANGUAGES)}")

    if request.vocal_style not in SUPPORTED_VOCAL_STYLES:
        raise HTTPException(status_code=400, detail=f"Unsupported vocal style. Choose one of: {', '.join(SUPPORTED_VOCAL_STYLES)}")

    if not request.lyrics.strip():
        raise HTTPException(status_code=400, detail="Lyrics are required.")

    try:
        from transformers import AutoProcessor, MusicgenForConditionalGeneration
        import scipy.io.wavfile

        model_name = "facebook/musicgen-small"
        processor = AutoProcessor.from_pretrained(model_name)
        model = MusicgenForConditionalGeneration.from_pretrained(model_name)

        full_prompt = f"[{request.vocal_style} vocal in {request.language} language] {request.music_style} style. Lyrics theme: {request.lyrics}"

        inputs = processor(text=[full_prompt], padding=True, return_tensors="pt")
        max_new_tokens = max(256, int(request.duration_seconds * 50))
        audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)

        sampling_rate = model.config.audio_encoder.sampling_rate
        audio_array = audio_values[0, 0].detach().cpu().numpy()

        wav_buffer = io.BytesIO()
        scipy.io.wavfile.write(wav_buffer, rate=sampling_rate, data=audio_array)
        audio_base64 = base64.b64encode(wav_buffer.getvalue()).decode("utf-8")

        return {
            "success": True,
            "language": request.language,
            "vocal_style": request.vocal_style,
            "music_style": request.music_style,
            "audio_base64": audio_base64,
            "format": "wav"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Music generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=7860, reload=False)
