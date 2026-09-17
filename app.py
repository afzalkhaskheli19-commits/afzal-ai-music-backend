
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

# Allow the Netlify website to connect to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 7 languages
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
        "note": "Music generation endpoint is ready. MusicGen generates instrumental audio; sung vocals require a separate singing/voice model or API."
    }

@app.get("/languages")
def languages():
    return {
        "languages": SUPPORTED_LANGUAGES
    }

@app.post("/generate")
def generate_music(request: GenerateRequest):
    if request.language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported language. Choose one of: {', '.join(SUPPORTED_LANGUAGES)}"
        )

    if request.vocal_style not in SUPPORTED_VOCAL_STYLES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported vocal style. Choose one of: {', '.join(SUPPORTED_VOCAL_STYLES)}"
        )

    if not request.lyrics.strip():
        raise HTTPException(status_code=400, detail="Lyrics are required.")

    # MusicGen is loaded only when generation is requested.
    try
