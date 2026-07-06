import whisper
import os
import requests
from pydub import AudioSegment

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small")

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

SARVAM_LANG_MAP = {
    "hinglish": "hi-IN",
    "hindi": "hi-IN",
    "bengali": "bn-IN",
    "gujarati": "gu-IN",
    "kannada": "kn-IN",
    "malayalam": "ml-IN",
    "marathi": "mr-IN",
    "odia": "od-IN",
    "punjabi": "pa-IN",
    "tamil": "ta-IN",
    "telugu": "te-IN",
}

_model = None


def load_model():
    global _model
    if _model is None:
        print(f"Loading Whisper model: {WHISPER_MODEL} ...")
        _model = whisper.load_model(WHISPER_MODEL)
        print("Whisper model loaded.")
    return _model


def transcribe_chunk_whisper(chunk_path: str) -> str:
    whisper_api_url = os.getenv("WHISPER_API_URL")
    if whisper_api_url:
        print(f"Offloading Whisper transcription to API: {whisper_api_url} ...")
        try:
            with open(chunk_path, "rb") as f:
                files = {"file": (os.path.basename(chunk_path), f, "audio/wav")}
                headers = {}
                hf_token = os.getenv("HF_TOKEN")
                if hf_token:
                    headers["Authorization"] = f"Bearer {hf_token}"
                response = requests.post(whisper_api_url, headers=headers, files=files, timeout=300)
            if response.ok:
                res_json = response.json()
                return res_json.get("text", res_json.get("transcript", ""))
            else:
                print(f"Whisper API returned error {response.status_code}: {response.text}")
                print("Falling back to local Whisper...")
        except Exception as e:
            print(f"Failed to call Whisper API: {e}")
            print("Falling back to local Whisper...")

    model = load_model()
    result = model.transcribe(chunk_path, task="transcribe")
    return result["text"]


def _send_to_sarvam(piece_path: str, language_code: str = None) -> str:
    """Send one ≤30s WAV file to Sarvam and return the English transcript."""
    headers = {"api-subscription-key": SARVAM_API_KEY}
    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {"model": SARVAM_MODEL, "with_diarization": "false"}
        if language_code:
            data["language_code"] = language_code
        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=120,
        )
    if not response.ok:
        print("Sarvam returned", response.status_code)
        print(f"Response body: {response.text}")
        response.raise_for_status()
    return response.json().get("transcript", "")


def transcribe_chunk_sarvam(chunk_path: str, language_code: str = None) -> str:
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")
    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000
    full_text = ""
    total_pieces = (len(audio) + piece_ms - 1) // piece_ms
    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start: start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")
        try:
            print(f"  → Sarvam piece {i + 1}/{total_pieces} ...")
            full_text += _send_to_sarvam(piece_path, language_code) + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)
    return full_text.strip()


def transcribe_chunk(chunk_path: str, language: str = "english") -> str:
    lang_lower = language.lower()
    if lang_lower in SARVAM_LANG_MAP:
        return transcribe_chunk_sarvam(chunk_path, SARVAM_LANG_MAP[lang_lower])
    return transcribe_chunk_whisper(chunk_path)


def transcribe_all(chunks: list, language: str = "english") -> str:
    full_transcript = ""
    lang_lower = language.lower()
    engine = "Sarvam AI" if lang_lower in SARVAM_LANG_MAP else "Whisper"
    print(f"Using {engine} for transcription.")
    for i, chunk in enumerate(chunks):
        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")
        text = transcribe_chunk(chunk, language=language)
        full_transcript += text + " "
    print("Transcription complete.")
    return full_transcript.strip()
