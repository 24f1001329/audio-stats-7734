import os
import json
import base64
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from openai import OpenAI
app = FastAPI()
AIPIPE_TOKEN = os.environ.get("AIPIPE_TOKEN")
AIPIPE_BASE_URL = os.environ.get("AIPIPE_BASE_URL", "https://aipipe.org/openai/v1")
client = OpenAI(api_key=AIPIPE_TOKEN, base_url=AIPIPE_BASE_URL)
SCHEMA_KEYS = [
    "rows", "columns", "mean", "std", "variance", "min", "max",
    "median", "mode", "range", "allowed_values", "value_range", "correlation"
]
EMPTY_RESULT = {
    "rows": 0, "columns": [], "mean": {}, "std": {}, "variance": {},
    "min": {}, "max": {}, "median": {}, "mode": {}, "range": {},
    "allowed_values": {}, "value_range": {}, "correlation": []
}
SYSTEM_PROMPT = """You will be given spoken audio describing a small dataset
(column names, values, and/or statistics spoken aloud).
Listen carefully, reconstruct the dataset as best as possible, then
compute and return ONLY a single JSON object with EXACTLY these keys,
no extra keys, no markdown, no explanation, no code fences:
{
  "rows": <integer, number of data rows>,
  "columns": [<list of column name strings>],
  "mean": {<column: number>, ...},
  "std": {<column: number>, ...},
  "variance": {<column: number>, ...},
  "min": {<column: number>, ...},
  "max": {<column: number>, ...},
  "median": {<column: number>, ...},
  "mode": {<column: number or string>, ...},
  "range": {<column: number>, ...},
  "allowed_values": {<column: [list of allowed/unique values]>, ...},
  "value_range": {<column: [min, max]>, ...},
  "correlation": [[<numbers>], [<numbers>], ...]
}
Only include a stat for a column if it is numeric and applicable.
If the audio does not give enough info for a field, use an empty
object {} or empty list [] for that field, but ALWAYS include ALL keys.
Return raw JSON only.
"""
def detect_audio_format(audio_bytes: bytes) -> str:
    if audio_bytes[:4] == b"RIFF":
        return "wav"
    if audio_bytes[:3] == b"ID3" or audio_bytes[:2] == b"\xff\xfb":
        return "mp3"
    return "wav"
def analyze_audio(audio_b64: str, audio_bytes: bytes) -> dict:
    fmt = detect_audio_format(audio_bytes)
    completion = client.chat.completions.create(
        model="gpt-4o-audio-preview",
        modalities=["text"],
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this audio and return the JSON."},
                    {"type": "input_audio", "input_audio": {"data": audio_b64, "format": fmt}}
                ]
            }
        ],
        temperature=0
    )
    raw = completion.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    try:
        data = json.loads(raw)
    except Exception:
        return dict(EMPTY_RESULT)
    result = dict(EMPTY_RESULT)
    for key in SCHEMA_KEYS:
        if key in data:
            result[key] = data[key]
    return result
@app.post("/")
async def analyze(request: Request):
    try:
        body = await request.json()
        audio_b64 = body.get("audio_base64", "")
        audio_bytes = base64.b64decode(audio_b64)
        result = analyze_audio(audio_b64, audio_bytes)
        return JSONResponse(content=result)
    except Exception as e:
        fallback = dict(EMPTY_RESULT)
        fallback["error"] = str(e)
        return JSONResponse(content=fallback, status_code=200)
@app.get("/")
async def health():
    return {"status": "ok"}