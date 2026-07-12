import os
import json
import base64
import tempfile
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

SYSTEM_PROMPT = """You are a strict data-extraction engine.
You will be given a transcript of spoken audio describing a small dataset
(column names, values, and/or statistics being read aloud).

Reconstruct the dataset as best as possible from the transcript, then
compute and return ONLY a single JSON object with EXACTLY these keys,
no extra keys, no markdown, no explanation:

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
  "correlation": [[<numbers>], [<numbers>], ...]  (matrix, numeric columns only, or [] if not applicable)
}

Only include a stat for a column if it is numeric and applicable.
If the transcript does not give enough info for a field, use an empty
object {} or empty list [] for that field, but ALWAYS include ALL keys.
Return raw JSON only — no ```json fences, no commentary.
"""

def transcribe_audio(audio_bytes: bytes) -> str:
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f
            )
        return result.text
    finally:
        os.remove(tmp_path)

def extract_stats(transcript: str) -> dict:
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Transcript:\n{transcript}"}
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
        transcript = transcribe_audio(audio_bytes)
        result = extract_stats(transcript)
        return JSONResponse(content=result)
    except Exception as e:
        fallback = dict(EMPTY_RESULT)
        fallback["error"] = str(e)
        return JSONResponse(content=fallback, status_code=200)

@app.get("/")
async def health():
    return {"status": "ok"}