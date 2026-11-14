"""FastAPI server. Browser streams 16 kHz int16 PCM over a WebSocket.

Protocol (client -> server):
    binary frame                      raw PCM16 mono @16 kHz
    {"type": "text", "text": "..."}   typed message (no mic needed)
    {"type": "playback_done"}         client finished playing bot audio
    {"type": "reset"}                 start a new call

Protocol (server -> client): JSON with type in
    vad | transcript | sources | bot_delta | audio | bot_done | metrics | interrupt | error
"""
from __future__ import annotations

import json
import logging

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .asr import build_asr
from .audio import pcm16_bytes_to_float
from .config import ROOT, settings
from .filler import FillerBank
from .llm import GeminiLLM
from .pipeline import CallSession
from .rag import BM25Retriever, load_kb
from .recorder import router as recorder_router
from .tts import CachedTTS, build_tts
from .vad import get_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("voicebot")

app = FastAPI(title="Bangla Voice Bot")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
app.include_router(recorder_router)  # /record: WER test-set recording page

STATE: dict = {}


@app.on_event("startup")
async def startup() -> None:
    get_model()  # warm up Silero so the first frame isn't slow
    STATE["retriever"] = BM25Retriever(load_kb(settings.kb_path))
    STATE["asr"] = build_asr()
    STATE["llm"] = GeminiLLM()
    STATE["tts"] = build_tts()
    if settings.tts_cache:
        STATE["tts"] = CachedTTS(STATE["tts"], ROOT / "cache" / "tts")
    STATE["fillers"] = FillerBank()
    if settings.filler_after_ms > 0:
        await STATE["fillers"].warm(STATE["tts"])
    log.info(
        "ready: asr=%s llm=%s tts=%s kb_chunks=%d",
        settings.asr_backend, settings.llm_model, settings.tts_backend,
        len(STATE["retriever"].chunks),
    )


@app.get("/")
def index() -> FileResponse:
    return FileResponse(ROOT / "static" / "index.html")


@app.get("/health")
def health() -> dict:
    return {"ok": True, "asr": settings.asr_backend, "llm": settings.llm_model, "tts": settings.tts_backend}


@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()

    async def send(msg: dict) -> None:
        await websocket.send_text(json.dumps(msg, ensure_ascii=False))

    def new_session() -> CallSession:
        return CallSession(send, STATE["asr"], STATE["llm"], STATE["tts"], STATE["retriever"], STATE["fillers"])

    session = new_session()
    try:
        while True:
            msg = await websocket.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            if msg.get("bytes"):
                await session.on_audio(pcm16_bytes_to_float(msg["bytes"]))
            elif msg.get("text"):
                data = json.loads(msg["text"])
                kind = data.get("type")
                if kind == "text" and data.get("text", "").strip():
                    await session.on_text(data["text"].strip())
                elif kind == "playback_done":
                    await session.on_playback_done()
                elif kind == "reset":
                    await session.close()
                    session = new_session()
    except WebSocketDisconnect:
        pass
    finally:
        await session.close()
