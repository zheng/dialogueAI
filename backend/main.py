"""DialogueAI backend: a voice lobby (search) + voice chat (per expert).

Two WebSocket surfaces share the same audio plumbing:
  - /ws/lobby        say who you want -> ExpertSearch + Gemini relevance -> route
  - /ws?expert=<id>  voice chat with the matched expert (Gemini -> ElevenLabs)

Gemini 3.5 Flash takes audio natively (no separate speech-to-text) but only
emits text, so spoken replies are synthesized by ElevenLabs in the expert's
cloned voice.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs.client import ElevenLabs
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from google import genai
from google.genai import types
from pydantic import BaseModel

from backend.experts import Expert, ExpertSearch, LocalExpertSearch

load_dotenv()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("dialogueai")

GEMINI_MODEL = "gemini-3.5-flash"
ELEVENLABS_MODEL = "eleven_multilingual_v2"
EXPERTS_ROOT = Path(__file__).resolve().parent.parent / "experts"

SYSTEM_PROMPT_TEMPLATE = """\
You are {name}, {persona}. You're having a casual, spoken conversation with \
someone who wants to learn from you. Speak in the first person, as yourself. \
Draw your knowledge, opinions, and speaking style from the content corpus \
below. Be conversational, warm, and concise (2-3 sentences) -- this is a live \
voice chat, not an essay.

YOUR CONTENT CORPUS:
{transcript}
"""

LOBBY_PROMPT = """\
You are the front desk of a voice app where people ask to talk to an AI expert.
The user just spoke a request. Decide which available expert they want, based
on the topic or name they mention. Available experts:

{roster}

Return the id of the best match. If the request clearly matches none of them
(different topic, different person), return "none".
"""


class Route(BaseModel):
    """Structured result of the lobby's relevance judgment."""

    expert_id: str  # a known expert id, or "none"


# --- shared clients / search, constructed once at startup -------------------

app = FastAPI()
search: ExpertSearch = LocalExpertSearch(EXPERTS_ROOT)
gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
elevenlabs = ElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"))


def synthesize(text: str, voice_id: str) -> bytes | None:
    """Speak `text` in the given cloned voice. None if no voice is configured."""
    if not voice_id:
        log.warning("No elevenlabs_voice_id set; returning text only.")
        return None
    chunks = elevenlabs.text_to_speech.convert(
        voice_id=voice_id,
        model_id=ELEVENLABS_MODEL,
        text=text,
        output_format="mp3_44100_128",
    )
    return b"".join(chunks)


async def receive_voice_input(ws: WebSocket) -> bytes | None:
    """Expect a {"type":"voice_input"} control frame then a binary WAV frame.

    Returns the WAV bytes, or None on disconnect / unexpected frame.
    """
    control = await ws.receive()
    if control.get("type") == "websocket.disconnect":
        return None
    text = control.get("text")
    if not text or json.loads(text).get("type") != "voice_input":
        return None
    audio = await ws.receive()
    if audio.get("type") == "websocket.disconnect":
        return None
    return audio.get("bytes")


@app.get("/api/experts")
def list_available_experts():
    """Cheap roster for the lobby's 'available now' fallback hint."""
    return [m.__dict__ for m in search.search("")]


@app.websocket("/ws/lobby")
async def lobby(ws: WebSocket):
    """Voice search: an utterance -> the matched expert id (or a clarify)."""
    await ws.accept()
    try:
        while True:
            wav = await receive_voice_input(ws)
            if wav is None:
                break

            candidates = search.search("")  # MVP: query text unused; Gemini judges
            roster = "\n".join(f"- {m.id}: {m.name} -- {m.tagline}" for m in candidates)
            valid_ids = {m.id for m in candidates}

            result = gemini.models.generate_content(
                model=GEMINI_MODEL,
                contents=[types.Part.from_bytes(data=wav, mime_type="audio/wav")],
                config=types.GenerateContentConfig(
                    system_instruction=LOBBY_PROMPT.format(roster=roster),
                    response_mime_type="application/json",
                    response_schema=Route,
                ),
            )
            route: Route = result.parsed

            if route.expert_id in valid_ids:
                await ws.send_text(
                    json.dumps({"type": "route", "expert": route.expert_id})
                )
            else:
                names = " or ".join(m.name for m in candidates) or "no one yet"
                await ws.send_text(
                    json.dumps(
                        {
                            "type": "clarify",
                            "text": f"The only expert I have right now is {names}"
                            " -- want to talk to them?",
                        }
                    )
                )
    except WebSocketDisconnect:
        pass


@app.websocket("/ws")
async def chat(ws: WebSocket):
    """Voice chat bound to one expert; per-connection conversation history."""
    expert_id = ws.query_params.get("expert", "")
    try:
        expert: Expert = search.get(expert_id)
    except KeyError:
        await ws.close(code=4004)
        return

    await ws.accept()
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
        name=expert.name, persona=expert.persona, transcript=expert.transcript
    )
    history: list[types.Content] = []

    try:
        while True:
            wav = await receive_voice_input(ws)
            if wav is None:
                break

            history.append(
                types.Content(
                    role="user",
                    parts=[types.Part.from_bytes(data=wav, mime_type="audio/wav")],
                )
            )
            result = gemini.models.generate_content(
                model=GEMINI_MODEL,
                contents=history,
                config=types.GenerateContentConfig(system_instruction=system_prompt),
            )
            reply = (result.text or "").strip()
            history.append(
                types.Content(role="model", parts=[types.Part.from_text(text=reply)])
            )

            await ws.send_text(json.dumps({"type": "response", "text": reply}))
            mp3 = synthesize(reply, expert.elevenlabs_voice_id)
            if mp3 is not None:
                await ws.send_bytes(mp3)
    except WebSocketDisconnect:
        pass
