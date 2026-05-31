"""DialogueAI backend: a voice lobby (search) + voice chat (per expert).

Two WebSocket surfaces share the same audio plumbing:
  - /ws/lobby        say who you want -> ExpertSearch + Gemini relevance -> route
  - /ws?expert=<id>  voice chat with the matched expert

The chat surface (see PLAN2.md) is a single, streaming, interruptible turn model
that serves two modes:
  - CONVERSATION: Gemini streams a SHORT, grounded reply (sentence-chunked to
    ElevenLabs so audio starts almost immediately).
  - LECTURE: the chosen transcript is played back VERBATIM from a cursor.
Gemini drives mode transitions via tool calls (start_lecture / jump /
resume_lecture / end_lecture); the user can also barge in (interrupt) or click
to resume/end. Groundedness: the whole corpus rides as a stable first content
each turn, hitting Gemini's implicit cache.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from elevenlabs.client import AsyncElevenLabs
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

# Caps conversation answer length. Thinking is currently disabled (NO_THINKING;
# see the "re-adding thinking" plan in PLAN2.md), so this budget is entirely for
# the visible reply: ~160 tokens lets a teaching answer breathe (3-4 sentences)
# while the prompt keeps chit-chat to one or two. NOTE: if thinking is re-enabled
# this must grow to fit thinking + answer -- thinking spends against this cap.
CONVERSATION_MAX_TOKENS = 160
NO_THINKING = types.ThinkingConfig(thinking_budget=0)
# Roughly how many characters of transcript to read per lecture segment.
LECTURE_SEGMENT_CHARS = 420

CHAT_SYSTEM_TEMPLATE = """\
You are {name}, {persona}. You're in a LIVE VOICE conversation with someone who
wants to learn from you. Speak in the first person, as yourself.

GROUNDING: Everything you say must be grounded in YOUR CONTENT CORPUS (provided
as the first message). If the corpus doesn't cover something, say you didn't
cover that rather than guessing. Never invent facts, numbers, or citations.
NEVER recite or dump the transcript as text in conversation.

STYLE: This is speech, not an essay -- be conversational and stop once you've
answered. For a quick factual question, one or two sentences. For a "how do I
build X" / "how does X work" question, give the first concrete step and the
intuition -- a few sentences (3-4), then stop. No preamble, no lists, no recap,
no "in summary". You are being SPOKEN aloud: never write code blocks, LaTeX, or
symbols -- describe code and math in plain spoken words (say "x squared", not
"x^2"; describe what a function does rather than dictating it).

LECTURE TOOLS: By DEFAULT you answer in conversation -- in your own words. Only
call `start_lecture` when the user EXPLICITLY asks to play/watch the actual
lecture or be walked through the whole thing end to end (e.g. "play the micrograd
lecture", "walk me through the full thing", "just show me the video"). A plain
"yes", "teach me", "go on", or "how do I build X" is NOT a lecture request --
keep talking in conversation and explain it yourself. When you do start one, pass
the matching lecture_id and a short verbatim `anchor_quote` copied exactly from
the corpus at the point to begin. Use `jump` to move within/across lectures the
same way, `resume_lecture` to continue a paused lecture, and `end_lecture` to
stop. Available lectures (use these exact ids):
{lecture_list}
"""

LOBBY_PROMPT = """\
You are the front desk of a voice app where people ask to talk to an AI expert.
The user just spoke a request. Decide which available expert they want, based
on the topic or name they mention. Available experts:

{roster}

Return the id of the best match. If the request clearly matches none of them
(different topic, different person), return "none".
"""

# Tool declarations Gemini uses to drive the mode / lecture state machine.
LECTURE_TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="start_lecture",
                description=(
                    "Switch to lecture mode and play a lecture verbatim from a "
                    "chosen spot. ONLY use when the user EXPLICITLY asks to play "
                    "the actual lecture or be walked through the whole thing end "
                    "to end. Do NOT use for a plain 'yes'/'teach me'/'how do I "
                    "build X' -- answer those yourself in conversation instead."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "lecture_id": types.Schema(
                            type="STRING", description="one of the known lecture ids"
                        ),
                        "anchor_quote": types.Schema(
                            type="STRING",
                            description=(
                                "a short exact verbatim phrase copied from the "
                                "corpus at the point to begin"
                            ),
                        ),
                    },
                    required=["lecture_id", "anchor_quote"],
                ),
            ),
            types.FunctionDeclaration(
                name="jump",
                description=(
                    "While lecturing, move the cursor to a different spot or "
                    "lecture. Same arguments as start_lecture."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "lecture_id": types.Schema(type="STRING"),
                        "anchor_quote": types.Schema(type="STRING"),
                    },
                    required=["lecture_id", "anchor_quote"],
                ),
            ),
            types.FunctionDeclaration(
                name="resume_lecture",
                description="Resume the paused lecture from where it left off.",
                parameters=types.Schema(type="OBJECT", properties={}),
            ),
            types.FunctionDeclaration(
                name="end_lecture",
                description="Stop lecturing and return to conversation.",
                parameters=types.Schema(type="OBJECT", properties={}),
            ),
        ]
    )
]


class Route(BaseModel):
    """Structured result of the lobby's relevance judgment."""

    expert_id: str  # a known expert id, or "none"


# --- shared clients / search, constructed once at startup -------------------

app = FastAPI()
search: ExpertSearch = LocalExpertSearch(EXPERTS_ROOT)
gemini = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
elevenlabs = AsyncElevenLabs(api_key=os.environ.get("ELEVENLABS_API_KEY"))


# --- text / transcript helpers ----------------------------------------------

_SENTENCE_END = re.compile(r"[.!?]+(\s|$)")


def split_leading_sentences(buffer: str, force: bool) -> tuple[list[str], str]:
    """Pull complete sentences off the front of `buffer`.

    Returns (sentences, remainder). With force=True the remainder is flushed too
    (used at the end of a stream).
    """
    sentences = []
    while True:
        m = _SENTENCE_END.search(buffer)
        if not m:
            break
        cut = m.end()
        sentence = buffer[:cut].strip()
        if sentence:
            sentences.append(sentence)
        buffer = buffer[cut:]
    if force and buffer.strip():
        sentences.append(buffer.strip())
        buffer = ""
    return sentences, buffer


def next_segment(text: str, cursor: int) -> tuple[str | None, int]:
    """Return the next verbatim lecture segment from `cursor`, and the new cursor.

    Groups whole sentences up to ~LECTURE_SEGMENT_CHARS so playback pauses fall
    on natural boundaries. Returns (None, cursor) when the lecture is exhausted.
    """
    remaining = text[cursor:]
    if not remaining.strip():
        return None, cursor
    consumed = 0
    for m in re.finditer(r"\s*\S[^.!?]*[.!?]+", remaining, re.S):
        consumed = m.end()
        if consumed >= LECTURE_SEGMENT_CHARS:
            break
    if consumed == 0:  # no sentence punctuation left; take the rest
        consumed = len(remaining)
    return remaining[:consumed].strip(), cursor + consumed


def find_offset(text: str, quote: str) -> int:
    """Locate a (verbatim) anchor quote in a lecture; best-effort fallbacks."""
    if not quote:
        return 0
    i = text.find(quote)
    if i != -1:
        return i
    # Fall back to the first several words, case-insensitively.
    probe = " ".join(quote.split()[:6])
    if probe:
        j = text.lower().find(probe.lower())
        if j != -1:
            return j
    return 0


# --- session state ----------------------------------------------------------


class Session:
    """Per-connection chat state for one expert."""

    def __init__(self, expert: Expert):
        self.expert = expert
        self.lectures = {lec.id: lec for lec in expert.lectures}
        self.history: list[types.Content] = []
        # A single stable first content -> Gemini implicit cache hits on it.
        self.corpus_part = types.Content(
            role="user",
            parts=[types.Part.from_text(text="YOUR CONTENT CORPUS:\n" + expert.corpus)],
        )
        lecture_list = "\n".join(
            f"- {lec.id}: {lec.title}" for lec in expert.lectures
        )
        self.system_instruction = CHAT_SYSTEM_TEMPLATE.format(
            name=expert.name, persona=expert.persona, lecture_list=lecture_list
        )
        self.mode = "conversation"
        self.lecture_id: str | None = None  # set while a lecture is active or held
        self.cursor = 0  # char offset into the active lecture

    @property
    def voice_id(self) -> str:
        return self.expert.elevenlabs_voice_id


# --- WebSocket I/O helpers --------------------------------------------------


async def send_json(ws: WebSocket, obj: dict) -> None:
    await ws.send_text(json.dumps(obj))


async def send_mode(ws: WebSocket, session: Session, paused: bool) -> None:
    """Tell the client the authoritative mode/lecture state for its indicator."""
    await send_json(
        ws,
        {
            "type": "mode",
            "mode": session.mode,
            "lecture": session.lecture_id,
            "paused": paused,
        },
    )


async def tts(text: str, voice_id: str) -> bytes | None:
    """Synthesize one chunk of speech in the cloned voice (None if no voice)."""
    if not voice_id:
        return None
    chunks = []
    async for c in elevenlabs.text_to_speech.convert(
        voice_id=voice_id,
        model_id=ELEVENLABS_MODEL,
        text=text,
        output_format="mp3_44100_128",
    ):
        chunks.append(c)
    return b"".join(chunks)


async def speak(ws: WebSocket, session: Session, text: str) -> None:
    """Send a text frame then its spoken audio frame."""
    await send_json(ws, {"type": "text", "text": text})
    audio = await tts(text, session.voice_id)
    if audio is not None:
        await ws.send_bytes(audio)


# --- turn runners (each runs as a cancellable asyncio task) ------------------


async def run_lecture(ws: WebSocket, session: Session) -> None:
    """Stream the active lecture verbatim from the cursor until done/interrupted.

    Cancellation (barge-in) raises inside an await; the cursor already points at
    the start of the segment currently playing, so a later resume replays it.
    """
    session.mode = "lecture"
    lec = session.lectures[session.lecture_id]
    total = len(lec.text)

    async def send_progress() -> None:
        await send_json(
            ws,
            {
                "type": "cursor",
                "lecture": session.lecture_id,
                "pos": session.cursor,
                "total": total,
            },
        )

    await send_mode(ws, session, paused=False)
    await send_json(ws, {"type": "turn_start", "mode": "lecture"})
    await send_progress()  # so the bar appears immediately at the right spot
    while True:
        segment, new_cursor = next_segment(lec.text, session.cursor)
        if segment is None:
            session.mode = "conversation"
            session.lecture_id = None
            session.cursor = 0
            await speak(ws, session, "That's the end of that lecture.")
            await send_mode(ws, session, paused=False)
            await send_json(ws, {"type": "turn_end"})
            return
        await speak(ws, session, segment)
        session.cursor = new_cursor
        await send_progress()


def resolve_lecture_id(session: Session, raw: str | None) -> str | None:
    """Map a model-provided lecture id to a real one, tolerating near-misses
    like 'micrograd' for '1_micrograd' or a title instead of an id."""
    if not raw:
        return None
    if raw in session.lectures:
        return raw
    needle = raw.lower()
    for lid, lec in session.lectures.items():
        hay = f"{lid} {lec.title}".lower()
        if needle in hay or lec.title.lower() in needle:
            return lid
    return None


async def dispatch_tool(ws: WebSocket, session: Session, name: str, args: dict) -> bool:
    """Act on a Gemini tool call. Returns True if it took over the turn."""
    log.info("tool call: %s %s", name, args)
    if name in ("start_lecture", "jump"):
        lecture_id = resolve_lecture_id(session, args.get("lecture_id"))
        if lecture_id is not None:
            session.lecture_id = lecture_id
            session.cursor = find_offset(
                session.lectures[lecture_id].text, args.get("anchor_quote", "")
            )
            log.info("entering lecture %s at cursor %d", lecture_id, session.cursor)
            session.history.append(
                types.Content(
                    role="model",
                    parts=[types.Part.from_text(text=f"(entering lecture: {lecture_id})")],
                )
            )
            await run_lecture(ws, session)
            return True
        return False
    if name == "resume_lecture":
        if session.lecture_id is not None:
            await run_lecture(ws, session)
            return True
        return False
    if name == "end_lecture":
        session.mode = "conversation"
        session.lecture_id = None
        session.cursor = 0
        await send_mode(ws, session, paused=False)
        return False
    return False


async def run_conversation_turn(ws: WebSocket, session: Session, wav: bytes) -> None:
    """Stream a short, grounded spoken reply; act on any tool call at the end."""
    session.history.append(
        types.Content(
            role="user", parts=[types.Part.from_bytes(data=wav, mime_type="audio/wav")]
        )
    )
    await send_json(ws, {"type": "turn_start", "mode": "conversation"})

    buffer = ""
    full_text = ""
    function_calls = []
    stream = await gemini.aio.models.generate_content_stream(
        model=GEMINI_MODEL,
        contents=[session.corpus_part, *session.history],
        config=types.GenerateContentConfig(
            system_instruction=session.system_instruction,
            tools=LECTURE_TOOLS,
            max_output_tokens=CONVERSATION_MAX_TOKENS,
            thinking_config=NO_THINKING,
        ),
    )
    async for chunk in stream:
        if chunk.text:
            buffer += chunk.text
            full_text += chunk.text
            sentences, buffer = split_leading_sentences(buffer, force=False)
            for sentence in sentences:
                await speak(ws, session, sentence)
        if chunk.function_calls:
            function_calls.extend(chunk.function_calls)

    sentences, buffer = split_leading_sentences(buffer, force=True)
    for sentence in sentences:
        await speak(ws, session, sentence)

    if full_text.strip():
        session.history.append(
            types.Content(role="model", parts=[types.Part.from_text(text=full_text.strip())])
        )

    for fc in function_calls:
        if await dispatch_tool(ws, session, fc.name, dict(fc.args or {})):
            return  # the tool (e.g. a lecture) took over and ends the turn itself

    await send_json(ws, {"type": "turn_end"})


# --- HTTP + WebSocket endpoints ---------------------------------------------


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
            control = await ws.receive()
            if control.get("type") == "websocket.disconnect":
                break
            text = control.get("text")
            if not text or json.loads(text).get("type") != "voice_input":
                continue
            audio = await ws.receive()
            wav = audio.get("bytes")
            if wav is None:
                continue

            candidates = search.search("")
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
                await send_json(ws, {"type": "route", "expert": route.expert_id})
            else:
                names = " or ".join(m.name for m in candidates) or "no one yet"
                await send_json(
                    ws,
                    {
                        "type": "clarify",
                        "text": f"The only expert I have right now is {names}"
                        " -- want to talk to them?",
                    },
                )
    except WebSocketDisconnect:
        pass


@app.websocket("/ws")
async def chat(ws: WebSocket):
    """Streaming, interruptible voice chat bound to one expert."""
    expert_id = ws.query_params.get("expert", "")
    try:
        expert: Expert = search.get(expert_id)
    except KeyError:
        await ws.close(code=4004)
        return

    await ws.accept()
    session = Session(expert)
    await send_mode(ws, session, paused=False)

    turn_task: asyncio.Task | None = None

    async def run_guarded(coro) -> None:
        try:
            await coro
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("turn failed")

    async def cancel_turn() -> None:
        nonlocal turn_task
        if turn_task and not turn_task.done():
            turn_task.cancel()
            try:
                await turn_task
            except (asyncio.CancelledError, Exception):
                pass
        turn_task = None

    try:
        while True:
            msg = await ws.receive()
            if msg.get("type") == "websocket.disconnect":
                break
            text = msg.get("text")
            if not text:
                continue
            data = json.loads(text)
            t = data.get("type")

            if t == "voice_input":
                audio = await ws.receive()
                wav = audio.get("bytes")
                if wav is None:
                    continue
                await cancel_turn()
                turn_task = asyncio.create_task(
                    run_guarded(run_conversation_turn(ws, session, wav))
                )
            elif t == "interrupt":
                await cancel_turn()
                if session.lecture_id is not None and session.mode == "lecture":
                    # Hold the lecture: keep cursor + lecture_id, drop to conversation.
                    session.mode = "conversation"
                    await send_mode(ws, session, paused=True)
            elif t == "resume":
                await cancel_turn()
                if session.lecture_id is not None:
                    turn_task = asyncio.create_task(run_guarded(run_lecture(ws, session)))
            elif t == "end_lecture":
                await cancel_turn()
                session.mode = "conversation"
                session.lecture_id = None
                session.cursor = 0
                await send_mode(ws, session, paused=False)
    except WebSocketDisconnect:
        pass
    finally:
        await cancel_turn()
