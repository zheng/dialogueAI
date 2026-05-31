# DialogueAI — Voice Chat with AI Experts

## Context
A conversation-first web app where you have a voice chat with an AI version of a real-world expert. You say **who** you want to talk to — "let me talk to Karpathy", "I have a health question" — and the app connects you. Then you have a natural spoken conversation. Each expert's knowledge, opinions, and voice are derived from their publicly available content (transcript as context, cloned voice for TTS).

The entire experience is voice-first: even **picking** the expert is spoken.

The system is designed so experts are **data, not code**: nothing is special-cased to one person. Karpathy is simply the first expert we ship while we tune quality. Future versions support talking to two experts at once, or a panel.

**MVP scope:** single-expert voice chat, with a picker up front. Karpathy is the only populated expert, but the architecture is general. **Out of scope for now:** data ingestion — transcripts will be supplied by hand (dropped into the expert's folder); no scraping/download pipeline yet.

## Architecture
```
Browser (mic + speaker)
  1. LOBBY: say who you want → search          ──┐ /ws/lobby
  2. CHAT:  voice chat with the matched expert ──┘ /ws?expert=<id>
     ↕ WebSocket
FastAPI backend
     ↓                    ↓                    ↓
ExpertSearch +            Gemini (chat)        ElevenLabs TTS
Gemini (query→match)      audio → text,        text → that expert's
audio → ranked expert     persona + corpus      cloned voice
                          as context
```

Two voice surfaces share the same audio plumbing (VAD + WebSocket + Gemini):
- **Lobby (search):** audio → query → `ExpertSearch` returns ranked matches (Gemini judges relevance) → frontend transitions into the top match's chat, or asks again if nothing matched.
- **Chat:** audio → Gemini answers as the expert → ElevenLabs speaks it back.

## Finding an expert is a search problem
You don't pick from an enumerated list — you **search**. When you say "let me talk to Karpathy" or "I have a health question", that utterance is a query, and the system returns the best-matching expert(s). This is the right model for where it's going: an open, growing space of experts (and eventually ones minted on demand), not a fixed menu. The lobby is just the front door to that search.

All expert access goes through one search-shaped interface — conceptually:

```python
class ExpertSearch(Protocol):
    def search(query: str) -> list[ExpertMatch]: ...   # ranked candidates (may be empty)
    def get(id: str) -> Expert: ...                     # load the full payload for a chosen match
```

This is deliberately **not** a registry or a `list()`-able store — there's no central table to enumerate, because in the real version you can't enumerate "everyone you might talk to." You query, you get ranked matches, you connect to one.

**MVP:** ship one implementation whose index contains exactly **one** expert, Karpathy. A query either matches him (good enough relevance) or returns no match ("the only expert I have right now is Andrej Karpathy"). The index is backed by local files for now — an implementation detail of this search, not a system-wide assumption. As more experts are added (or generated on the fly), only the `ExpertSearch` implementation grows; nothing else changes.

```
experts/
└── karpathy/
    ├── config.json
    └── transcripts/
        └── <video-name>.txt      # one or more; supplied by hand
```

`config.json` shape:
```json
{
  "id": "karpathy",
  "name": "Andrej Karpathy",
  "tagline": "Neural networks, from scratch",
  "persona": "a warm, intuition-first AI educator who builds things from scratch",
  "elevenlabs_voice_id": "<set after cloning>"
}
```

## Project Structure
```
dialogueai/
├── .gitignore
├── .env                          # API keys (not committed)
├── pyproject.toml                # Python deps
├── experts/
│   └── karpathy/
│       ├── config.json
│       └── transcripts/          # hand-supplied .txt files, concatenated on load
│           └── <video-name>.txt
├── backend/
│   ├── __init__.py
│   ├── experts.py                # ExpertSearch interface + local search impl
│   └── main.py                   # FastAPI app, WebSocket, Gemini + ElevenLabs
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx               # routes between lobby and chat
        ├── Lobby.jsx             # voice search: say who you want → match → connect
        ├── ChatView.jsx          # voice chat with the matched expert
        ├── useVAD.js             # Voice Activity Detection hook
        └── useWebSocket.js       # WebSocket hook
```

## Implementation Steps

### Step 1: Scaffolding
- `.gitignore`, `pyproject.toml`, `package.json`, `vite.config.js`, `index.html`, `main.jsx`
- Python deps: `fastapi`, `uvicorn[standard]`, `google-genai`, `elevenlabs`, `python-dotenv`
- Node deps: `react`, `react-dom`, `@vitejs/plugin-react`, `vite`, `@ricky0123/vad-web`

### Step 2: Backend (`backend/main.py` + `backend/experts.py`)
- **On startup:** init Gemini + ElevenLabs clients; construct the `ExpertSearch` implementation (MVP: local index over `experts/`, currently just Karpathy). The rest of the code depends on the `ExpertSearch` interface, not on files.
- **WebSocket `/ws/lobby`** (voice search): receives `{"type":"voice_input"}` + binary WAV, runs the utterance through `ExpertSearch.search(...)`. For the MVP single-entry index, Gemini judges whether the request matches Karpathy (and returns the matched id via structured output). Responds:
  - matched → `{"type":"route","expert":"karpathy"}`
  - no match → `{"type":"clarify","text":"The only expert I have right now is Andrej Karpathy, on neural networks — want to talk to him?"}`

  The relevance judgment uses Gemini structured output (`response_schema`) returning a matched id or `"none"`, so there's no brittle string parsing. With one expert this is effectively a yes/no relevance check; as the index grows it becomes true ranked retrieval (semantic search over expert descriptions, optionally re-ranked by Gemini).
- **WebSocket `/ws?expert=<id>`** (chat): per-connection conversation history, bound to the matched expert (loaded via `ExpertSearch.get(id)`)
  - Receive: JSON `{"type": "voice_input"}` followed by binary WAV audio
  - Build a **generic, templated** system prompt (below) from the expert's config + transcript
  - Send to Gemini (native audio input) → text
  - Send text to ElevenLabs with that expert's `elevenlabs_voice_id` → MP3
  - Respond: JSON `{"type": "response", "text": "..."}` then binary MP3
- **Generic system prompt template** (no person hardcoded):
  ```
  You are {name}, {persona}. You're having a casual, spoken conversation
  with someone who wants to learn from you. Speak in the first person, as
  yourself. Draw your knowledge, opinions, and speaking style from the
  content corpus below. Be conversational, warm, and concise (2–3
  sentences) — this is a live voice chat, not an essay.

  YOUR CONTENT CORPUS:
  {transcript}
  ```

### Step 3: Frontend — useVAD.js
- `@ricky0123/vad-web` (Silero VAD in browser via ONNX)
- `onSpeechEnd`: encode Float32Array → WAV blob (inline helper, ~20 lines)
- Expose `start()`, `stop()`, `isListening`

### Step 4: Frontend — useWebSocket.js
- Connect to `ws://localhost:5173/ws?expert=<id>` (proxied to FastAPI)
- Handle text (JSON) and binary (audio) messages; auto-reconnect

### Step 5: Frontend — lobby + chat
- **App.jsx:** simple state machine — show `Lobby` until an expert is matched, then `ChatView` (with a back button to return to the lobby and search again)
- **Lobby.jsx (voice search front door):**
  - On mount: open `/ws/lobby` + start VAD immediately. A prompt invites the user to just say who they want ("Who would you like to talk to?")
  - VAD captures speech → send to `/ws/lobby` → on `route`, store the matched expert id and open the chat; on `clarify`, show (and optionally speak) the message and keep listening
  - Search-shaped UI, not a menu. A small "available now" hint (just Karpathy today) doubles as a clickable shortcut fallback, but the primary path is speaking your request
- **ChatView.jsx:** opens the WebSocket for the chosen expert, runs the voice loop. Centered status indicator + last exchange text. States: `idle → listening → processing → speaking → idle`
  1. View mounts → connect WebSocket (with expert id), init VAD
  2. VAD detects speech → capture audio
  3. Speech ends → send audio, show "processing"
  4. Receive response → play MP3, show response text
  5. Audio ends → back to idle, VAD resumes

### Step 6: Vite proxy config
```js
proxy: {
  '/ws':  { target: 'ws://localhost:8000', ws: true },
  '/api': { target: 'http://localhost:8000' },
}
```

## Pre-requisites (user must do before running)
1. **Gemini API key** from Google AI Studio → `.env: GEMINI_API_KEY`
2. **ElevenLabs API key** from elevenlabs.io → `.env: ELEVENLABS_API_KEY`
3. **Provide Karpathy's transcript(s):** drop one or more `.txt` files into `experts/karpathy/transcripts/` (supplied separately, by hand)
4. **Clone the voice:** create a cloned voice in ElevenLabs → put the resulting Voice ID into `experts/karpathy/config.json` as `elevenlabs_voice_id`

## Key Design Decisions
- **Finding experts is a search problem** — access goes through an `ExpertSearch` interface (`search(query) → ranked matches`, `get(id)`), not a registry or enumerable list. The real space of experts can't be enumerated, so we query it. The MVP index holds only Karpathy; growing it (more files, a DB, on-the-fly minting) only changes the `ExpertSearch` implementation.
- **Generic, templated system prompt** — persona + corpus are injected per expert; no hardcoded personality.
- **Voice-first throughout** — even finding an expert is spoken: the lobby is a voice search box, not a menu. The "available now" hint is a fallback, not the primary path.
- **Structured-output relevance** — the lobby's match step uses Gemini's `response_schema` to return a matched id (or `"none"`), reusing the same audio/WebSocket plumbing as chat. One expert today = a relevance check; many = ranked retrieval.
- **Ingestion is out of scope** — transcripts are hand-supplied for now; no scraping/download pipeline in this MVP.
- **VAD in browser** — natural conversation, no push-to-talk.
- **WAV upload / MP3 download** — simplest formats; Gemini accepts WAV natively, MP3 is small and universally playable.

## How to Run
```bash
# Terminal 1
uv run uvicorn backend.main:app --reload --port 8000

# Terminal 2
cd frontend && npm run dev
```
Open http://localhost:5173 → say who you want → talk.

## How to Verify
1. Allow microphone, then **say** "I want to talk to Karpathy" → lobby search matches him and opens his chat (proves `/ws/lobby` + `ExpertSearch` relevance)
2. Say something off-topic like "I have a health question" → lobby replies that only Karpathy is available (proves the no-match path)
3. (Fallback) clicking the "available now" Karpathy hint also opens the chat
4. In chat, say "Hey, can you explain what a derivative is?"
5. Hear Karpathy's cloned voice answer in the context of his lecture
6. Backend terminal shows Gemini search + chat + ElevenLabs calls; browser console shows WebSocket traffic

## Roadmap (post-MVP)
- **Grow the search index:** add more experts (more local folders, then a real index / vector search over expert descriptions) so the lobby returns genuinely ranked matches — only the `ExpertSearch` implementation changes.
- **Ingestion pipeline:** a separate component that pulls a creator's content + clones their voice and adds them to the index — slots in behind the `ExpertSearch` seam with no changes to backend/frontend.
- **On-the-fly experts:** search for someone not yet in the index → ingestion runs → they become searchable and you connect.
- **Multi-expert conversations:** talk to two experts at once, or moderate a panel — the WebSocket/session layer generalizes from one expert to several.
- **Video mode:** optionally bring back the watch-and-interrupt experience as an alternate mode on top of the same backend.
