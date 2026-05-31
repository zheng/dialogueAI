import { useCallback, useMemo, useRef, useState } from 'react'
import { useVAD } from './useVAD.js'
import { useWebSocket } from './useWebSocket.js'
import { useAudioQueue } from './useAudioQueue.js'

// Rough spoken pace (chars/sec) for estimating lecture time from the cursor.
const CHARS_PER_SEC = 14

function fmtTime(seconds) {
  const s = Math.max(0, Math.round(seconds))
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`
}

// Streaming, interruptible voice chat with the matched expert. The mic stays on
// (unless muted) so the user can barge in: starting to speak flushes playback
// and tells the server to interrupt. Mode is driven by Gemini tools; chrome is
// minimal -- a clickable mode indicator, a mute button, and (in lecture) a
// progress bar.
export default function ChatView({ expertId, onBack }) {
  const [status, setStatus] = useState('listening') // listening|processing|speaking
  const [mode, setMode] = useState({ mode: 'conversation', lecture: null, paused: false })
  const [text, setText] = useState('')
  const [progress, setProgress] = useState({ pos: 0, total: 0 })
  const [muted, setMuted] = useState(false)
  const [diag, setDiag] = useState({ heard: 0, frames: 0 })
  const [audioBlocked, setAudioBlocked] = useState(false)
  const turnEnded = useRef(true)
  const isLecture = useRef(false)

  const path = useMemo(() => `/ws?expert=${encodeURIComponent(expertId)}`, [expertId])

  const { enqueue, flush, isPlaying } = useAudioQueue({
    onStart: () => setStatus('speaking'),
    onDrain: () => {
      if (turnEnded.current) setStatus('listening')
    },
    onBlocked: () => setAudioBlocked(true),
  })

  const { connected, send, sendVoice } = useWebSocket(path, {
    onJson: (msg) => {
      switch (msg.type) {
        case 'turn_start':
          turnEnded.current = false
          setText('')
          setStatus('processing')
          break
        case 'text':
          // In lecture mode show only the current segment (clamped to 2 lines);
          // in conversation, accumulate the (short) reply.
          if (isLecture.current) setText(msg.text)
          else setText((t) => (t ? `${t} ${msg.text}` : msg.text))
          break
        case 'cursor':
          setProgress({ pos: msg.pos, total: msg.total ?? 0 })
          break
        case 'mode':
          isLecture.current = msg.mode === 'lecture'
          setMode({ mode: msg.mode, lecture: msg.lecture, paused: msg.paused })
          break
        case 'turn_end':
          turnEnded.current = true
          if (!isPlaying()) setStatus('listening')
          break
        default:
          break
      }
    },
    onBinary: (buf) => {
      setDiag((d) => ({ ...d, frames: d.frames + 1 }))
      enqueue(buf)
    },
  })

  // Barge-in: the instant the user speaks, stop playback and cancel the server turn.
  const handleSpeechStart = useCallback(() => {
    flush()
    turnEnded.current = true
    send({ type: 'interrupt' })
    setStatus('listening')
  }, [flush, send])

  const handleUtterance = useCallback(
    (wav) => {
      setDiag((d) => ({ ...d, heard: d.heard + 1 }))
      setStatus('processing')
      sendVoice(wav)
    },
    [sendVoice],
  )

  const { ready, error } = useVAD({
    enabled: !muted, // muting pauses the mic (no listening, no barge-in)
    onSpeechStart: handleSpeechStart,
    onUtterance: handleUtterance,
  })

  const unlockAudio = () => {
    new Audio(
      'data:audio/mp3;base64,//uQxAAAAAAAAAAAAAAAAAAAAAAAWGluZwAAAA8AAAACAAACcQCA',
    )
      .play()
      .catch(() => {})
    setAudioBlocked(false)
  }

  // Mode indicator: tap to resume a held lecture or stop a playing one.
  const lectureHeld = mode.lecture && mode.mode === 'conversation'
  const lecturePlaying = mode.mode === 'lecture'
  const onIndicatorClick = () => {
    if (lecturePlaying) send({ type: 'end_lecture' })
    else if (lectureHeld) send({ type: 'resume' })
  }

  const indicatorLabel = lecturePlaying
    ? `▶ Lecture · ${mode.lecture} — tap to stop`
    : lectureHeld
      ? `⏸ Lecture paused · ${mode.lecture} — tap to resume`
      : 'Conversation'

  const statusLabel = error
    ? `Mic/VAD failed to start: ${error.message || error}`
    : !ready
      ? 'Loading microphone…'
      : muted && status !== 'speaking'
        ? 'Muted — mic off'
        : { listening: 'Listening…', processing: 'Thinking…', speaking: 'Speaking…' }[status]

  const showProgress = (lecturePlaying || lectureHeld) && progress.total > 0
  const pct = showProgress ? Math.min(100, (100 * progress.pos) / progress.total) : 0

  return (
    <div className="screen">
      <div className="topbar">
        <button className="back" onClick={onBack}>
          ← back to lobby
        </button>
        <button
          className={`mute ${muted ? 'on' : ''}`}
          onClick={() => setMuted((m) => !m)}
          title={muted ? 'Unmute microphone' : 'Mute microphone'}
        >
          {muted ? '🔇 Muted' : '🎙️ Mute'}
        </button>
      </div>

      <button
        className={`mode-pill ${lecturePlaying ? 'lecture' : lectureHeld ? 'paused' : ''}`}
        onClick={onIndicatorClick}
        disabled={!lecturePlaying && !lectureHeld}
        title="Tap to resume or stop a lecture"
      >
        {indicatorLabel}
      </button>

      <div className="orb" data-state={ready && !(muted && status !== 'speaking') ? status : 'idle'} />
      <div className="status">{statusLabel}</div>

      {showProgress && (
        <div className="lecture-progress">
          <div className="progress">
            <div className="progress-fill" style={{ width: `${pct}%` }} />
          </div>
          <div className="progress-time">
            {fmtTime(progress.pos / CHARS_PER_SEC)} / {fmtTime(progress.total / CHARS_PER_SEC)}
            {' · ~'}
            {fmtTime((progress.total - progress.pos) / CHARS_PER_SEC)} left
          </div>
        </div>
      )}

      {audioBlocked && (
        <button className="chip" onClick={unlockAudio}>
          🔊 Tap to enable sound
        </button>
      )}

      {text && (
        <div className="exchange">
          <div className={`bubble expert ${lecturePlaying ? 'clamp2' : ''}`}>
            <div className="who">{expertId}</div>
            {text}
          </div>
        </div>
      )}

      <div className="diag">
        ws {connected ? '✓' : '✗'} · mic {error ? 'error' : muted ? 'muted' : ready ? '✓' : '…'} ·
        heard {diag.heard} · audio {diag.frames}
      </div>
    </div>
  )
}
