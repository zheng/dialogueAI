import { useCallback, useMemo, useRef, useState } from 'react'
import { useVAD } from './useVAD.js'
import { useWebSocket } from './useWebSocket.js'
import { useAudioQueue } from './useAudioQueue.js'

// Streaming, interruptible voice chat with the matched expert. The mic stays on
// the whole time so the user can barge in: starting to speak flushes playback
// and tells the server to interrupt. Mode is driven by Gemini tools; the only
// chrome is a clickable mode indicator (resume/stop a lecture).
export default function ChatView({ expertId, onBack }) {
  const [status, setStatus] = useState('listening') // listening|processing|speaking
  const [mode, setMode] = useState({ mode: 'conversation', lecture: null, paused: false })
  const [text, setText] = useState('')
  const [diag, setDiag] = useState({ heard: 0, frames: 0 })
  const [audioBlocked, setAudioBlocked] = useState(false)
  const turnEnded = useRef(true)

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
          setText((t) => (t ? `${t} ${msg.text}` : msg.text))
          break
        case 'mode':
          setMode({ mode: msg.mode, lecture: msg.lecture, paused: msg.paused })
          break
        case 'turn_end':
          turnEnded.current = true
          // If the turn produced no audio (e.g. a tool call that didn't start a
          // lecture), nothing else will move us off "Thinking" -- do it here.
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
    console.log('[chat] speech start → barge-in')
    flush()
    turnEnded.current = true
    send({ type: 'interrupt' })
    setStatus('listening')
  }, [flush, send])

  const handleUtterance = useCallback(
    (wav) => {
      console.log('[chat] utterance captured', wav.size, 'bytes → sending')
      setDiag((d) => ({ ...d, heard: d.heard + 1 }))
      setStatus('processing')
      sendVoice(wav)
    },
    [sendVoice],
  )

  const { ready, error } = useVAD({
    enabled: true, // always on, so barge-in works during playback
    onSpeechStart: handleSpeechStart,
    onUtterance: handleUtterance,
  })

  // Unlock audio after a user gesture (needed when chat was entered by voice).
  const unlockAudio = () => {
    new Audio(
      'data:audio/mp3;base64,//uQxAAAAAAAAAAAAAAAAAAAAAAAWGluZwAAAA8AAAACAAACcQCA',
    )
      .play()
      .catch(() => {})
    setAudioBlocked(false)
  }

  // The one manual control: tap the indicator to resume a held lecture or stop
  // a playing one. (Jump is never a UI action -- it's a Gemini-only tool.)
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
      : { listening: 'Listening…', processing: 'Thinking…', speaking: 'Speaking…' }[status]

  return (
    <div className="screen">
      <button className="back" onClick={onBack}>
        ← back to lobby
      </button>

      <button
        className={`mode-pill ${lecturePlaying ? 'lecture' : lectureHeld ? 'paused' : ''}`}
        onClick={onIndicatorClick}
        disabled={!lecturePlaying && !lectureHeld}
        title="Tap to resume or stop a lecture"
      >
        {indicatorLabel}
      </button>

      <div className="orb" data-state={ready ? status : 'idle'} />
      <div className="status">{statusLabel}</div>

      {audioBlocked && (
        <button className="chip" onClick={unlockAudio}>
          🔊 Tap to enable sound
        </button>
      )}

      {text && (
        <div className="exchange">
          <div className="bubble expert">
            <div className="who">{expertId}</div>
            {text}
          </div>
        </div>
      )}

      <div className="diag">
        ws {connected ? '✓' : '✗'} · mic {error ? 'error' : ready ? '✓' : '…'} · heard{' '}
        {diag.heard} · audio {diag.frames}
      </div>
    </div>
  )
}
