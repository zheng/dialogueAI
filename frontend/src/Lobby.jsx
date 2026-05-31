import { useEffect, useState } from 'react'
import { useVAD } from './useVAD.js'
import { useWebSocket } from './useWebSocket.js'

// The voice search front door. You say who you want; the lobby runs the
// utterance through the backend's ExpertSearch + Gemini relevance and either
// routes you into a chat or asks again. Finding an expert is a search problem,
// so this is a search box (voice-first), not a menu -- the chips below are a
// fallback shortcut, not the primary path.
export default function Lobby({ onMatch }) {
  const [status, setStatus] = useState('')
  const [thinking, setThinking] = useState(false)
  const [available, setAvailable] = useState([])

  const { sendVoice } = useWebSocket('/ws/lobby', {
    onJson: (msg) => {
      setThinking(false)
      if (msg.type === 'route') onMatch(msg.expert)
      else if (msg.type === 'clarify') setStatus(msg.text)
    },
  })

  const { ready, error } = useVAD({
    enabled: !thinking,
    onUtterance: (wav) => {
      setThinking(true)
      setStatus('')
      sendVoice(wav)
    },
  })

  useEffect(() => {
    fetch('/api/experts')
      .then((r) => r.json())
      .then(setAvailable)
      .catch(() => {})
  }, [])

  return (
    <div className="screen">
      <div
        className="orb"
        data-state={!ready ? 'idle' : thinking ? 'processing' : 'listening'}
      />
      <div className="prompt">Who would you like to talk to?</div>
      <div className="sub">Just say a name or a topic out loud.</div>
      <div className="status">
        {error
          ? `Mic/VAD failed to start: ${error.message || error}`
          : !ready
            ? 'Loading microphone…'
            : thinking
              ? 'Finding them…'
              : status || 'Listening…'}
      </div>

      {available.length > 0 && (
        <div className="hint">
          Available now:
          <div>
            {available.map((e) => (
              <button key={e.id} className="chip" onClick={() => onMatch(e.id)}>
                {e.name} — {e.tagline}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
