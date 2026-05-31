import { useMemo, useRef, useState } from 'react'
import { useVAD } from './useVAD.js'
import { useWebSocket } from './useWebSocket.js'

// Voice chat with the matched expert. Drives the loop:
//   idle -> listening -> processing -> speaking -> idle
// The mic is muted while the expert speaks so playback can't self-trigger VAD.
export default function ChatView({ expertId, onBack }) {
  const [state, setState] = useState('listening')
  const [reply, setReply] = useState('')
  const audioRef = useRef(null)

  const path = useMemo(() => `/ws?expert=${encodeURIComponent(expertId)}`, [expertId])

  const { sendVoice } = useWebSocket(path, {
    onJson: (msg) => {
      if (msg.type === 'response') setReply(msg.text)
    },
    onBinary: (buf) => {
      // Expert's cloned-voice MP3 arrives after the text frame.
      const url = URL.createObjectURL(new Blob([buf], { type: 'audio/mpeg' }))
      const audio = new Audio(url)
      audioRef.current = audio
      setState('speaking')
      audio.onended = () => {
        URL.revokeObjectURL(url)
        setState('listening')
      }
      audio.play().catch(() => setState('listening'))
    },
  })

  const { ready, error } = useVAD({
    enabled: state === 'listening',
    onUtterance: (wav) => {
      setState('processing')
      sendVoice(wav)
    },
  })

  const label = error
    ? `Mic/VAD failed to start: ${error.message || error}`
    : !ready
      ? 'Loading microphone…'
      : { listening: 'Listening…', processing: 'Thinking…', speaking: 'Speaking…' }[
          state
        ]

  return (
    <div className="screen">
      <button className="back" onClick={onBack}>
        ← back to lobby
      </button>
      <div className="orb" data-state={state} />
      <div className="status">{label}</div>
      {reply && (
        <div className="exchange">
          <div className="bubble expert">
            <div className="who">{expertId}</div>
            {reply}
          </div>
        </div>
      )}
    </div>
  )
}
