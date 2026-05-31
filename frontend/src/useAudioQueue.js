import { useCallback, useRef } from 'react'

// A small sequential audio player. `enqueue` appends an MP3 chunk and plays
// chunks back-to-back; `flush` stops everything instantly (for barge-in).
// `onStart` fires when playback begins from idle; `onDrain` when the queue
// empties. Callbacks are kept in refs so they never go stale.
export function useAudioQueue({ onStart, onDrain, onBlocked } = {}) {
  const queue = useRef([])
  const current = useRef(null)
  const playing = useRef(false)
  const cbRef = useRef({ onStart, onDrain, onBlocked })
  cbRef.current = { onStart, onDrain, onBlocked }

  const playNext = useCallback(() => {
    if (queue.current.length === 0) {
      playing.current = false
      cbRef.current.onDrain?.()
      return
    }
    playing.current = true
    const url = queue.current.shift()
    const audio = new Audio(url)
    current.current = audio
    const done = () => {
      URL.revokeObjectURL(url)
      playNext()
    }
    audio.onended = done
    audio.onerror = done
    // Browsers block programmatic playback until a user gesture. If we entered
    // chat by voice (no click), the first play() rejects with NotAllowedError;
    // surface that so the UI can ask for a tap.
    audio.play().catch((e) => {
      if (e && e.name === 'NotAllowedError') cbRef.current.onBlocked?.()
      done()
    })
  }, [])

  const enqueue = useCallback(
    (arrayBuffer) => {
      const url = URL.createObjectURL(new Blob([arrayBuffer], { type: 'audio/mpeg' }))
      queue.current.push(url)
      if (!playing.current) {
        cbRef.current.onStart?.()
        playNext()
      }
    },
    [playNext],
  )

  const flush = useCallback(() => {
    queue.current.forEach(URL.revokeObjectURL)
    queue.current = []
    if (current.current) {
      current.current.pause()
      current.current.src = ''
      current.current = null
    }
    playing.current = false
  }, [])

  const isPlaying = useCallback(() => playing.current, [])

  return { enqueue, flush, isPlaying }
}
