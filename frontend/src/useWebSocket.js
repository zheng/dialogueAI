import { useEffect, useRef, useState } from 'react'

// Thin WebSocket hook for the voice protocol: outgoing is a JSON control frame
// followed by a binary WAV; incoming is JSON (text) and/or binary (audio).
// `onJson` and `onBinary` receive parsed messages. Reconnects on drop.
export function useWebSocket(path, { onJson, onBinary } = {}) {
  const [connected, setConnected] = useState(false)
  const wsRef = useRef(null)
  const cbRef = useRef({ onJson, onBinary })
  cbRef.current = { onJson, onBinary }

  useEffect(() => {
    let closedByUs = false
    let reconnectTimer = null

    const connect = () => {
      const proto = location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${proto}://${location.host}${path}`)
      ws.binaryType = 'arraybuffer'
      ws.onopen = () => setConnected(true)
      ws.onclose = () => {
        setConnected(false)
        if (!closedByUs) reconnectTimer = setTimeout(connect, 1000)
      }
      ws.onmessage = (e) => {
        if (typeof e.data === 'string') cbRef.current.onJson?.(JSON.parse(e.data))
        else cbRef.current.onBinary?.(e.data)
      }
      wsRef.current = ws
    }

    connect()
    return () => {
      closedByUs = true
      clearTimeout(reconnectTimer)
      wsRef.current?.close()
    }
  }, [path])

  // Send a control message (interrupt / resume / end_lecture / …).
  const send = (obj) => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    ws.send(JSON.stringify(obj))
  }

  // Send a voice utterance: control frame + the WAV blob.
  const sendVoice = async (wavBlob) => {
    const ws = wsRef.current
    if (!ws || ws.readyState !== WebSocket.OPEN) return
    ws.send(JSON.stringify({ type: 'voice_input' }))
    ws.send(await wavBlob.arrayBuffer())
  }

  return { connected, send, sendVoice }
}
