import { useEffect, useRef, useState } from 'react'
import { MicVAD } from '@ricky0123/vad-web'

// Encode mono Float32 samples (the format Silero VAD emits) as a 16 kHz,
// 16-bit PCM WAV blob -- the format Gemini accepts natively.
function encodeWAV(samples, sampleRate = 16000) {
  const buffer = new ArrayBuffer(44 + samples.length * 2)
  const view = new DataView(buffer)
  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i))
  }
  writeString(0, 'RIFF')
  view.setUint32(4, 36 + samples.length * 2, true)
  writeString(8, 'WAVE')
  writeString(12, 'fmt ')
  view.setUint32(16, 16, true) // PCM chunk size
  view.setUint16(20, 1, true) // PCM format
  view.setUint16(22, 1, true) // mono
  view.setUint32(24, sampleRate, true)
  view.setUint32(28, sampleRate * 2, true) // byte rate
  view.setUint16(32, 2, true) // block align
  view.setUint16(34, 16, true) // bits per sample
  writeString(36, 'data')
  view.setUint32(40, samples.length * 2, true)
  let offset = 44
  for (let i = 0; i < samples.length; i++, offset += 2) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return new Blob([view], { type: 'audio/wav' })
}

// Voice Activity Detection via Silero (ONNX, in browser). Calls `onSpeechStart`
// the moment the user begins talking (used for barge-in) and `onUtterance` with
// a WAV blob when they finish. `enabled` lets the caller mute the mic.
//
// Assets (worklet, ONNX model, ort wasm) are served locally from /vad/ -- see
// scripts/copy-vad-assets.mjs -- rather than from vad-web's default CDN.
export function useVAD({ onUtterance, onSpeechStart, enabled = true }) {
  const [ready, setReady] = useState(false)
  const [error, setError] = useState(null)
  const vadRef = useRef(null)
  const onUtteranceRef = useRef(onUtterance)
  const onSpeechStartRef = useRef(onSpeechStart)
  onUtteranceRef.current = onUtterance
  onSpeechStartRef.current = onSpeechStart

  // Construct the VAD once. This requests mic access and loads the model.
  useEffect(() => {
    let cancelled = false
    MicVAD.new({
      baseAssetPath: '/vad/',
      onnxWASMBasePath: '/vad/',
      // Less twitchy than defaults: the mic is always on (for barge-in), so the
      // expert's OWN audio leaks back in. Require more confident, sustained
      // speech before firing, so echo-cancellation residual doesn't self-trigger
      // and flush the lecture that's playing.
      positiveSpeechThreshold: 0.82,
      negativeSpeechThreshold: 0.6,
      minSpeechFrames: 8,
      redemptionFrames: 16,
      onSpeechStart: () => onSpeechStartRef.current?.(),
      onSpeechEnd: (audio) => onUtteranceRef.current(encodeWAV(audio)),
    })
      .then((vad) => {
        if (cancelled) {
          vad.destroy()
          return
        }
        vadRef.current = vad
        setReady(true) // state change -> the start/stop effect below re-runs
      })
      .catch((e) => {
        console.error('[useVAD] failed to initialize:', e)
        setError(e)
      })
    return () => {
      cancelled = true
      vadRef.current?.destroy()
      vadRef.current = null
      setReady(false)
    }
  }, [])

  // Start/stop listening. Depends on `ready` (state), NOT vadRef.current (a ref
  // that never triggers re-runs) -- the earlier bug was that this effect ran
  // once before the async load finished and so start() was never called.
  useEffect(() => {
    if (!ready || !vadRef.current) return
    if (enabled) vadRef.current.start()
    else vadRef.current.pause()
  }, [ready, enabled])

  return { isListening: ready && enabled, ready, error }
}
