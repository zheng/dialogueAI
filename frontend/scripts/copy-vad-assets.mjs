// Copy the VAD worklet, Silero ONNX model, and onnxruntime-web wasm out of
// node_modules into public/vad/ so the app serves them itself instead of
// fetching from a CDN pinned to @latest (a common source of version-mismatch
// breakage). Run automatically before `dev` and `build`.
import { mkdirSync, copyFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const root = join(dirname(fileURLToPath(import.meta.url)), '..')
const out = join(root, 'public', 'vad')
mkdirSync(out, { recursive: true })

const vad = join(root, 'node_modules', '@ricky0123', 'vad-web', 'dist')
const ort = join(root, 'node_modules', 'onnxruntime-web', 'dist')

const files = [
  [vad, 'vad.worklet.bundle.min.js'],
  [vad, 'silero_vad_legacy.onnx'],
  [vad, 'silero_vad_v5.onnx'],
  [ort, 'ort-wasm-simd.wasm'],
  [ort, 'ort-wasm-simd-threaded.wasm'],
  [ort, 'ort-wasm-threaded.wasm'],
  [ort, 'ort-wasm.wasm'],
]

for (const [dir, name] of files) {
  copyFileSync(join(dir, name), join(out, name))
}
console.log(`copied ${files.length} VAD assets -> public/vad/`)
