import sys, time, whisper

AUDIO = "/tmp/micrograd_16k.wav"
OUT = "experts/karpathy/transcripts/micrograd.txt"

t0 = time.time()
print(f"[{time.strftime('%H:%M:%S')}] loading model base.en", flush=True)
model = whisper.load_model("base.en")
print(f"[{time.strftime('%H:%M:%S')}] model loaded in {time.time()-t0:.0f}s, transcribing...", flush=True)

result = model.transcribe(AUDIO, language="en", fp16=False, verbose=False)

text = result["text"].strip()
with open(OUT, "w", encoding="utf-8") as f:
    f.write(text + "\n")

print(f"[{time.strftime('%H:%M:%S')}] done in {time.time()-t0:.0f}s, wrote {len(text)} chars to {OUT}", flush=True)
