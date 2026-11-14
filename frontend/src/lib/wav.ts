// 16 kHz mono 16-bit WAV: the backend reads it without ffmpeg, and it matches openSMILE/wav2vec2 input.

const TARGET_SR = 16_000

/** Raw mono samples (any rate) -> 16 kHz WAV. Resamples with an OfflineAudioContext; nothing is decoded. */
export async function encodeWav16k(pcm: Float32Array<ArrayBuffer>, sampleRate: number): Promise<Blob> {
  if (sampleRate === TARGET_SR) return wavBlob(pcm)
  const offline = new OfflineAudioContext(1, Math.ceil((pcm.length * TARGET_SR) / sampleRate), TARGET_SR)
  const buf = offline.createBuffer(1, pcm.length, sampleRate)
  buf.copyToChannel(pcm, 0)
  const src = offline.createBufferSource()
  src.buffer = buf
  src.connect(offline.destination)
  src.start()
  return wavBlob((await offline.startRendering()).getChannelData(0))
}

/** A compressed recording (webm/opus, mp4/aac …) -> 16 kHz WAV. Needs the browser to decode it. */
export async function toWav16k(blob: Blob): Promise<Blob> {
  const ctx = new AudioContext()
  let decoded: AudioBuffer
  try { decoded = await ctx.decodeAudioData(await blob.arrayBuffer()) } finally { await ctx.close() }

  const offline = new OfflineAudioContext(1, Math.ceil(decoded.duration * TARGET_SR), TARGET_SR) // 1 channel = downmix
  const src = offline.createBufferSource()
  src.buffer = decoded
  src.connect(offline.destination)
  src.start()
  return wavBlob((await offline.startRendering()).getChannelData(0))
}

function wavBlob(pcm: Float32Array): Blob {
  const buf = new ArrayBuffer(44 + pcm.length * 2)
  const v = new DataView(buf)
  const str = (o: number, s: string) => [...s].forEach((c, i) => v.setUint8(o + i, c.charCodeAt(0)))
  str(0, 'RIFF'); v.setUint32(4, 36 + pcm.length * 2, true); str(8, 'WAVE')
  str(12, 'fmt '); v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true)
  v.setUint32(24, TARGET_SR, true); v.setUint32(28, TARGET_SR * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true)
  str(36, 'data'); v.setUint32(40, pcm.length * 2, true)
  for (let i = 0; i < pcm.length; i++) {
    const s = Math.max(-1, Math.min(1, pcm[i]))
    v.setInt16(44 + i * 2, s < 0 ? s * 0x8000 : s * 0x7fff, true)
  }
  return new Blob([buf], { type: 'audio/wav' })
}
