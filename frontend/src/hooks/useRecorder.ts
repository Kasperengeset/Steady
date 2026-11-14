// Capture microphone samples directly, then make a mono 16 kHz WAV for the voice pipeline.
import { useCallback, useEffect, useRef, useState } from 'react'
import { encodeWav16k } from '../lib/wav'

const MAX_MS = 30_000
const MIN_S = 3
const TAP = `class SteadyTap extends AudioWorkletProcessor {
  process(inputs) {
    const channel = inputs[0] && inputs[0][0]
    if (channel && channel.length) this.port.postMessage(channel.slice(0))
    return true
  }
}
registerProcessor('steady-tap', SteadyTap)`

function micMessage(error: unknown): string {
  const name = error instanceof DOMException ? error.name : ''
  if (name === 'NotAllowedError' || name === 'SecurityError')
    return 'Microphone access is blocked. Allow it from the address bar, or write a check-in instead.'
  if (name === 'NotFoundError' || name === 'OverconstrainedError')
    return 'No microphone was found. Connect one and try again, or write a check-in.'
  if (name === 'NotReadableError' || name === 'AbortError')
    return 'The microphone is busy or disconnected. Close other apps using it and try again.'
  return 'This browser could not start the microphone. Try Chrome or Edge, or write a check-in.'
}

function concat(chunks: Float32Array[]): Float32Array<ArrayBuffer> {
  const samples = new Float32Array(chunks.reduce((length, chunk) => length + chunk.length, 0))
  let offset = 0
  for (const chunk of chunks) { samples.set(chunk, offset); offset += chunk.length }
  return samples
}

function recordingProblem(samples: Float32Array, sampleRate: number): string | null {
  if (samples.length < sampleRate * MIN_S)
    return `That was a little short. Record at least ${MIN_S} seconds before finishing.`
  let peak = 0
  for (const sample of samples) peak = Math.max(peak, Math.abs(sample))
  if (peak < 0.001)
    return 'The recording was silent. Check that your microphone is selected and unmuted, then try again.'
  return null
}

export function useRecorder(maxMs = MAX_MS) {
  const [recording, setRecording] = useState(false)
  const [starting, setStarting] = useState(false)
  const [preparing, setPreparing] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [activeMaxMs, setActiveMaxMs] = useState(maxMs)
  const [error, setError] = useState<string | null>(null)
  const mounted = useRef(true)
  const finalizing = useRef(false)
  const session = useRef<{ finish: () => void; abort: () => void } | null>(null)
  const stop = useCallback(() => session.current?.finish(), [])
  const cancel = useCallback(() => session.current?.abort(), [])

  /** Resolves only when a usable recording is ready. Cancelling never uploads anything. */
  const start = useCallback((limitMs = maxMs): Promise<Blob> => {
    const durationMs = Number.isFinite(limitMs) ? Math.max(1000, Math.min(120000, limitMs)) : maxMs
    if (session.current || finalizing.current) return Promise.reject(new Error('A recording is already starting or running.'))
    if (!navigator.mediaDevices?.getUserMedia || !window.AudioContext) {
      const message = 'Recording needs a supported browser on HTTPS or localhost. You can write a check-in instead.'
      setError(message)
      return Promise.reject(new Error(message))
    }
    setError(null)
    setElapsed(0)
    setActiveMaxMs(durationMs)
    setStarting(true)

    return new Promise<Blob>((resolve, reject) => {
      let finished = false
      let stream: MediaStream | null = null
      let context: AudioContext | null = null
      let source: MediaStreamAudioSourceNode | null = null
      let tap: AudioWorkletNode | null = null
      let interval: number | undefined
      let timeout: number | undefined
      let startedAt = 0
      const chunks: Float32Array[] = []

      const cleanup = () => {
        window.clearInterval(interval)
        window.clearTimeout(timeout)
        if (tap) { tap.port.onmessage = null; tap.disconnect(); tap.port.close() }
        source?.disconnect()
        stream?.getAudioTracks().forEach((track) => {
          track.removeEventListener('ended', finish)
          track.stop()
        })
        if (context && context.state !== 'closed') void context.close().catch(() => {})
        session.current = null
        if (mounted.current) { setRecording(false); setStarting(false) }
      }
      const fail = (message: string) => {
        if (finished) return
        finished = true
        cleanup()
        if (mounted.current) setError(message)
        reject(new Error(message))
      }
      const finish = () => {
        if (finished || !context || !startedAt) return
        finished = true
        const sampleRate = context.sampleRate
        const samples = concat(chunks)
        if (mounted.current) setElapsed(Math.min(durationMs, samples.length / sampleRate * 1000))
        cleanup()
        const problem = recordingProblem(samples, sampleRate)
        if (problem) {
          if (mounted.current) setError(problem)
          reject(new Error(problem))
          return
        }
        finalizing.current = true
        if (mounted.current) setPreparing(true)
        encodeWav16k(samples, sampleRate).then((blob) => {
          finalizing.current = false
          if (mounted.current) setPreparing(false)
          resolve(blob)
        }, () => {
          finalizing.current = false
          const message = 'The recording could not be prepared. Try again, or write a check-in instead.'
          if (mounted.current) { setPreparing(false); setError(message) }
          reject(new Error(message))
        })
      }
      const abort = () => {
        if (finished) return
        finished = true
        cleanup()
        reject(new DOMException('Recording cancelled', 'AbortError'))
      }
      // Register before asking permission: double clicks and navigation during the permission dialog are safe.
      session.current = { finish, abort }
      void (async () => {
        try {
          stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1 } })
          if (finished || !mounted.current) { stream.getTracks().forEach((track) => track.stop()); return }
          context = new AudioContext()
          await context.resume()
          if (finished) return
          if (!context.audioWorklet) throw new Error('AudioWorklet is unavailable')
          const moduleUrl = URL.createObjectURL(new Blob([TAP], { type: 'text/javascript' }))
          try { await context.audioWorklet.addModule(moduleUrl) } finally { URL.revokeObjectURL(moduleUrl) }
          if (finished) return
          source = context.createMediaStreamSource(stream)
          tap = new AudioWorkletNode(context, 'steady-tap')
          tap.port.onmessage = (event: MessageEvent<Float32Array>) => { if (!finished) chunks.push(event.data) }
          source.connect(tap)
          tap.connect(context.destination) // No output is produced; this connection keeps capture active.
          startedAt = performance.now()
          setStarting(false)
          setRecording(true)
          interval = window.setInterval(() => {
            if (mounted.current) setElapsed(Math.min(durationMs, performance.now() - startedAt))
          }, 150)
          timeout = window.setTimeout(finish, durationMs)
          stream.getAudioTracks().forEach((track) => track.addEventListener('ended', finish, { once: true }))
        } catch (cause) { fail(micMessage(cause)) }
      })()
    })
  }, [maxMs])

  useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false; session.current?.abort() }
  }, [])

  return { recording, starting, preparing, elapsed, maxMs: recording || starting ? activeMaxMs : maxMs, error, start, stop, cancel }
}
