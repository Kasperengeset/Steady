import { useCallback, useEffect, useRef, useState } from 'react'

const MAX_MS = 20_000

export function useRecorder() {
  const [recording, setRecording] = useState(false)
  const [elapsed, setElapsed] = useState(0)
  const [error, setError] = useState<string | null>(null)
  const recRef = useRef<MediaRecorder | null>(null)
  const resolveRef = useRef<((b: Blob) => void) | null>(null)
  const timers = useRef<number[]>([])

  const clearTimers = () => { timers.current.forEach(clearTimeout); timers.current.forEach(clearInterval); timers.current = [] }

  const stop = useCallback(() => {
    if (recRef.current?.state === 'recording') recRef.current.stop()
  }, [])

  /** Starts recording; resolves with the audio when stopped (manually or after 20 s). */
  const start = useCallback(async (): Promise<Blob> => {
    setError(null)
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true }).catch((e: Error) => {
      setError(e.message)
      throw e
    })
    const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus') ? 'audio/webm;codecs=opus' : ''
    const rec = new MediaRecorder(stream, mime ? { mimeType: mime } : undefined)
    const chunks: Blob[] = []
    rec.ondataavailable = (e) => e.data.size && chunks.push(e.data)
    const done = new Promise<Blob>((resolve) => { resolveRef.current = resolve })
    rec.onstop = () => {
      clearTimers()
      stream.getTracks().forEach((t) => t.stop())
      setRecording(false)
      resolveRef.current?.(new Blob(chunks, { type: rec.mimeType || 'audio/webm' }))
    }
    recRef.current = rec
    rec.start()
    setRecording(true)
    setElapsed(0)
    const t0 = Date.now()
    timers.current.push(window.setInterval(() => setElapsed(Date.now() - t0), 200))
    timers.current.push(window.setTimeout(() => rec.stop(), MAX_MS))
    return done
  }, [])

  useEffect(() => () => { clearTimers(); stop() }, [stop])

  return { recording, elapsed, maxMs: MAX_MS, error, start, stop }
}
