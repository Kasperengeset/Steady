import { useEffect, useRef, useState } from 'react'
import type { Slot } from '../../lib/api'
import { checkinApi, PHRASE, type DailyCheckInResponse } from '../../lib/api_checkin'
import { useRecorder } from '../../hooks/useRecorder'
import './checkin.css'

const PHRASE_MS = 15_000
const MEMO_MS = 30_000

type Step = 'ready' | 'phrase' | 'memo' | 'sending'

interface Props {
  patientId: number
  slot: Slot
  /** Bump to jump straight to the phrase step (e.g. from the "Your check-in is ready" banner). */
  startSignal?: number
  onResult: (r: DailyCheckInResponse) => void
}

/**
 * The daily moment (#21): 1. read the same phrase aloud (large text, stops on tap or after 15 s),
 * 2. talk about the day (30 s ring). Both go up in ONE request, as `phrase` + `audio`, so the server
 * never has to find the phrase inside free talk. The phrase can be skipped; the memo still counts.
 */
export default function DailyCheckIn({ patientId, slot, startSignal = 0, onResult }: Props) {
  const [step, setStep] = useState<Step>('ready')
  const [error, setError] = useState<string | null>(null)
  const phrase = useRef<Blob | null>(null)
  const rec = useRecorder()

  useEffect(() => {
    if (startSignal > 0) setStep((s) => (s === 'ready' ? 'phrase' : s))
  }, [startSignal])

  function begin() {
    setError(null)
    phrase.current = null
    setStep('phrase')
  }

  async function recordPhrase() {
    if (rec.recording) return rec.stop()
    setError(null)
    try {
      phrase.current = await rec.start(PHRASE_MS)
      setStep('memo')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }

  function skipPhrase() {
    phrase.current = null
    setStep('memo')
  }

  async function recordMemo() {
    if (rec.recording) return rec.stop()
    setError(null)
    try {
      const memo = await rec.start(MEMO_MS)
      setStep('sending')
      onResult(await checkinApi.checkIn(patientId, slot, memo, phrase.current))
      phrase.current = null
      setStep('ready')
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
      setStep('memo')
    }
  }

  const progress = rec.recording ? Math.min(rec.elapsed / rec.maxMs, 1) : 0
  const secs = Math.max(0, Math.ceil((rec.maxMs - rec.elapsed) / 1000))
  const err = error || rec.error

  if (step === 'ready') {
    return (
      <div className="daily">
        <p className="prompt">{slot === 'morning' ? 'How did you sleep, and how does the day feel?' : 'How has your day been?'}</p>
        <button className="record" onClick={begin}>Check in</button>
        <p className="hint">Read one sentence, then talk for 30 seconds</p>
        {err && <p className="error">{err}</p>}
      </div>
    )
  }

  return (
    <div className="daily">
      <ol className="steps" aria-label="Check-in steps">
        <li aria-current={step === 'phrase'}>1 Read</li>
        <li aria-current={step !== 'phrase'}>2 Your day</li>
      </ol>

      {step === 'phrase' ? (
        <>
          <h2>Read this aloud</h2>
          <blockquote className="phrase">{PHRASE}</blockquote>
          <RecordButton recording={rec.recording} progress={progress} onClick={recordPhrase}
            label={rec.recording ? `${secs}s` : 'Read'} />
          <p className="hint">{rec.recording ? 'Tap when you are done' : 'The same sentence every day, at your normal pace'}</p>
          {!rec.recording && <button className="link" onClick={skipPhrase}>Skip reading today</button>}
        </>
      ) : (
        <>
          <h2>Now tell me about your day</h2>
          <p className="prompt">{slot === 'morning' ? 'How did you sleep, and how does the day feel?' : 'How has your day been?'}</p>
          <RecordButton recording={rec.recording} progress={progress} onClick={recordMemo} disabled={step === 'sending'}
            label={step === 'sending' ? '…' : rec.recording ? `${secs}s` : 'Speak'} />
          <p className="hint">{step === 'sending' ? 'Saving…' : rec.recording ? 'Tap again to stop' : 'About 30 seconds'}</p>
        </>
      )}

      {err && <p className="error">{err}</p>}
      {!rec.recording && step !== 'sending' && (
        <button className="link" onClick={() => setStep('ready')}>Cancel</button>
      )}
    </div>
  )
}

function RecordButton({ recording, progress, onClick, label, disabled }: {
  recording: boolean; progress: number; onClick: () => void; label: string; disabled?: boolean
}) {
  const r = 90
  const c = 2 * Math.PI * r
  return (
    <div className="ring-wrap">
      <svg className="ring" viewBox="0 0 196 196" aria-hidden="true">
        <circle cx="98" cy="98" r={r} className="ring-track" />
        <circle cx="98" cy="98" r={r} className="ring-progress"
          strokeDasharray={c} strokeDashoffset={c * (1 - progress)} />
      </svg>
      <button className={`record ${recording ? 'on' : ''}`} onClick={onClick} disabled={disabled}
        aria-label={recording ? 'Stop recording' : 'Start recording'}>
        {label}
      </button>
    </div>
  )
}
