import { useState } from 'react'
import { api, todayUTC, type CheckInResponse, type Slot } from '../lib/api'
import { useRecorder } from '../hooks/useRecorder'

const PATIENT_ID = 1 // TODO: login is last on the cut list – demo patient for now

export default function PatientView() {
  const [slot, setSlot] = useState<Slot>(new Date().getHours() < 14 ? 'morning' : 'evening')
  const [result, setResult] = useState<CheckInResponse | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [feedbackSent, setFeedbackSent] = useState<number | null>(null)
  const rec = useRecorder()

  async function onRecord() {
    if (rec.recording) return rec.stop()
    setError(null)
    try {
      const audio = await rec.start()
      setBusy(true)
      setResult(await api.checkIn(PATIENT_ID, slot, audio))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  async function sendFeedback(completed: boolean, rating: number | null) {
    const r = await api.feedback(PATIENT_ID, todayUTC(), completed, rating)
    setFeedbackSent(r.reward)
  }

  const secs = Math.ceil((rec.maxMs - rec.elapsed) / 1000)

  return (
    <section className="view patient">
      <div className="segmented" role="tablist">
        {(['morning', 'evening'] as const).map((s) => (
          <button key={s} role="tab" aria-selected={slot === s} onClick={() => setSlot(s)}>
            {s === 'morning' ? 'Morning' : 'Evening'}
          </button>
        ))}
      </div>

      <p className="prompt">
        {slot === 'morning' ? 'How did you sleep, and how does the day feel?' : 'How has your day been?'}
      </p>

      <button
        className={`record ${rec.recording ? 'on' : ''}`}
        onClick={onRecord}
        disabled={busy}
        aria-label={rec.recording ? 'Stopp opptak' : 'Start opptak'}
      >
        {busy ? '…' : rec.recording ? `${secs}s` : 'Speak'}
      </button>
      <p className="hint">{rec.recording ? 'Tap again to stop' : 'About 20 seconds'}</p>

      {(error || rec.error) && <p className="error">{error || rec.error}</p>}

      {result?.crisis && (
        <div className="card crisis">
          <strong>You are not alone.</strong> Your clinician has been notified. If you need help now, call 112.
        </div>
      )}

      {result?.baseline.status === 'building' && (
        <div className="card">
          <h2>Building your normal</h2>
          <p>
            Day {Math.min(Math.floor(result.baseline.days_elapsed) + 1, result.baseline.period_days)} of {result.baseline.period_days} ·{' '}
            {result.baseline.checkins.morning + result.baseline.checkins.evening} check-ins
          </p>
          <p className="muted">For the first weeks the app learns how you sound when you are well. After that we compare with it.</p>
        </div>
      )}

      {result?.baseline.status === 'active' && result.baseline_building && (
        <div className="card">
          <h2>Updating your normal</h2>
          <p>
            Day {Math.min(Math.floor(result.baseline_building.days_elapsed) + 1, result.baseline_building.period_days)} of{' '}
            {result.baseline_building.period_days}
          </p>
          <p className="muted">Your previous normal is used meanwhile.</p>
        </div>
      )}

      {result && (
        <div className="card">
          <h2>Today's activity</h2>
          <p className="intervention">{result.intervention.label}</p>
          {slot === 'evening' && feedbackSent === null && (
            <div className="feedback">
              <p>Did you do it? How helpful was it?</p>
              <div className="rating">
                {[1, 2, 3, 4, 5].map((n) => (
                  <button key={n} onClick={() => sendFeedback(true, n)}>{n}</button>
                ))}
              </div>
              <button className="link" onClick={() => sendFeedback(false, null)}>Didn't do it</button>
            </div>
          )}
          {feedbackSent !== null && <p className="muted">Thanks! Saved.</p>}
        </div>
      )}

      {result && (
        <details className="debug">
          <summary>Debug: features and state</summary>
          <p>S = {result.state.S?.toFixed(2) ?? '— (building baseline)'} · level {result.state.level}
            {result.state.driver && ` · driven by ${result.state.driver === 'anchored' ? 'healthy baseline' : 'last 14 days'}`}</p>
          <p>Healthy baseline: {result.state.S_anchored?.toFixed(2) ?? '—'} · Last 14 days: {result.state.S_rolling?.toFixed(2) ?? '—'}</p>
          {result.state.explanation && <p>{result.state.explanation.sentence}</p>}
          <p>Branches: {Object.entries(result.state.branches).map(([b, d]) => `${b} ${d.toFixed(2)}`).join(' · ') || '—'}</p>
          <ul>
            {result.features.map((f) => (
              <li key={f.name}>{f.source}/{f.name}: {f.value}</li>
            ))}
          </ul>
        </details>
      )}
    </section>
  )
}
