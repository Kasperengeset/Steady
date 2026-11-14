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
            {s === 'morning' ? 'Morgen' : 'Kveld'}
          </button>
        ))}
      </div>

      <p className="prompt">
        {slot === 'morning' ? 'Hvordan sov du, og hvordan kjennes dagen ut?' : 'Hvordan har dagen vært?'}
      </p>

      <button
        className={`record ${rec.recording ? 'on' : ''}`}
        onClick={onRecord}
        disabled={busy}
        aria-label={rec.recording ? 'Stopp opptak' : 'Start opptak'}
      >
        {busy ? '…' : rec.recording ? `${secs}s` : 'Snakk'}
      </button>
      <p className="hint">{rec.recording ? 'Trykk igjen for å stoppe' : 'Ca. 20 sekunder'}</p>

      {(error || rec.error) && <p className="error">{error || rec.error}</p>}

      {result?.crisis && (
        <div className="card crisis">
          <strong>Du er ikke alene.</strong> Behandleren din er varslet. Trenger du hjelp nå, ring 112.
        </div>
      )}

      {result && (
        <div className="card">
          <h2>Dagens tiltak</h2>
          <p className="intervention">{result.intervention.label}</p>
          {slot === 'evening' && feedbackSent === null && (
            <div className="feedback">
              <p>Fikk du gjort det? Hvor nyttig var det?</p>
              <div className="rating">
                {[1, 2, 3, 4, 5].map((n) => (
                  <button key={n} onClick={() => sendFeedback(true, n)}>{n}</button>
                ))}
              </div>
              <button className="link" onClick={() => sendFeedback(false, null)}>Ble ikke gjort</button>
            </div>
          )}
          {feedbackSent !== null && <p className="muted">Takk! Registrert.</p>}
        </div>
      )}

      {result && (
        <details className="debug">
          <summary>Debug: trekk og tilstand</summary>
          <p>S = {result.state.S?.toFixed(2) ?? '— (bygger baseline)'} · nivå {result.state.level}</p>
          <p>Grener: {Object.entries(result.state.branches).map(([b, d]) => `${b} ${d.toFixed(2)}`).join(' · ') || '—'}</p>
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
