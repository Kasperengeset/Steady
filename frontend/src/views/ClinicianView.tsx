import { useEffect, useState } from 'react'
import { api, type PatientDetail, type PatientSummary } from '../lib/api'

const LEVEL_LABEL = ['Stable', 'Watch', 'Contact', 'Crisis']
const BASELINE_LABEL = { building: 'Building baseline', active: 'Baseline active' } as const
const DRIVER_LABEL = { anchored: 'drift from healthy normal', rolling: 'abrupt change' } as const

export default function ClinicianView() {
  const [patients, setPatients] = useState<PatientSummary[]>([])
  const [selected, setSelected] = useState<PatientDetail | null>(null)
  const [error, setError] = useState<string | null>(null)

  const load = () => api.patients().then(setPatients).catch((e) => setError(String(e)))
  useEffect(() => {
    load()
  }, [])

  async function baselineAction(id: number, action: 'start' | 'reject') {
    const note = action === 'reject' ? 'Rejected by clinician' : 'New baseline started by clinician'
    await api.baselineAction(id, action, { note })
    load()
  }

  return (
    <section className="view">
      <h1>Patients</h1>
      {error && <p className="error">{error}</p>}
      <table className="patients">
        <thead>
          <tr><th>Name</th><th>Status</th><th>S</th><th>Why</th><th>Baseline</th><th>Trend (14 d)</th></tr>
        </thead>
        <tbody>
          {patients.map((p) => (
            <tr key={p.id} onClick={() => api.patient(p.id).then(setSelected)}>
              <td>{p.name}</td>
              <td>
                <span className={`level l${p.level}`}>{LEVEL_LABEL[p.level]}</span>
                {p.elevated && <span className="level elevated" title="Unusually elevated vs healthy normal – clinician only"> elevated</span>}
              </td>
              <td>{p.S?.toFixed(2) ?? '—'}</td>
              <td className="muted">{p.driver ? DRIVER_LABEL[p.driver] : '—'}</td>
              <td onClick={(e) => e.stopPropagation()}>
                <span className="muted">{p.baseline_status ? BASELINE_LABEL[p.baseline_status] : '—'}</span>{' '}
                <button className="link" onClick={() => baselineAction(p.id, 'start')}>new</button>{' '}
                <button className="link" onClick={() => baselineAction(p.id, 'reject')}>reject</button>
              </td>
              <td className="muted">{p.trend.map((s) => s.toFixed(1)).join(' · ') || '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {selected && (
        <div className="card">
          <h2>Why? – {String(selected.patient.name)}</h2>
          {selected.explanation ? (
            <>
              <p>{selected.explanation.sentence}</p>
              <p className="muted">
                S = {selected.explanation.S.toFixed(2)} ({selected.explanation.track === 'anchored' ? 'vs healthy normal' : 'vs last 14 days'}),
                split exactly over features and days:
              </p>
              <ul className="bars">
                {Object.entries(selected.explanation.by_feature).slice(0, 5).map(([f, v]) => (
                  <li key={f}>
                    <span className="bar-label">{f}</span>
                    <span className={`bar ${v > 0 ? 'worse' : 'better'}`} style={{ width: `${Math.min(Math.abs(v) * 60, 100)}%` }} />
                    <span className="muted">{v > 0 ? '+' : ''}{v.toFixed(2)}</span>
                  </li>
                ))}
              </ul>
              <p className="muted">
                Last days: {selected.explanation.by_day.slice(-5).map((d) => `${d.date.slice(5)} ${d.contribution > 0 ? '+' : ''}${d.contribution.toFixed(2)}`).join(' · ')}
                {Math.abs(selected.explanation.older) > 0.01 && ` · older ${selected.explanation.older.toFixed(2)}`}
              </p>
            </>
          ) : (
            <p className="muted">No score yet – the healthy baseline is still being built.</p>
          )}
          <p className="muted">{selected.checkins.length} check-ins recorded</p>
        </div>
      )}
    </section>
  )
}
