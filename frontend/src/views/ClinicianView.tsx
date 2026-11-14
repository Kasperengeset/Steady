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
              <td><span className={`level l${p.level}`}>{LEVEL_LABEL[p.level]}</span></td>
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
          {/* TODO: trend chart + top-3 contributing features visualised */}
          <ol>
            {selected.top_features.map((f) => (
              <li key={f.branch + f.feature}>{f.branch}/{f.feature}: {f.contribution > 0 ? '+' : ''}{f.contribution}</li>
            ))}
          </ol>
          <p className="muted">{selected.checkins.length} check-ins recorded</p>
        </div>
      )}
    </section>
  )
}
