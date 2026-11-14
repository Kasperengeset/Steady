import { useEffect, useState } from 'react'
import { api, type PatientDetail, type PatientSummary } from '../lib/api'

const LEVEL_LABEL = ['Stabil', 'Følg med', 'Ta kontakt', 'Krise']

export default function ClinicianView() {
  const [patients, setPatients] = useState<PatientSummary[]>([])
  const [selected, setSelected] = useState<PatientDetail | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.patients().then(setPatients).catch((e) => setError(String(e)))
  }, [])

  return (
    <section className="view">
      <h1>Pasienter</h1>
      {error && <p className="error">{error}</p>}
      <table className="patients">
        <thead>
          <tr><th>Navn</th><th>Status</th><th>S</th><th>Trend (14 d)</th></tr>
        </thead>
        <tbody>
          {patients.map((p) => (
            <tr key={p.id} onClick={() => api.patient(p.id).then(setSelected)}>
              <td>{p.name}</td>
              <td><span className={`level l${p.level}`}>{LEVEL_LABEL[p.level]}</span></td>
              <td>{p.S?.toFixed(2) ?? '—'}</td>
              <td className="muted">{p.trend.map((s) => s.toFixed(1)).join(' · ') || '—'}</td>
            </tr>
          ))}
        </tbody>
      </table>

      {selected && (
        <div className="card">
          <h2>Hvorfor? – {String(selected.patient.name)}</h2>
          {/* TODO(A, lør): trendgraf + topp-3 bidragende trekk visualisert */}
          <ol>
            {selected.top_features.map((f) => (
              <li key={f.branch + f.feature}>{f.branch}/{f.feature}: {f.contribution > 0 ? '+' : ''}{f.contribution}</li>
            ))}
          </ol>
          <p className="muted">{selected.checkins.length} innsjekker registrert</p>
        </div>
      )}
    </section>
  )
}
