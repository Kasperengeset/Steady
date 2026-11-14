// Patients: who needs attention first, with a 14-day trend.
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, type PatientSummary } from '../lib/api'
import { Icon, Sparkline } from '../ui/kit'
import { STATUS } from './Portal'

function statusOf(p: PatientSummary) {
  if (p.baseline_status !== 'active' || p.S === null) return 'building'
  return p.S < 1 ? 'close' : p.S < 2 ? 'different' : 'care'
}

export default function Patients() {
  const [rows, setRows] = useState<PatientSummary[] | null>(null)
  const [error, setError] = useState('')
  useEffect(() => {
    let active = true
    const load = () => { void api.patients().then((value) => { if (active) { setRows(value); setError('') } }).catch(() => { if (active) setError('Could not refresh the patient list. Return to this tab to retry.') }) }
    load(); window.addEventListener('focus', load)
    return () => { active = false; window.removeEventListener('focus', load) }
  }, [])

  return (
    <div className="mx-auto max-w-4xl">
      <h1 className="text-3xl font-semibold">Patients</h1>
      {error && <p role="alert" className="mt-3 text-sm text-ink-soft">{error}</p>}
      <div className="mt-6 overflow-hidden rounded-3xl bg-surface-low">
        <table className="w-full text-left">
          <thead className="text-xs uppercase tracking-wider text-ink-soft">
            <tr><th className="px-6 py-4 font-medium">Name</th><th className="font-medium">Status</th><th className="font-medium">Last 14 days</th><th className="font-medium">Change</th><th /></tr>
          </thead>
          <tbody>
            {(rows ?? []).map((p) => {
              const st = STATUS[p.level >= 3 ? 'care' : statusOf(p)]
              return (
                <tr key={p.id} className="border-t border-surface-high">
                  <td className="px-6 py-4 font-semibold">{p.name}{p.level >= 3 && <span className="ml-2 text-xs font-semibold text-terracotta">crisis flag</span>}</td>
                  <td><span className={`rounded-full px-3 py-1 text-sm font-semibold ${st.chip}`}>{st.label}</span>{p.elevated && <span className="ml-2 inline-block rounded-full bg-ochre-soft px-2 py-1 text-xs font-semibold text-secondary" title="Unusually elevated compared with their own healthy baseline. Clinician context only.">Elevated</span>}</td>
                  <td><Sparkline values={p.trend} color={p.S !== null && p.S >= 1 ? '#b4674a' : '#43654f'} /></td>
                  <td className="text-sm text-ink-soft">{p.driver === 'rolling' ? 'Sudden change' : p.driver === 'anchored' ? 'Slow drift' : '-'}</td>
                  <td className="pr-6 text-right">
                    <Link to={`/clinician/patients/${p.id}`} className="inline-flex items-center gap-1 text-primary">Open <Icon name="arrow_forward" /></Link>
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  )
}
