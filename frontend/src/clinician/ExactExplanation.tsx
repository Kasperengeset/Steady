import { useEffect, useState } from 'react'
import { api, type PatientDetail } from '../lib/api'
import { Card, shortDate } from '../ui/kit'

export default function ExactExplanation({ pid }: { pid: number }) {
  const [detail, setDetail] = useState<PatientDetail | null>(null)
  const [error, setError] = useState('')
  useEffect(() => { let active = true; const load = () => { void api.patient(pid).then((value) => { if (active) { setDetail(value); setError('') } }).catch(() => { if (active) setError('The exact score explanation could not be loaded.') }) }; load(); window.addEventListener('focus', load); return () => { active = false; window.removeEventListener('focus', load) } }, [pid])
  if (error) return <Card><p className="text-sm text-ink-soft">{error}</p></Card>
  if (!detail?.explanation) return null
  const e = detail.explanation
  const top = Object.entries(e.by_feature).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 6)
  const scale = Math.max(...top.map(([, value]) => Math.abs(value)), 0.01)
  return <Card className="bg-white p-6 shadow-sm">
    <h2 className="text-lg font-semibold">How the current score adds up</h2>
    {Boolean(detail.patient.elevated) && <p className="mt-3 rounded-2xl bg-ochre-soft p-3 text-sm">Unusually elevated compared with their own healthy baseline. This is clinician context, separate from the deterioration alert.</p>}
    <p className="mt-3 text-sm leading-relaxed">{e.sentence}</p><p className="mt-2 text-xs text-ink-soft">S = {e.S.toFixed(2)} · {e.track === 'anchored' ? 'healthy baseline' : 'last 14 days'}. Exact feature and day contributions from the engine.</p>
    <ul className="mt-5 flex flex-col gap-3">{top.map(([feature, contribution]) => <li key={feature}><div className="flex justify-between gap-3 text-xs"><span>{feature.replace(/_/g, ' ')}</span><span className="font-semibold">{contribution > 0 ? '+' : ''}{contribution.toFixed(2)}</span></div><div className="mt-2 h-2 rounded-full bg-surface-mid"><div className={`h-2 rounded-full ${contribution > 0 ? 'bg-terracotta/70' : 'bg-primary-container'}`} style={{ width: `${Math.abs(contribution) / scale * 100}%` }} /></div></li>)}</ul>
    <details className="mt-5 border-t border-surface-high pt-4"><summary className="cursor-pointer text-xs font-semibold text-primary">Contributions over time</summary><ul className="mt-3 grid gap-2 text-xs text-ink-soft">{e.by_day.slice(-14).map((day) => <li key={day.date} className="flex justify-between"><span>{shortDate(day.date)}</span><span>{day.contribution > 0 ? '+' : ''}{day.contribution.toFixed(3)}</span></li>)}{Math.abs(e.older) > 0.001 && <li className="flex justify-between"><span>Earlier history</span><span>{e.older.toFixed(3)}</span></li>}</ul></details>
  </Card>
}
