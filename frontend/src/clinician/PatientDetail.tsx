// Patient detail: drift over a period, day by day, their own words, what helps, baseline, and why it changes.
import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { api } from '../lib/api'
import { views, type Timeline } from '../lib/views'
import { Card, Icon, LineChart, shortDate, toneStyle } from '../ui/kit'
import { STATUS } from './Portal'
import WhyPanel from './WhyPanel'
import ExactExplanation from './ExactExplanation'

const PERIODS = [7, 30, 90]

export default function PatientDetail() {
  const pid = Number(useParams().id)
  const [days, setDays] = useState(30)
  const [t, setT] = useState<Timeline | null>(null)
  const [picked, setPicked] = useState<number | null>(null)
  const [error, setError] = useState('')
  const load = () => views.timeline(pid, days).then((value) => { setT(value); setError('') }).catch(() => setError('Could not refresh this patient. Return to this tab to retry.'))
  useEffect(() => { setPicked(null); void load(); const focus = () => { void load() }; window.addEventListener('focus', focus); return () => window.removeEventListener('focus', focus) }, [pid, days]) // eslint-disable-line react-hooks/exhaustive-deps

  if (!t) return <p className="text-ink-soft">{error || 'Loading…'}</p>
  const st = STATUS[t.patient.level >= 3 ? 'care' : t.patient.status]
  const day = picked !== null ? t.days[picked] : null
  const dayWords = day ? t.feelings.filter((f) => f.ts.slice(0, 10) === day.date) : []
  const b = t.baseline

  return (
    <div className="mx-auto flex max-w-6xl flex-col gap-6">
      <Link to="/clinician" className="flex items-center gap-1 text-ink-soft"><Icon name="arrow_back" /> Notifications</Link>
      {error && <p role="alert" className="text-sm text-ink-soft">{error}</p>}
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="text-3xl font-semibold">{t.patient.name}</h1>
          <p className="mt-2 flex items-center gap-3 text-ink-soft">
            <span className={`rounded-full px-3 py-1 text-sm font-semibold ${st.chip}`}>{st.label}</span>
            {t.patient.level >= 3 && <span className="rounded-full bg-terracotta-soft px-3 py-1 text-sm font-semibold text-terracotta">Crisis flag</span>}
            {t.patient.days_since_start !== null && <span>Day {t.patient.days_since_start} since TMS</span>}
          </p>
        </div>
        <div className="flex gap-2">
          {PERIODS.map((p) => (
            <button key={p} onClick={() => setDays(p)} className={`rounded-full px-4 py-2 ${days === p ? 'bg-primary text-on-primary' : 'bg-surface-mid text-ink-soft'}`}>{p} days</button>
          ))}
        </div>
      </div>

      <div className="grid gap-6 lg:grid-cols-[1fr_340px]">
        <div className="flex min-w-0 flex-col gap-6">
          <Card className="bg-white p-6 shadow-sm">
            <div className="flex flex-wrap items-center justify-between gap-2">
              <h2 className="text-lg font-semibold">Distance from their own baseline</h2>
              <p className="flex gap-4 text-sm text-ink-soft">
                <span className="flex items-center gap-1.5"><span className="h-0.5 w-5 bg-primary" /> slow drift (healthy baseline)</span>
                <span className="flex items-center gap-1.5"><span className="h-0.5 w-5 border-t-2 border-dashed border-[#7d9db0]" /> sudden change (last 14 days)</span>
              </p>
            </div>
            <div className="mt-4">
              <LineChart height={220} smooth={false} band={[-1, 1]} yMin={-2} yMax={3}
                         refs={[{ y: 1, label: 'friends nudged' }, { y: 2, label: 'clinic' }]}
                         series={[{ values: t.days.map((d) => d.S_anchored), color: '#43654f' },
                                  { values: t.days.map((d) => d.S_rolling), color: '#7d9db0', dashed: true, width: 2 }]}
                         onPick={setPicked} picked={picked} />
            </div>
            <div className="mt-3 flex gap-[3px]">
              {t.days.map((d, i) => {
                const c = d.S === null ? 'bg-surface-high' : d.S >= 2 ? 'bg-terracotta' : d.S >= 1 ? 'bg-ochre' : 'bg-primary-container'
                return (
                  <button key={d.date} onClick={() => setPicked(i)} title={shortDate(d.date)}
                          className={`relative flex h-9 flex-1 flex-col items-center justify-end rounded-md pb-0.5 ${c} ${picked === i ? 'ring-2 ring-ink' : ''}`}>
                    {d.action_done && <Icon name="check" className="text-[11px] text-white" />}
                    {d.checkin && <span className="absolute top-1 h-1 w-1 rounded-full bg-white" />}
                  </button>
                )
              })}
            </div>
            <p className="mt-2 flex justify-between text-xs text-ink-soft">
              <span>{shortDate(t.days[0].date)}</span><span>dot = check-in · tick = action done</span><span>Today</span>
            </p>
            {day && (
              <div className="mt-4 rounded-2xl bg-surface-low p-4">
                <p className="font-semibold">{shortDate(day.date)}</p>
                <p className="mt-1 text-sm text-ink-soft">
                  {day.S === null ? 'No score yet (baseline building).' : `Distance from baseline ${day.S.toFixed(2)}.`}
                  {day.checkin ? ' Checked in.' : ' No check-in.'}
                  {day.action_done ? ` Did today's action${day.rating ? `, rated ${day.rating}/5` : ''}.` : ''}
                </p>
                {dayWords.map((f) => <p key={f.ts} className="mt-2 text-sm italic">"{f.text}"</p>)}
              </div>
            )}
          </Card>

          <WhyPanel pid={pid} days={days} />
          <ExactExplanation pid={pid} />
        </div>

        <div className="flex flex-col gap-6">
          <Card>
            <h2 className="text-lg font-semibold">How they've been feeling</h2>
            <p className="text-sm text-ink-soft">Their own words, newest first</p>
            <div className="mt-4 flex flex-col gap-4">
              {t.feelings.length === 0 && <p className="text-sm text-ink-soft">No written or spoken check-ins in this period.</p>}
              {t.feelings.map((f) => (
                <div key={f.ts} className={`rounded-2xl p-3 ${f.crisis ? 'bg-terracotta-soft' : 'bg-surface'}`}>
                  <p className="flex items-center justify-between text-xs text-ink-soft">
                    <span>{shortDate(f.ts)} · {f.slot}</span>
                    {f.tone && <span className={`rounded-full px-2 py-0.5 font-semibold ${toneStyle[f.tone]}`}>{f.tone}</span>}
                  </p>
                  <p className="mt-1.5 text-sm">{f.text.length > 220 ? f.text.slice(0, 217) + '…' : f.text}</p>
                </div>
              ))}
            </div>
          </Card>

          <Card>
            <h2 className="text-lg font-semibold">What helps</h2>
            <div className="mt-3 flex flex-col gap-2">
              {t.what_helps.map((a) => (
                <div key={a.key} className="flex items-center justify-between text-sm">
                  <span>{a.label}</span>
                  <span className="text-ink-soft">{a.tried ? `helped ${a.helped} of ${a.tried}` : 'not tried'}</span>
                </div>
              ))}
            </div>
          </Card>

          <Card>
            <h2 className="text-lg font-semibold">Healthy baseline</h2>
            {b ? (
              <>
                <p className="mt-2 text-sm text-ink-soft">
                  {b.status === 'building'
                    ? `Building, day ${Math.ceil(b.days_elapsed)} of ${b.period_days}.`
                    : `Active since ${b.locked_at ? shortDate(b.locked_at) : '-'}${b.confirmed_by ? `, confirmed by ${b.confirmed_by}` : ', not confirmed yet'}.`}
                  {' '}{Object.values(b.checkins).reduce((x, y) => x + y, 0)} check-ins.
                </p>
                <div className="mt-4 flex flex-wrap gap-2">
                  {b.status === 'active' && !b.confirmed_by && (
                    <button onClick={async () => { await views.confirmBaseline(pid, 'Dr. Virtanen'); load() }}
                            className="rounded-full bg-primary px-4 py-2 text-sm font-semibold text-on-primary">Confirm baseline</button>
                  )}
                  <button onClick={async () => { await api.baselineAction(pid, 'start', {}); load() }}
                          className="rounded-full bg-surface-high px-4 py-2 text-sm">Start a new baseline</button>
                </div>
              </>
            ) : <p className="mt-2 text-sm text-ink-soft">No baseline yet.</p>}
          </Card>
        </div>
      </div>
      <p className="text-center text-xs text-ink-soft">Each patient is compared with their own baseline. Signals show what changed, not why. This is not a diagnosis.</p>
    </div>
  )
}
