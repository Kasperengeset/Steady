// Insights: what helps this person, learned by Thompson sampling from their own answers.
import { useEffect, useState } from 'react'
import { Link, useParams } from 'react-router-dom'
import { views, type ArmInsight } from '../lib/views'
import { Card, Icon, shortDate } from '../ui/kit'
import { PID } from './Shell'

const ICON: Record<string, string> = {
  walk_outside: 'directions_walk', call_friend: 'chat', morning_light: 'wb_twilight', music: 'graphic_eq',
  breathing: 'air', small_task: 'task_alt',
}

export function Insights() {
  const [arms, setArms] = useState<ArmInsight[] | null>(null)
  useEffect(() => { views.insights(PID).then(setArms) }, [])
  if (!arms) return <p className="pt-8 text-ink-soft">Loading…</p>
  // Order = what the bandit learned; bar = the plain fact the patient reads ("helped 6 of 9").
  const share = (a: ArmInsight) => (a.tried ? a.helped / a.tried : 0)
  const tried = arms.filter((a) => a.done > 0)

  return (
    <div className="flex flex-col gap-5 pt-2">
      <div>
        <h1 className="text-3xl font-semibold">What helps you</h1>
        <p className="mt-1 text-ink-soft">Learned from your own answers.</p>
      </div>
      {tried.length === 0 && <p className="text-ink-soft">Try a few suggestions first. Steady learns from how they went.</p>}
      {arms.map((a) => (
        <Link key={a.key} to={`/insights/${a.key}`}>
          <Card className="flex flex-col gap-3">
            <div className="flex items-center gap-3">
              <span className="grid h-10 w-10 place-items-center rounded-full bg-surface text-primary"><Icon name={ICON[a.key] ?? 'spa'} /></span>
              <span className="flex-1 text-lg font-semibold">{a.label}</span>
              <Icon name="chevron_right" className="text-outline" />
            </div>
            <div className="h-2.5 rounded-full bg-surface-high">
              <div className="h-2.5 rounded-full bg-primary-container" style={{ width: `${Math.max(3, share(a) * 100)}%` }} />
            </div>
            <p className="text-sm text-ink-soft">{a.tried ? `Helped ${a.helped} of ${a.tried} times` : 'Not tried yet'}</p>
          </Card>
        </Link>
      ))}
    </div>
  )
}

export function ActionDetail() {
  const { key } = useParams()
  const [a, setA] = useState<ArmInsight | null>(null)
  useEffect(() => { views.insights(PID).then((xs) => setA(xs.find((x) => x.key === key) ?? null)) }, [key])
  if (!a) return <p className="pt-8 text-ink-soft">Loading…</p>

  return (
    <div className="flex flex-col gap-5 pt-2">
      <Link to="/insights" className="flex items-center gap-1 text-ink-soft"><Icon name="arrow_back" /> What helps you</Link>
      <div className="flex items-center gap-3">
        <span className="grid h-12 w-12 place-items-center rounded-full bg-sage-soft text-primary"><Icon name={ICON[a.key] ?? 'spa'} /></span>
        <h1 className="text-3xl font-semibold">{a.label}</h1>
      </div>
      <Card className="flex flex-col gap-5">
        <p className="text-xs font-medium uppercase tracking-wider text-ink-soft">How Steady scores it</p>
        <Row title="How much it helped" value={a.mean_rating ? `${a.mean_rating} / 5` : '-'}>
          <div className="flex gap-1.5">{[1, 2, 3, 4, 5].map((i) => (
            <span key={i} className={`h-2.5 flex-1 rounded-full ${a.mean_rating && i <= Math.round(a.mean_rating) ? 'bg-primary' : 'bg-surface-high'}`} />
          ))}</div>
        </Row>
        <Row title="Helped" value={`${a.helped} of ${a.tried} times`}>
          <p className="text-sm text-ink-soft">Suggested {a.suggested} times, done {a.done}, skipped {a.skipped}. Skipping is fine.</p>
        </Row>
        {a.fits_when.length > 0 && (
          <Row title="Suggested when" value="">
            <p className="text-sm text-ink-soft">{a.fits_when.slice(0, 4).join(', ')}</p>
          </Row>
        )}
        {a.history.length > 0 && (
          <Row title="Recent tries" value="">
            <div className="flex flex-wrap gap-2">{a.history.slice(-8).map((h) => (
              <span key={h.date} title={shortDate(h.date)}
                    className={`grid h-9 w-9 place-items-center rounded-full text-sm ${h.completed ? (h.rating && h.rating >= 4 ? 'bg-primary text-on-primary' : 'bg-sage-soft text-primary') : 'bg-surface-high text-outline'}`}>
                {h.completed ? h.rating ?? '✓' : '·'}
              </span>
            ))}</div>
          </Row>
        )}
      </Card>
      <p className="rounded-2xl bg-sand p-4 text-sm text-ink-soft">
        Steady suggests actions that fit what is lacking today, then learns from your answers which ones help you most. Resting also counts.
      </p>
    </div>
  )
}

function Row({ title, value, children }: { title: string; value: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-2 border-b border-surface-high pb-4 last:border-0 last:pb-0">
      <div className="flex justify-between"><span className="font-medium">{title}</span><span className="font-semibold text-primary">{value}</span></div>
      {children}
    </div>
  )
}
