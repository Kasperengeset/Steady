// Notifications: what changed, newest first, crisis pinned. Open → patient detail; Mark as handled.
import { useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { views, type Notification } from '../lib/views'
import { Icon, timeAgo } from '../ui/kit'
import { KIND } from './Portal'

const TABS = [
  { key: 'open', label: 'Needs action' },
  { key: 'all', label: 'All' },
  { key: 'handled', label: 'Handled' },
] as const

export default function Notifications() {
  const [tab, setTab] = useState<'open' | 'all' | 'handled'>('open')
  const [items, setItems] = useState<Notification[] | null>(null)
  const [error, setError] = useState('')
  const load = () => views.notifications(tab).then((value) => { setItems(value); setError('') }).catch(() => setError('Could not refresh notifications. Return to this tab to retry.'))
  useEffect(() => { setItems(null); void load(); const focus = () => { void load() }; window.addEventListener('focus', focus); return () => window.removeEventListener('focus', focus) }, [tab]) // eslint-disable-line react-hooks/exhaustive-deps

  const handle = async (id: string) => { await views.markHandled(id); load() }
  const today = new Date().toISOString().slice(0, 10)
  const groups = [
    { label: 'Today', rows: (items ?? []).filter((i) => i.ts.slice(0, 10) === today) },
    { label: 'Earlier', rows: (items ?? []).filter((i) => i.ts.slice(0, 10) !== today) },
  ]

  return (
    <div className="mx-auto max-w-4xl">
      <h1 className="text-3xl font-semibold">Notifications</h1>
      {error && <p role="alert" className="mt-3 text-sm text-ink-soft">{error}</p>}
      <div className="mt-6 flex gap-2">
        {TABS.map((t) => (
          <button key={t.key} onClick={() => setTab(t.key)}
                  className={`rounded-full px-5 py-2 ${tab === t.key ? 'bg-primary text-on-primary' : 'bg-surface-mid text-ink-soft'}`}>{t.label}</button>
        ))}
      </div>
      {!items && <p className="mt-8 text-ink-soft">Loading…</p>}
      {items && items.length === 0 && <p className="mt-8 text-ink-soft">Nothing here. All patients are within their usual range.</p>}
      {groups.map((g) => g.rows.length > 0 && (
        <section key={g.label} className="mt-8">
          <p className="mb-3 text-xs font-medium uppercase tracking-wider text-ink-soft">{g.label}</p>
          <div className="flex flex-col gap-3">
            {g.rows.map((n) => (
              <div key={n.id} className={`flex items-start gap-4 rounded-3xl p-5 ${n.kind === 'crisis' && !n.handled ? 'bg-terracotta-soft' : 'bg-surface-low'}`}>
                <span className={`mt-2 h-3 w-3 shrink-0 rounded-full ${KIND[n.kind].dot}`} />
                <div className="min-w-0 flex-1">
                  <p className="flex flex-wrap items-center gap-2">
                    <span className="font-semibold">{n.patient}</span>
                    <span className={`rounded-full px-2.5 py-0.5 text-xs font-semibold ${KIND[n.kind].chip}`}>{KIND[n.kind].label}</span>
                    <span className="text-sm text-ink-soft">{timeAgo(n.ts)}</span>
                  </p>
                  <p className="mt-1">{n.message}</p>
                  {n.quote && <p className="mt-2 rounded-2xl bg-white/70 p-3 text-sm italic text-ink-soft">"{n.quote}"</p>}
                </div>
                <div className="flex shrink-0 gap-2">
                  <Link to={`/clinician/patients/${n.patient_id}`} className="flex items-center gap-1 rounded-full bg-primary px-4 py-2 text-sm font-semibold text-on-primary">
                    Open <Icon name="arrow_forward" />
                  </Link>
                  {!n.handled && n.kind !== 'baseline' && (
                    <button onClick={() => handle(n.id)} className="rounded-full bg-surface-high px-4 py-2 text-sm">Mark as handled</button>
                  )}
                </div>
              </div>
            ))}
          </div>
        </section>
      ))}
    </div>
  )
}
