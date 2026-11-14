import { useState } from 'react'
import { Link } from 'react-router-dom'
import { engagement, useCompanion, type Habit } from '../lib/companion'
import { Food, LoadError, Week } from '../ui/Appie'
import { Card, Icon } from '../ui/kit'
import HealthConnection from './HealthConnection'
import PromptSettings from './PromptSettings'
import { PID } from './Shell'

export default function Habits() {
  const { data: c, error, refresh } = useCompanion(PID)
  const [label, setLabel] = useState('')
  const [busy, setBusy] = useState('')
  const [notice, setNotice] = useState('')
  const [actionError, setActionError] = useState('')
  const [editing, setEditing] = useState<string | null>(null)
  const [editLabel, setEditLabel] = useState('')
  if (!c) return error ? <LoadError error={error} retry={refresh} /> : <p className="pt-8 text-ink-soft">Gathering your little wins…</p>
  const action = async (key: string, work: () => Promise<unknown>, success: string) => {
    setBusy(key); setActionError(''); setNotice('')
    try { await work(); if (success) setNotice(success); await refresh() }
    catch (e) { setActionError(e instanceof Error ? e.message : 'Could not save that. Please try again.') }
    finally { setBusy('') }
  }
  const add = (e: React.FormEvent) => {
    e.preventDefault(); if (label.trim().length < 2) return
    void action('add', async () => { await engagement.addHabit(PID, label.trim()); setLabel('') }, 'Your habit is ready. Start whenever you like.')
  }
  const tick = (h: Habit) => action(h.key, async () => {
    const r = await engagement.tickHabit(PID, h.key)
    setNotice(r.reward ? `A little win. +${r.reward.amount} pet food for ${c.name}.` : 'Done for today. Every little thing counts.')
  }, '')
  const done = c.habits.filter((h) => h.done).length
  const renderHabit = (h: Habit) => <div key={h.key} className={`rounded-3xl border p-4 ${h.done ? 'border-primary/10 bg-sage-soft/50' : 'border-surface-high bg-white/60'}`}>
    <div className="flex items-center gap-3">
      {h.auto ? <span className={`grid h-11 w-11 shrink-0 place-items-center rounded-2xl ${h.done ? 'bg-primary text-white' : 'bg-sand text-secondary'}`}><Icon name={h.done ? 'check' : h.icon} /></span> : <button aria-label={h.done ? `${h.label}, completed today` : `Complete ${h.label}`} disabled={Boolean(busy) || h.done} onClick={() => tick(h)} className={`grid h-11 w-11 shrink-0 place-items-center rounded-2xl border ${h.done ? 'border-primary bg-primary text-white' : 'border-outline-soft bg-surface text-primary'}`}><Icon name={h.done ? 'check' : 'add'} /></button>}
      <div className="min-w-0 flex-1"><p className="text-sm font-semibold">{h.label}</p><p className="mt-1 text-xs text-ink-soft">{h.done ? 'A little win, done.' : h.hint}</p></div>
      <Food amount={h.food} className="shrink-0 text-xs font-semibold text-primary" />
      {!h.auto && <button disabled={Boolean(busy)} aria-label={`Edit ${h.label}`} onClick={() => { setEditing(editing === h.key ? null : h.key); setEditLabel(h.label) }} className="grid h-9 w-8 shrink-0 place-items-center text-outline"><Icon name="more_horiz" /></button>}
    </div>
    {h.auto && !h.done && (h.key === 'memo' || h.key === 'action') && <Link to={h.key === 'memo' ? '/checkin' : '/'} className="ml-14 mt-2 inline-flex items-center gap-1 text-xs font-semibold text-primary">{h.key === 'memo' ? 'Check in' : 'See today’s action'}<Icon name="arrow_forward" /></Link>}
    {editing === h.key && <form onSubmit={(e) => { e.preventDefault(); void action(h.key, async () => { await engagement.renameHabit(PID, h.key, editLabel.trim()); setEditing(null) }, 'Habit updated.') }} className="mt-3 border-t border-primary/10 pt-3"><label htmlFor={`edit-${h.key}`} className="sr-only">Habit name</label><input id={`edit-${h.key}`} className="form-input" value={editLabel} onChange={(e) => setEditLabel(e.target.value)} minLength={2} maxLength={40} required /><div className="mt-2 flex gap-2"><button disabled={Boolean(busy) || editLabel.trim().length < 2} className="button-primary flex-1 text-xs">Save</button><button type="button" disabled={Boolean(busy)} onClick={() => action(h.key, async () => { await engagement.removeHabit(PID, h.key); setEditing(null) }, 'Habit removed. Your earned food stays yours.')} className="button-soft flex-1 text-xs">Remove habit</button></div></form>}
  </div>
  return <div className="flex flex-col gap-5 pt-2">
    <div><p className="eyebrow">SMALL THINGS, REAL PROGRESS</p><div className="mt-1 flex items-center justify-between"><h1 className="text-3xl font-semibold">Your daily rhythm</h1><span className="rounded-full bg-sage-soft px-3 py-2 text-xs font-semibold text-primary">{done} of {c.habits.length}</span></div><p className="mt-2 text-sm text-ink-soft">Make a little room for what makes you, you.</p></div>
    <Card><Week data={c} /></Card>
    {(notice || actionError) && <p role={actionError ? 'alert' : 'status'} className={`rounded-2xl p-4 text-sm ${actionError ? 'bg-sand text-terracotta' : 'bg-sage-soft text-primary'}`}>{actionError || notice}</p>}
    <div><h2 className="mb-3 flex items-center gap-2 text-sm font-semibold text-ink-soft"><Icon name="auto_awesome" /> The everyday essentials</h2><div className="flex flex-col gap-2.5">{c.habits.filter((h) => h.auto).map(renderHabit)}</div></div>
    <div><h2 className="mb-3 text-lg font-semibold">A little time for you</h2>{c.habits.some((h) => !h.auto) ? <div className="flex flex-col gap-2.5">{c.habits.filter((h) => !h.auto).map(renderHabit)}</div> : <p className="rounded-3xl border border-dashed border-outline-soft p-5 text-sm leading-relaxed text-ink-soft">Reading a few pages, playing guitar, watering your plants. Add something you enjoy.</p>}
      <form onSubmit={add} className="mt-3 rounded-3xl bg-surface-low p-4"><label htmlFor="new-habit" className="text-xs font-semibold text-ink-soft">Add your own habit</label><div className="mt-2 flex gap-2"><input id="new-habit" value={label} onChange={(e) => setLabel(e.target.value)} placeholder="e.g. A little sketching" required minLength={2} maxLength={40} className="form-input min-w-0 flex-1" /><button aria-label="Add habit" disabled={Boolean(busy) || label.trim().length < 2} className="grid h-12 w-12 shrink-0 place-items-center rounded-2xl bg-primary text-white disabled:opacity-40"><Icon name="add" /></button></div><p className="mt-3 text-xs text-ink-soft">+2 food for each of your first 3 custom habits a day.</p></form>
    </div>
    <HealthConnection />
    <PromptSettings />
    <p className="px-2 text-center text-xs leading-relaxed text-ink-soft">Hobbies are yours to choose and check off. Wearable habits use available health data. No need to do everything.</p>
  </div>
}
