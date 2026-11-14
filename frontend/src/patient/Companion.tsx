import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { engagement, useCompanion } from '../lib/companion'
import { Appie, Food, LevelProgress, LoadError, Week } from '../ui/Appie'
import { Card, Icon } from '../ui/kit'
import { PID } from './Shell'

export default function Companion() {
  const { data: c, error, refresh } = useCompanion(PID)
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState('')
  const [actionError, setActionError] = useState('')
  const [eating, setEating] = useState(false)
  const [name, setName] = useState('')
  const [editing, setEditing] = useState(false)
  const feeding = useRef(false)
  const feedRequest = useRef<{ amount: number; id: string } | null>(null)
  useEffect(() => { if (!eating) return; const timer = setTimeout(() => setEating(false), 2400); return () => clearTimeout(timer) }, [eating])
  if (!c) return error ? <LoadError error={error} retry={refresh} /> : <p className="pt-8 text-ink-soft">Appie is getting comfy…</p>

  const feed = async (amount: number) => {
    if (feeding.current) return
    feeding.current = true
    setBusy(true); setActionError(''); setNotice('')
    try {
      if (!feedRequest.current || feedRequest.current.amount !== amount) feedRequest.current = { amount, id: globalThis.crypto?.randomUUID?.() || `feed-${Date.now()}-${Math.random().toString(36).slice(2)}` }
      const r = await engagement.feed(PID, amount, feedRequest.current.id)
      feedRequest.current = null
      setEating(true)
      setNotice(r.tier_up ? `A new look! ${c.name} is now level ${r.level}.` : r.leveled_up ? `Level ${r.level}! Growing a little, together.` : `${c.name} loved those ${r.fed} little treats. Your XP stays yours.`)
      await refresh()
    } catch (e) { setActionError(e instanceof Error ? e.message : 'Could not feed Appie. Try again.') }
    finally { feeding.current = false; setBusy(false) }
  }
  const rename = async (e: React.FormEvent) => {
    e.preventDefault(); if (!name.trim()) return
    setBusy(true); setActionError('')
    try { await engagement.profile(PID, { avatar_name: name.trim() }); setEditing(false); setNotice('A lovely name. Saved.'); await refresh() }
    catch (e) { setActionError(e instanceof Error ? e.message : 'Could not save the name.') }
    finally { setBusy(false) }
  }

  return <div className="flex flex-col gap-5 pt-2">
    <div className="flex items-start justify-between"><div><p className="eyebrow">YOUR LITTLE COMPANION</p><h1 className="mt-1 text-3xl font-semibold">Meet {c.name}</h1></div><span className="rounded-full bg-sand px-3 py-2 text-sm font-semibold"><Food amount={c.food} /></span></div>
    <section className="appie-habitat relative overflow-hidden rounded-[2rem] px-6 pb-6 pt-3 text-center">
      <span className="absolute left-5 top-5 rounded-full bg-white/75 px-3 py-1.5 text-xs font-semibold text-primary">Level {c.level} · {c.title}</span>
      <Appie mood={eating ? 'eat' : c.mood} tier={c.tier} name={c.name} className={`relative mx-auto h-72 w-72 max-w-full ${eating ? 'appie-bounce' : 'appie-float'}`} />
      <p className="relative mx-auto max-w-xs text-base font-medium leading-relaxed">{c.message}</p>
      <div className="relative mt-5"><LevelProgress data={c} /></div>
    </section>
    <Card>
      <div className="flex items-center justify-between"><h2 className="text-xl font-semibold">A snack for {c.name}</h2><Food amount={c.food} className="text-sm font-semibold" /></div>
      <p className="mt-2 text-sm leading-relaxed text-ink-soft">The little things you do earn food and XP. Give {c.name} a treat. Spending food never takes away your XP or new looks.</p>
      <div className="mt-5 grid grid-cols-3 gap-2">
        {[5, 20, Math.min(c.food, 500)].map((amount, i) => <button key={i} disabled={busy || c.food < amount || amount === 0} onClick={() => feed(amount)} className={`min-h-12 rounded-full px-2 text-sm font-semibold disabled:opacity-40 ${i === 0 ? 'bg-primary text-white' : 'bg-sage-soft text-primary'}`}>{busy ? '…' : i === 2 ? c.food > 500 ? 'Feed 500' : 'Feed all' : `Feed ${amount}`}<span aria-hidden className="ml-1">🥭</span></button>)}
      </div>
      {c.food === 0 && <Link to="/habits" className="mt-4 inline-flex items-center gap-1 text-sm font-semibold text-primary">Find a little thing to do <Icon name="arrow_forward" /></Link>}
      {notice && <p role="status" className="mt-4 rounded-2xl bg-sage-soft p-3 text-sm font-medium text-primary">{notice}</p>}
      {actionError && <p role="alert" className="mt-4 text-sm text-terracotta">{actionError}</p>}
    </Card>
    <Card>
      <div className="mb-5 flex items-center justify-between"><h2 className="text-lg font-semibold">Days together</h2><span className="flex items-center gap-1 text-sm font-semibold text-primary"><Icon name="local_fire_department" /> {c.streak.current} days</span></div>
      <Week data={c} />
      <p className="mt-4 text-xs leading-relaxed text-ink-soft">Your longest stretch: {c.streak.longest} days. A rest day each week is okay. Your food, XP and new looks stay yours.</p>
    </Card>
    <div><div className="flex items-center justify-between"><h2 className="text-xl font-semibold">Room to grow</h2><span className="text-xs text-ink-soft">Little things add up</span></div>
      <div className="mt-3 grid grid-cols-3 gap-2">{[{ tier: 1, level: 1, label: 'Your new pal' }, { tier: 2, level: 3, label: 'Good vibrations' }, { tier: 3, level: 5, label: 'Adventure buddy' }].map((look) => <div key={look.tier} className={`relative overflow-hidden rounded-3xl p-3 text-center ${c.tier >= look.tier ? 'bg-sage-soft' : 'bg-surface-mid'}`}><Appie tier={look.tier} name={c.name} className={`h-24 w-full ${c.tier < look.tier ? 'opacity-55' : ''}`} /><p className="mt-2 text-xs font-semibold">{look.label}</p><p className="mt-1 text-[11px] text-ink-soft">{c.tier >= look.tier ? 'Unlocked' : `Level ${look.level}`}</p></div>)}</div>
    </div>
    <Card><h2 className="font-semibold">Today's little wins</h2>{c.earned_today.length ? <ul className="mt-3 flex flex-col gap-3">{c.earned_today.map((r) => <li key={r.reason} className="flex items-center justify-between gap-3 text-sm"><span>{r.label}</span><Food amount={r.amount} className="font-semibold text-primary" /></li>)}</ul> : <p className="mt-2 text-sm text-ink-soft">Your first little win can be a few words about your day.</p>}</Card>
    <Card>{editing ? <form onSubmit={rename}><label htmlFor="avatar-name" className="text-sm font-semibold">Your companion's name</label><input id="avatar-name" value={name} onChange={(e) => setName(e.target.value)} maxLength={16} required className="form-input mt-2" /><div className="mt-3 flex gap-2"><button disabled={busy || !name.trim()} className="button-primary flex-1">Save name</button><button type="button" onClick={() => setEditing(false)} className="button-soft flex-1">Cancel</button></div></form> : <button onClick={() => { setName(c.name); setEditing(true) }} className="flex w-full items-center justify-between text-sm"><span>Give {c.name} a nickname</span><Icon name="edit" className="text-primary" /></button>}</Card>
    <p className="px-3 text-center text-xs leading-relaxed text-ink-soft">Good days and hard days earn the same check-in reward. {c.name} is here for all of them.</p>
  </div>
}
