import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { api } from '../lib/api'
import { promptTitle, useCompanion } from '../lib/companion'
import { views, type Today as TodayData } from '../lib/views'
import { Appie, Food, LevelProgress, LoadError } from '../ui/Appie'
import { Card, Icon, LineChart } from '../ui/kit'
import { PID } from './Shell'

function greeting() { const hour = new Date().getHours(); return hour < 12 ? 'Good morning' : hour < 18 ? 'Good afternoon' : 'Good evening' }

export default function Today() {
  const { data: c, error: companionError, refresh: refreshCompanion } = useCompanion(PID)
  const [t, setT] = useState<TodayData | null>(null)
  const [error, setError] = useState('')
  const [started, setStarted] = useState(false)
  const [saving, setSaving] = useState(false)
  const [notice, setNotice] = useState('')
  const [actionError, setActionError] = useState('')
  const load = useCallback(async () => { try { setT(await views.today(PID)); setError('') } catch { setError('Steady could not load your day. Check your connection and try again.') } }, [])
  useEffect(() => { void load() }, [load])
  if (!t || !c) return error || companionError ? <LoadError error={error || companionError!} retry={() => { void load(); void refreshCompanion() }} /> : <div className="pt-12"><div className="h-5 w-28 animate-pulse rounded bg-sage-soft" /><div className="mt-4 h-9 w-64 animate-pulse rounded-xl bg-surface-high" /><div className="mt-6 h-80 animate-pulse rounded-[2rem] bg-sage-soft/60" /><p className="mt-5 text-center text-sm text-ink-soft">A little moment for you…</p></div>
  const date = new Date(`${t.date}T12:00:00`).toLocaleDateString('en-GB', { weekday: 'long', day: 'numeric', month: 'short' })
  const a = t.action
  const building = t.status === 'building'
  const b = t.baseline
  const done = c.habits.filter((h) => h.done).length
  const momentText = c.streak.today ? `Thanks for letting ${c.name} in. Good days, quiet days, all of them count.` : c.prompt.bonus_available ? `${c.prompt.minutes_left} minutes left for +5 bonus food. Later is okay too.` : `A voice memo or a few written lines. ${promptTitle(c.prompt)}.`
  const notToday = async () => {
    if (!a) return
    setSaving(true); setActionError('')
    try {
      const r = await api.feedback(PID, a.date, false, null)
      const food = r.rewards?.reduce((n, x) => n + x.amount, 0) ?? 0
      setNotice(food ? `Thanks for checking in. +${food} pet food.` : 'Thanks for letting us know. Another day is okay.')
      await Promise.all([load(), refreshCompanion()])
    } catch { setActionError('Could not save your answer. Please try again.') }
    finally { setSaving(false) }
  }
  return <div className="flex flex-col gap-5 pt-1">
    <div><p className="eyebrow">{date}</p><h1 className="mt-1 text-[29px] font-semibold leading-tight tracking-tight">{greeting()}, {t.patient.name}<span className="text-primary">.</span></h1><p className="mt-2 text-sm text-ink-soft">A little care goes a long way.</p></div>
    <div className="grid grid-cols-3 gap-2 rounded-2xl bg-surface-low p-3 text-center"><Link to="/appie" className="border-r border-outline-soft/40"><span className="flex items-center justify-center gap-1 text-base font-semibold text-primary"><Icon name="local_fire_department" />{c.streak.current}</span><span className="mt-0.5 block text-[10px] text-ink-soft">days together</span></Link><Link to="/appie" className="border-r border-outline-soft/40"><Food amount={c.food} className="text-base font-semibold text-primary" /><span className="mt-0.5 block text-[10px] text-ink-soft">pet food</span></Link><Link to="/habits"><span className="flex items-center justify-center gap-1 text-base font-semibold text-primary"><Icon name="task_alt" />{done}/{c.habits.length}</span><span className="mt-0.5 block text-[10px] text-ink-soft">little wins today</span></Link></div>
    <section className="appie-habitat relative overflow-hidden rounded-[2rem] px-5 pb-5 pt-4">
      <div className="relative z-[1] flex items-center justify-between"><span className="rounded-full bg-white/75 px-3 py-1.5 text-[11px] font-semibold text-primary">{c.name}'s little corner</span><Link to="/appie" aria-label={`Visit ${c.name}`} className="grid h-8 w-8 place-items-center rounded-full bg-white/60 text-primary"><Icon name="north_east" className="text-base" /></Link></div>
      <Link to="/appie" aria-label={`Spend time with ${c.name}`} className="relative block"><Appie mood={c.mood} tier={c.tier} name={c.name} className="appie-float mx-auto -mb-1 -mt-3 h-60 w-64 max-w-full" /></Link>
      <div className="relative rounded-[1.4rem] bg-white/80 px-5 py-4 text-center shadow-[0_4px_20px_#43654f05]"><p className="text-sm font-medium leading-relaxed">“{c.message}”</p></div>
      <div className="relative mt-4"><LevelProgress data={c} /></div>
    </section>
    <section className={`relative overflow-hidden rounded-[1.75rem] p-5 ${c.streak.today ? 'bg-surface-low' : 'bg-primary text-white'}`}>
      <div className="flex items-start gap-3"><span className={`grid h-10 w-10 shrink-0 place-items-center rounded-2xl ${c.streak.today ? 'bg-sage-soft text-primary' : 'bg-white/15'}`}><Icon name={c.streak.today ? 'check' : 'graphic_eq'} /></span><div className="flex-1"><p className={`text-[10px] font-semibold uppercase tracking-[.14em] ${c.streak.today ? 'text-ink-soft' : 'text-white/70'}`}>{c.streak.today ? 'A MOMENT, TAKEN' : c.prompt.bonus_available ? 'YOUR DAILY MOMENT IS HERE' : 'THIRTY SECONDS FOR YOU'}</p><h2 className="mt-1 text-lg font-semibold">{c.streak.today ? 'You showed up today.' : 'How is your day, really?'}</h2><p className={`mt-2 text-xs leading-relaxed ${c.streak.today ? 'text-ink-soft' : 'text-white/80'}`}>{momentText}</p></div></div>
      <Link to="/checkin" className={`mt-4 flex min-h-12 items-center justify-center gap-2 rounded-full text-sm font-semibold ${c.streak.today ? 'bg-sage-soft text-primary' : 'bg-white text-primary'}`}><Icon name={c.streak.today ? 'add' : 'mic'} />{c.streak.today ? 'Another little check-in' : 'Check in with Appie'}{!c.streak.today && <span className="ml-1 rounded-full bg-sand px-2 py-1 text-[10px]">+10 🥭 check-in</span>}</Link>
      {!c.streak.today && <p className="mt-3 text-center text-[10px] text-white/75">One reward each day: 10 seconds of speech or 20 written words.</p>}
    </section>
    {a && !building && <Card className="bg-sand/75"><p className="flex items-center gap-2 text-xs font-medium text-ink-soft"><Icon name="wb_sunny" /> ONE SMALL THING</p>{a.completed === false ? <p className="mt-3 text-lg font-semibold">Another day is okay.</p> : a.completed ? <><p className="mt-3 text-lg font-semibold">{a.label}, done.</p><p className="mt-2 text-sm text-ink-soft">A little win to take with you.</p></> : <><h2 className="mt-3 text-xl font-semibold">{a.label}</h2><p className="mt-2 text-sm leading-relaxed text-ink-soft">{started ? 'Take your time. Let us know how it went when you’re ready.' : a.reason}</p><div className="mt-4 flex gap-2">{started ? <Link to="/followup" className="button-primary flex-1 text-sm">How did it go?</Link> : <button onClick={() => setStarted(true)} className="button-primary flex-1 text-sm">Let's do it <Icon name="arrow_forward" /></button>}<button onClick={notToday} disabled={saving} className="button-soft text-xs">{saving ? 'Saving…' : 'Not today'}</button></div></>}{(notice || actionError) && <p role={actionError ? 'alert' : 'status'} className="mt-3 text-xs text-ink-soft">{actionError || notice}</p>}</Card>}
    <div className="grid grid-cols-2 gap-3"><Link to="/habits" className="rounded-3xl bg-surface-low p-5"><Icon name="favorite" className="text-primary" /><h2 className="mt-3 text-sm font-semibold">Make time for you</h2><p className="mt-1 text-xs leading-relaxed text-ink-soft">Hobbies, movement, little rituals.</p><p className="mt-3 text-xs font-semibold text-primary">Your habits <span aria-hidden>↗</span></p></Link><Link to="/circle" className="rounded-3xl bg-surface-low p-5"><Icon name="group" className="text-primary" /><h2 className="mt-3 text-sm font-semibold">A little company</h2><p className="mt-1 text-xs leading-relaxed text-ink-soft">Send someone a little cheer.</p><p className="mt-3 text-xs font-semibold text-primary">Your circle <span aria-hidden>↗</span></p></Link></div>
    <details className="rounded-[1.75rem] border border-surface-high p-5"><summary className="flex cursor-pointer list-none items-center justify-between text-sm font-semibold"><span className="flex items-center gap-2"><Icon name="monitor_heart" className="text-primary" /> Your personal rhythm</span><Icon name="expand_more" className="text-outline" /></summary><div className="pt-4"><h2 className="text-lg font-semibold">{t.status_text}</h2>{building && b ? <><p className="mt-3 text-sm text-ink-soft">Day {Math.max(1, Math.ceil(b.days_elapsed))} of {b.period_days}. Every check-in helps Steady learn your usual.</p><div className="mt-3 h-2 overflow-hidden rounded-full bg-surface-high"><div className="h-full rounded-full bg-primary" style={{ width: `${Math.min(100, b.days_elapsed / b.period_days * 100)}%` }} /></div></> : <><div className="mt-4"><LineChart series={[{ values: t.series.map((p) => p.S), color: '#43654f' }]} band={[-1, 1]} height={110} /></div><div className="mt-2 flex justify-between text-[10px] text-ink-soft"><span>2 weeks ago</span><span>Your own usual, over time</span><span>Today</span></div></>}<Link to="/insights" className="mt-4 inline-flex items-center gap-1 text-xs font-semibold text-primary">See what helps you <Icon name="arrow_forward" /></Link></div></details>
    <p className="pb-1 text-center text-[11px] text-outline">Growing gently, one day at a time.</p>
  </div>
}
