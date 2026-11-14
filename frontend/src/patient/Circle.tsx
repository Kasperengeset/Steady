import { useCallback, useEffect, useState } from 'react'
import { Link } from 'react-router-dom'
import { community, type CircleFeed, type CircleSummary, type SocialProfile } from '../lib/community'
import { Appie, LoadError } from '../ui/Appie'
import { Card, Icon, timeAgo } from '../ui/kit'
import { PID } from './Shell'

function CircleBoard({ feed, presets, busy, send }: { feed: CircleFeed; presets: { kind: string; label: string }[]; busy: boolean; send: (to: number, kind: string) => void }) {
  const [recipient, setRecipient] = useState<number | null>(null)
  const selected = feed.members.find((m) => m.member_id === recipient)
  return <>
    {feed.reach_out_cards.map((card) => <p key={card.to_member_id} className="rounded-3xl bg-sand p-5 text-sm">{card.message}</p>)}
    <div className="grid grid-cols-2 gap-3">{feed.members.map((m) => <section key={m.member_id} className={`relative rounded-[1.75rem] p-4 text-center ${m.is_you ? 'border border-primary/15 bg-sage-soft/45' : 'bg-surface-low'}`}>
      {m.is_you && <span className="absolute right-3 top-2 text-[10px] font-semibold uppercase tracking-wider text-primary">You</span>}
      {m.photo_url ? <img src={m.photo_url} alt={`${m.name}'s chosen profile photo`} className="mx-auto my-4 h-24 w-24 rounded-full object-cover" referrerPolicy="no-referrer" /> : m.avatar ? <Appie name={m.avatar.name || 'Appie'} className="mx-auto h-32 w-full" /> : <span className="mx-auto my-4 grid h-24 w-24 place-items-center rounded-full bg-sand text-3xl font-semibold text-primary" aria-hidden>{m.emoji || m.name.slice(0, 1)}</span>}
      <h2 className="mt-1 truncate text-sm font-semibold">{m.name}</h2>
      {m.emoji && (m.avatar || m.photo_url) && <p className="mt-1 text-lg">{m.emoji}</p>}
      {m.level !== null && <p className="mt-1 text-[11px] text-ink-soft">Level {m.level}</p>}
      {m.streak !== null && <p className="mt-1 text-[11px] text-ink-soft">{m.streak} day{m.streak === 1 ? '' : 's'} together</p>}
      {m.checked_in_today !== null && <p className="mt-2 text-[10px] text-ink-soft">{m.checked_in_today ? 'Checked in today' : 'Taking their own time'}</p>}
      {m.level === null && m.streak === null && m.checked_in_today === null && <p className="mt-2 text-[10px] text-ink-soft">{m.role === 'friend' ? 'Here as a friend' : 'Keeping the little things private'}</p>}
      {!m.is_you && <button disabled={busy} onClick={() => setRecipient(recipient === m.member_id ? null : m.member_id)} className="mt-3 min-h-10 w-full rounded-full bg-white px-2 text-xs font-semibold text-primary">Send a little cheer</button>}
    </section>)}</div>
    {feed.members.length === 1 && <p className="px-4 text-center text-sm text-ink-soft">It starts with you. Share your invite code when you feel ready.</p>}
    {selected && <Card><div className="flex items-center justify-between"><h2 className="font-semibold">A little cheer for {selected.name}</h2><button aria-label="Close cheer choices" onClick={() => setRecipient(null)} className="grid h-9 w-9 place-items-center"><Icon name="close" /></button></div><div className="mt-3 grid gap-2">{presets.map((preset) => <button key={preset.kind} disabled={busy} onClick={() => { send(selected.member_id, preset.kind); setRecipient(null) }} className="min-h-11 rounded-2xl bg-sage-soft px-4 py-3 text-left text-sm font-medium text-primary">{preset.label}</button>)}</div><p className="mt-3 text-xs text-ink-soft">This stays inside your circle.</p></Card>}
    {feed.reactions.length > 0 && <Card><h2 className="font-semibold">Recent encouragement</h2><ul className="mt-4 flex flex-col gap-4">{feed.reactions.slice(0, 10).map((reaction) => <li key={reaction.id}><p className="text-sm">{reaction.label}</p><p className="mt-1 text-xs text-ink-soft">{reaction.from_name} → {feed.members.find((m) => m.member_id === reaction.to_member_id)?.name || 'A friend'} · {timeAgo(reaction.ts)}</p></li>)}</ul></Card>}
  </>
}

const sharing = [{ key: 'share_avatar', label: 'My companion' }, { key: 'share_level', label: 'My level' }, { key: 'share_streak', label: 'My streak' }, { key: 'share_checked_in', label: 'Whether I checked in today' }] as const

export default function Circle() {
  const [summary, setSummary] = useState<CircleSummary | null>(null)
  const [feed, setFeed] = useState<CircleFeed | null>(null)
  const [profile, setProfile] = useState<SocialProfile | null>(null)
  const [draft, setDraft] = useState<SocialProfile | null>(null)
  const [presets, setPresets] = useState<{ kind: string; label: string }[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [actionError, setActionError] = useState('')
  const [busy, setBusy] = useState(false)
  const [notice, setNotice] = useState('')
  const [code, setCode] = useState('')
  const [leaving, setLeaving] = useState(false)
  const [crisis, setCrisis] = useState(false)
  const load = useCallback(async (resetDraft = false) => {
    try {
      const [p, circle, choices] = await Promise.all([community.profile(PID), community.circle(PID), community.presets()])
      setProfile(p); setDraft((old) => resetDraft || !old ? p : old); setSummary(circle); setPresets(choices.reactions)
      setFeed(circle ? await community.feed(circle.token) : null); setError('')
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not load your circle.') }
    finally { setLoading(false) }
  }, [])
  useEffect(() => { void load() }, [load])
  const action = async (work: () => Promise<void>, success: string) => {
    setBusy(true); setActionError(''); setNotice('')
    try { await work(); setNotice(success); await load() }
    catch (e) { setActionError(e instanceof Error ? e.message : 'Could not save that. Please try again.') }
    finally { setBusy(false) }
  }
  if (loading) return <p className="pt-8 text-ink-soft">Finding your circle…</p>
  if (error || !profile || !draft) return <LoadError error={error || 'Could not load your profile.'} retry={() => { void load() }} />
  const saveProfile = (e: React.FormEvent) => {
    e.preventDefault()
    void action(async () => {
      const p = await community.saveProfile(PID, draft)
      setProfile(p); setDraft(p); setCrisis(Boolean(p.crisis))
    }, 'Your circle preferences are saved.')
  }
  return <div className="flex flex-col gap-5 pt-2">
    <div><p className="eyebrow">BETTER WITH A LITTLE COMPANY</p><h1 className="mt-1 text-3xl font-semibold">Your circle</h1><p className="mt-2 text-sm text-ink-soft">A few familiar faces. A little encouragement.</p></div>
    <div className="flex items-start gap-3 rounded-3xl bg-sage-soft/70 p-4 text-sm text-primary"><Icon name="lock" className="mt-0.5 shrink-0" /><p className="leading-relaxed">You choose what to share, item by item. Your voice, diary, scores and health data stay out of the circle.</p></div>
    {(notice || actionError) && <p role={actionError ? 'alert' : 'status'} className={`rounded-2xl p-4 text-sm ${actionError ? 'bg-sand text-terracotta' : 'bg-sage-soft text-primary'}`}>{actionError || notice}</p>}
    {crisis && <Card className="bg-sand"><h2 className="font-semibold">You deserve support.</h2><p className="mt-2 text-sm">Your words were flagged for care-team review in the app. If you need help now, contact someone you trust or local emergency services.</p><Link to="/friends" className="mt-3 inline-block text-sm font-semibold text-primary underline">See support contacts</Link></Card>}
    {!summary ? <>
      <section className="appie-habitat relative overflow-hidden rounded-[2rem] p-6 text-center"><Appie mood="wave" className="mx-auto h-48 w-48" /><h2 className="mt-1 text-xl font-semibold">A small circle, a softer landing.</h2><p className="mt-3 text-sm leading-relaxed text-ink-soft">Start a circle or use a friend's invite code. First choose how you want to appear and give permission in the settings below.</p><button disabled={busy || !profile.social_consent} onClick={() => action(async () => { await community.create(PID) }, 'Your circle is ready. Share its code with someone you trust.')} className="button-primary mt-5 w-full">Start a circle</button></section>
      <Card><h2 className="font-semibold">Have an invite code?</h2><form onSubmit={(e) => { e.preventDefault(); void action(async () => { await community.join(code.trim().toUpperCase(), { patient_id: PID }) }, 'You’re in. Say a little hello.') }} className="mt-3 flex gap-2"><label htmlFor="circle-code" className="sr-only">Circle invite code</label><input id="circle-code" value={code} onChange={(e) => setCode(e.target.value.toUpperCase())} placeholder="Invite code" autoComplete="off" maxLength={12} required className="form-input min-w-0 flex-1 font-mono uppercase tracking-widest" /><button disabled={busy || !code.trim() || !profile.social_consent} className="button-soft">Join</button></form><Link to="/join" className="mt-4 inline-block text-xs font-medium text-primary underline underline-offset-4">Joining as a friend without a patient profile?</Link></Card>
    </> : <>
      <Card className="bg-sand"><div className="flex items-center justify-between gap-3"><div><p className="text-xs text-ink-soft">{summary.name} · Invite someone you trust</p><p className="mt-1 font-mono text-2xl font-semibold tracking-[.22em]">{summary.invite_code}</p></div><button aria-label="Copy circle invite code" className="grid h-11 w-11 place-items-center rounded-full bg-white/70 text-primary" onClick={async () => { try { await navigator.clipboard.writeText(summary.invite_code); setNotice('Invite code copied.') } catch { setNotice(`Your invite code is ${summary.invite_code}.`) } }}><Icon name="content_copy" /></button></div><p className="mt-2 text-xs text-ink-soft">Friends can join at /join. Share the code yourself.</p></Card>
      {summary.nudge && <Card className="bg-sand"><h2 className="text-sm font-semibold">A gentle support card</h2><p className="mt-2 text-sm leading-relaxed text-ink-soft">{summary.nudge.message}</p>{summary.nudge.pending && <p className="mt-2 text-xs text-ink-soft">Held until {new Date(summary.nudge.visible_from).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit' })} so you have time to choose.</p>}{summary.nudge.can_cancel && <button disabled={busy} onClick={() => action(async () => { await community.cancelNudge(PID) }, 'The circle support card was cancelled. Care-team review is unchanged.')} className="button-soft mt-3 w-full text-xs">Cancel this support card</button>}</Card>}
      {feed && <CircleBoard feed={feed} presets={presets} busy={busy} send={(to, kind) => { void action(async () => { await community.react(summary.token, to, kind) }, 'A little encouragement, sent.') }} />}
    </>}
    <Card><h2 className="font-semibold">How you show up</h2><form onSubmit={saveProfile} className="mt-4 flex flex-col gap-4">
      <label className="text-xs font-semibold text-ink-soft">Your identity<select value={draft.mode} onChange={(e) => setDraft({ ...draft, mode: e.target.value as SocialProfile['mode'] })} className="form-input mt-2"><option value="anonymous">A nickname, kept separate</option><option value="profile">My chosen name and photo</option></select></label>
      {draft.mode === 'anonymous' ? <label className="text-xs text-ink-soft">Nickname<input value={draft.pseudonym || ''} onChange={(e) => setDraft({ ...draft, pseudonym: e.target.value })} maxLength={40} placeholder="Anonymous" className="form-input mt-2" /></label> : <><label className="text-xs text-ink-soft">Display name<input value={draft.display_name || ''} onChange={(e) => setDraft({ ...draft, display_name: e.target.value })} maxLength={40} placeholder="Your chosen name" className="form-input mt-2" /></label><label className="text-xs text-ink-soft">Photo URL, optional<input type="url" value={draft.photo_url || ''} onChange={(e) => setDraft({ ...draft, photo_url: e.target.value || null })} maxLength={500} placeholder="https://…" className="form-input mt-2" /></label></>}
      <label className="text-xs text-ink-soft">An emoji, optional<input value={draft.emoji || ''} onChange={(e) => setDraft({ ...draft, emoji: e.target.value || null })} maxLength={8} className="form-input mt-2" /></label>
      <label className="flex items-start gap-3 rounded-2xl bg-sand p-4"><input type="checkbox" checked={draft.social_consent} onChange={(e) => setDraft({ ...draft, social_consent: e.target.checked, ...(!e.target.checked ? { share_avatar: false, share_level: false, share_streak: false, share_checked_in: false } : {}) })} className="mt-1 h-4 w-4 accent-primary" /><span className="text-xs leading-relaxed"><strong className="block text-sm font-semibold">Let me take part in the circle</strong>I agree to the social part of Steady and choose what to share below. I can withdraw this permission anytime.</span></label>
      <fieldset disabled={!draft.social_consent} className="flex flex-col gap-3 disabled:opacity-50"><legend className="mb-3 text-xs font-semibold text-ink-soft">Only share the things you choose</legend>{sharing.map((item) => <label key={item.key} className="flex items-center gap-3 text-sm"><input type="checkbox" checked={draft[item.key]} onChange={(e) => setDraft({ ...draft, [item.key]: e.target.checked })} className="h-4 w-4 accent-primary" />{item.label}</label>)}</fieldset>
      <button disabled={busy} className="button-soft">Save circle preferences</button><p className="text-xs leading-relaxed text-ink-soft">Withdrawing permission switches every shared item off. You can also leave your circle below.</p>
    </form></Card>
    <Link to="/friends" className="flex items-center justify-between rounded-3xl bg-surface-low px-5 py-4 text-sm"><span className="flex items-center gap-2"><Icon name="diversity_1" className="text-primary" /> Your close support contacts</span><Icon name="chevron_right" /></Link>
    {summary && <div className="text-center">{leaving ? <div className="rounded-3xl bg-sand p-5"><p className="text-sm">Leave this circle? Your companion and progress stay with you.</p><div className="mt-3 flex gap-2"><button disabled={busy} onClick={() => action(async () => { await community.leave(summary.token); setSummary(null); setFeed(null); setLeaving(false) }, 'You left the circle. You can join another whenever you like.')} className="button-soft flex-1">Leave circle</button><button onClick={() => setLeaving(false)} className="button-primary flex-1">Stay</button></div></div> : <button onClick={() => setLeaving(true)} className="py-3 text-xs text-ink-soft underline underline-offset-4">Leave this circle</button>}</div>}
  </div>
}

export function FriendCircle() {
  const [token, setToken] = useState<string | null>(() => sessionStorage.getItem('steady.friend-circle'))
  const [feed, setFeed] = useState<CircleFeed | null>(null)
  const [presets, setPresets] = useState<{ kind: string; label: string }[]>([])
  const [code, setCode] = useState(() => new URLSearchParams(window.location.search).get('code') || '')
  const [name, setName] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [consent, setConsent] = useState(false)
  const load = useCallback(async () => { if (!token) return; setFeed(await community.feed(token)); setPresets((await community.presets()).reactions) }, [token])
  useEffect(() => { void load().catch((e) => setError(e instanceof Error ? e.message : 'Could not load the circle.')) }, [load])
  const run = async (work: () => Promise<void>) => { setBusy(true); setError(''); setNotice(''); try { await work() } catch (e) { setError(e instanceof Error ? e.message : 'Could not save that.') } finally { setBusy(false) } }
  return <main className="mx-auto flex min-h-full max-w-[480px] flex-col gap-5 bg-surface px-5 py-8"><div><p className="eyebrow">STEADY · A LITTLE COMPANY</p><h1 className="mt-2 text-3xl font-semibold">{feed ? feed.circle.name : 'Join a friend’s circle'}</h1><p className="mt-3 text-sm leading-relaxed text-ink-soft">Your friend chooses what to share. You will never see their diary, recordings or health data.</p></div>
    {error && <p role="alert" className="rounded-2xl bg-sand p-4 text-sm">{error}</p>}{notice && <p role="status" className="text-sm text-primary">{notice}</p>}
    {!token ? <Card><form onSubmit={(e) => { e.preventDefault(); void run(async () => { const membership = await community.join(code.trim().toUpperCase(), { name: name.trim() }); sessionStorage.setItem('steady.friend-circle', membership.token); setToken(membership.token) }) }} className="flex flex-col gap-4"><label className="text-xs text-ink-soft">Invite code<input value={code} onChange={(e) => setCode(e.target.value.toUpperCase())} required maxLength={12} className="form-input mt-2 font-mono tracking-widest" /></label><label className="text-xs text-ink-soft">Name your friends will see<input value={name} onChange={(e) => setName(e.target.value)} required maxLength={40} className="form-input mt-2" /></label><label className="flex items-start gap-3 text-xs leading-relaxed"><input type="checkbox" required checked={consent} onChange={(e) => setConsent(e.target.checked)} className="mt-1 h-4 w-4 accent-primary" />I want to join this circle and share this name with its members. My membership stays in this browser session.</label><button disabled={busy || !name.trim() || !code.trim() || !consent} className="button-primary">Join as a friend</button></form></Card> : <>{feed ? <CircleBoard feed={feed} presets={presets} busy={busy} send={(to, kind) => { void run(async () => { await community.react(token, to, kind); await load(); setNotice('A little encouragement, sent.') }) }} /> : <button onClick={() => run(load)} className="button-soft">Load circle</button>}<button disabled={busy} onClick={() => run(async () => { await community.leave(token); sessionStorage.removeItem('steady.friend-circle'); setToken(null); setFeed(null) })} className="py-3 text-xs text-ink-soft underline underline-offset-4">Leave this circle</button>{error && <button onClick={() => { sessionStorage.removeItem('steady.friend-circle'); setToken(null); setFeed(null); setError('') }} className="text-xs text-primary underline">Use a different invite</button>}</>}
    <Link to="/" className="mt-2 text-center text-xs text-primary underline underline-offset-4">Back to Steady</Link>
  </main>
}
