import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { api, type CheckInResponse } from '../lib/api'
import { promptTitle, useCompanion } from '../lib/companion'
import { useRecorder } from '../hooks/useRecorder'
import { Card, Icon, toneStyle } from '../ui/kit'
import { PID } from './Shell'

const DEFAULT_PHRASE = 'When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. The rainbow is a division of white light into many beautiful colors.'
const newRequestId = () => globalThis.crypto?.randomUUID?.() ?? `checkin-${Date.now()}-${Math.random().toString(36).slice(2)}`
const slot = () => (new Date().getHours() < 14 ? 'morning' : 'evening') as 'morning' | 'evening'
const primaryButton = 'flex min-h-14 w-full items-center justify-center gap-2 rounded-full bg-primary px-5 py-3 text-base font-semibold text-on-primary disabled:cursor-wait disabled:opacity-50'
const secondaryButton = 'flex min-h-12 items-center justify-center gap-2 rounded-full bg-surface-high px-5 py-3 font-medium text-primary disabled:opacity-50'

function toneOf(result: CheckInResponse): string | null {
  if (result.baseline.status === 'building') return null
  const deviation = result.state.branches.text ?? result.state.branches.acoustic
  if (deviation === undefined || deviation === null) return null
  return deviation >= 1 ? 'Quieter' : deviation <= -1 ? 'Lifted' : 'Steady'
}

function useAudioUrl(blob: Blob | null) {
  const [url, setUrl] = useState<string | undefined>()
  useEffect(() => {
    if (!blob) { setUrl(undefined); return }
    const next = URL.createObjectURL(blob)
    setUrl(next)
    return () => URL.revokeObjectURL(next)
  }, [blob])
  return url
}

export default function CheckIn() {
  const { data: companion } = useCompanion(PID)
  const [mode, setMode] = useState<'voice' | 'text'>('voice')
  const [voiceStep, setVoiceStep] = useState<'phrase' | 'memo'>('phrase')
  const rec = useRecorder(voiceStep === 'phrase' ? 15_000 : 30_000)
  const [phrase, setPhrase] = useState<Blob | null>(null)
  const [memo, setMemo] = useState<Blob | null>(null)
  const [retaking, setRetaking] = useState(false)
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<CheckInResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const mounted = useRef(true)
  const saving = useRef(false)
  const voiceRequestId = useRef<string | null>(null)
  const textRequestId = useRef<string | null>(null)
  const phraseUrl = useAudioUrl(phrase)
  const memoUrl = useAudioUrl(memo)
  const name = companion?.name ?? 'Appie'
  const dailyPhrase = companion?.daily_phrase || DEFAULT_PHRASE

  useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false }
  }, [])

  const send = async (request: () => Promise<CheckInResponse>) => {
    if (saving.current) return
    saving.current = true
    setBusy(true)
    setError(null)
    try {
      const saved = await request()
      window.dispatchEvent(new Event('steady:engagement'))
      if (mounted.current) { setResult(saved); setMemo(null); setPhrase(null) }
    } catch {
      if (mounted.current) setError('We could not save that yet. Your check-in is still here. Check your connection and try Save again.')
    } finally {
      saving.current = false
      if (mounted.current) setBusy(false)
    }
  }

  const record = async () => {
    if (rec.recording) { rec.stop(); return }
    if (rec.starting || rec.preparing || busy) return
    const step = voiceStep
    setError(null)
    try {
      const blob = await rec.start()
      if (!mounted.current) return
      if (step === 'phrase') { setPhrase(blob); setVoiceStep('memo') }
      else { voiceRequestId.current = null; setMemo(blob); setRetaking(false) }
    } catch { /* The recorder displays permission, quality and browser errors itself. */ }
  }

  const changeMode = (next: 'voice' | 'text') => {
    rec.cancel()
    setError(null)
    setRetaking(false)
    setMode(next)
  }

  if (result?.crisis) return <Support />
  if (result) {
    const tone = toneOf(result)
    const rewards = result.rewards ?? []
    const food = rewards.reduce((sum, reward) => sum + reward.amount, 0)
    return (
      <div className="flex flex-col items-center gap-5 pt-8 text-center">
        <img src="/appie/happy.webp" alt={`${name} smiling beside your saved check-in`} className="h-48 w-48 object-contain" />
        <span className="grid h-12 w-12 place-items-center rounded-full bg-sage-soft text-primary"><Icon name="check" className="text-2xl" /></span>
        <h1 className="text-[28px] font-semibold">A little moment, saved.</h1>
        <p className="max-w-xs text-ink-soft">Thanks for showing up. Every kind of day belongs here.</p>
        <p className="max-w-xs text-xs leading-relaxed text-ink-soft">Steady follows your rhythm over days. One brighter check-in helps, while the recent week still counts.</p>
        {food > 0 && <Card className="w-full bg-sand text-left">
          <p className="flex items-center justify-between font-semibold"><span>Food for {name}</span><span className="text-xl text-primary">+{food} 🥭</span></p>
          <ul className="mt-3 space-y-2 text-sm text-ink-soft">
            {rewards.map((reward) => <li key={reward.reason} className="flex justify-between gap-3"><span>{reward.label}</span><span>+{reward.amount}</span></li>)}
          </ul>
        </Card>}
        {tone ? <p className="text-sm text-ink-soft">Today sounds <span className={`ml-1 rounded-full px-3 py-1 font-semibold ${toneStyle[tone]}`}>{tone}</span></p>
          : <p className="text-sm text-ink-soft">Steady is still learning your usual.</p>}
        <Link to="/appie" className={primaryButton}>Visit {name}<Icon name="arrow_forward" /></Link>
        <Link to="/" className="py-2 text-primary underline underline-offset-4">Back to today</Link>
      </div>
    )
  }

  const words = text.trim() ? text.trim().split(/\s+/).length : 0
  const capturing = rec.recording || rec.starting || rec.preparing
  const secondsLeft = Math.max(0, Math.ceil((rec.maxMs - rec.elapsed) / 1000))
  const radius = 82, circumference = 2 * Math.PI * radius
  const review = memo !== null && !retaking

  return (
    <div className="flex flex-col gap-5 pt-5">
      <div className="text-center">
        <p className="mb-2 text-xs font-semibold uppercase tracking-[0.16em] text-primary">A moment for you</p>
        <h1 className="text-[28px] font-semibold leading-tight">How is your day, really?</h1>
        <p className="mt-2 text-sm text-ink-soft">No perfect answers. Just you and {name}.</p>
      </div>

      {companion && <div className="flex items-start gap-3 rounded-2xl bg-sage-soft px-4 py-3 text-sm text-primary">
        <Icon name={companion.prompt.open ? 'notifications_active' : 'schedule'} className="mt-0.5 shrink-0" />
        <div>
          <p className="font-semibold">{promptTitle(companion.prompt)}</p>
          <p className="mt-0.5 text-xs leading-relaxed">{companion.prompt.bonus_available ? `There is a little bonus food for checking in in the next ${companion.prompt.minutes_left} min.` : 'You can check in whenever it works for you. Every day counts.'}</p>
        </div>
      </div>}

      <div className="grid grid-cols-2 rounded-full bg-surface-high p-1" role="group" aria-label="Check-in format">
        {(['voice', 'text'] as const).map((choice) => <button key={choice} onClick={() => changeMode(choice)} disabled={busy}
          aria-pressed={mode === choice} className={`flex min-h-11 items-center justify-center gap-2 rounded-full text-sm font-semibold ${mode === choice ? 'bg-surface-low text-primary shadow-sm' : 'text-ink-soft'}`}>
          <Icon name={choice === 'voice' ? 'mic' : 'edit_note'} />{choice === 'voice' ? 'Voice memo' : 'Write instead'}
        </button>)}
      </div>

      {mode === 'voice' ? <>
        <ol className="flex items-center justify-center gap-4 text-xs text-ink-soft" aria-label="Voice check-in steps">
          {[{ label: 'Read a line', done: !!phrase, current: !review && voiceStep === 'phrase' }, { label: 'Your day', done: review, current: !review && voiceStep === 'memo' }, { label: 'Save', done: false, current: review }].map((step, index) => (
            <li key={step.label} aria-current={step.current ? 'step' : undefined} className={`flex items-center gap-1.5 ${step.current ? 'font-semibold text-primary' : ''}`}>
              <span className={`grid h-5 w-5 place-items-center rounded-full ${step.current || step.done ? 'bg-sage-soft text-primary' : 'bg-surface-high'}`}>{step.done ? '✓' : index + 1}</span>{step.label}
            </li>
          ))}
        </ol>

        {review ? <Card className="flex flex-col gap-4">
          <div className="flex items-center gap-3"><span className="grid h-11 w-11 place-items-center rounded-full bg-sage-soft text-primary"><Icon name="graphic_eq" /></span><div><h2 className="font-semibold">Your voice memo is ready</h2><p className="text-sm text-ink-soft">Listen back, then save when you are ready.</p></div></div>
          {memoUrl && <audio controls src={memoUrl} className="w-full" aria-label="Listen to your voice memo" />}
          {phraseUrl && <details className="rounded-2xl bg-surface px-4 py-3 text-sm"><summary className="cursor-pointer font-medium text-primary">Your daily phrase is included</summary><audio controls src={phraseUrl} className="mt-3 w-full" aria-label="Listen to your daily phrase" /></details>}
          <button onClick={() => { if (memo) void send(() => {
            voiceRequestId.current ??= newRequestId()
            return api.checkIn(PID, slot(), memo, phrase ?? undefined, voiceRequestId.current)
          }) }} disabled={busy} className={primaryButton}><Icon name={busy ? 'hourglass_top' : 'check'} />{busy ? 'Saving your check-in…' : 'Save my check-in'}</button>
          <button onClick={() => { setRetaking(true); setError(null); setVoiceStep('memo') }} disabled={busy} className={secondaryButton}><Icon name="replay" />Record the memo again</button>
          <button onClick={() => { setMemo(null); setPhrase(null); setVoiceStep('phrase'); voiceRequestId.current = null; setError(null) }} disabled={busy} className="min-h-10 text-sm text-ink-soft underline underline-offset-4">Discard this voice check-in</button>
          <p className="text-center text-xs leading-relaxed text-ink-soft">Your recording stays in this tab until you save. Closing the tab discards it.</p>
        </Card> : <Card className="flex flex-col items-center gap-4 py-6 text-center">
          <h2 className="text-lg font-semibold">{voiceStep === 'phrase' ? 'First, read this phrase aloud' : 'Now, tell Appie about your day'}</h2>
          {voiceStep === 'phrase' ? <blockquote className="rounded-2xl bg-sand px-5 py-4 text-lg leading-relaxed text-ink">“{dailyPhrase}”</blockquote>
            : <p className="max-w-xs text-sm leading-relaxed text-ink-soft">What did you enjoy? What felt difficult? Thirty seconds is plenty, and a shorter memo counts too.</p>}
          <p className="text-xs text-ink-soft">{voiceStep === 'phrase' ? 'The same words each day help us compare your voice. This step is optional.' : 'Find a quiet spot and keep your phone at a comfortable, similar distance.'}</p>
          <button onClick={() => { void record() }} disabled={busy || rec.starting || rec.preparing}
            aria-label={rec.recording ? 'Finish recording' : voiceStep === 'phrase' ? 'Record daily phrase' : 'Record voice memo'}
            className="relative grid h-48 w-48 place-items-center rounded-full focus-visible:outline-2 focus-visible:outline-offset-4 focus-visible:outline-primary disabled:opacity-60">
            <svg aria-hidden className="absolute inset-0 -rotate-90" viewBox="0 0 180 180">
              <circle cx="90" cy="90" r={radius} fill="#dfeee3" stroke="#e8e8e4" strokeWidth="5" />
              <circle cx="90" cy="90" r={radius} fill="none" stroke="#43654f" strokeWidth="5" strokeLinecap="round"
                strokeDasharray={circumference} strokeDashoffset={rec.recording ? circumference * (1 - rec.elapsed / rec.maxMs) : 0} />
            </svg>
            <span className={`relative grid h-24 w-24 place-items-center rounded-full text-on-primary ${rec.recording ? 'bg-terracotta' : 'bg-primary'}`}><Icon name={rec.recording ? 'stop' : rec.starting ? 'hourglass_top' : 'mic'} fill className="text-4xl" /></span>
          </button>
          <p aria-live="polite" className="font-semibold">{rec.preparing ? 'Preparing your recording…' : rec.starting ? 'Waiting for microphone access…' : rec.recording ? `${secondsLeft} seconds left. Tap to finish.` : 'Tap to record'}</p>
          {(rec.recording || rec.starting) && <button onClick={rec.cancel} className="min-h-10 text-sm text-ink-soft underline underline-offset-4">Cancel recording</button>}
          {voiceStep === 'phrase' && <button onClick={() => setVoiceStep('memo')} disabled={capturing} className="min-h-10 text-sm text-primary underline underline-offset-4 disabled:opacity-40">Skip the phrase this time</button>}
          {phraseUrl && voiceStep === 'memo' && <p className="flex items-center gap-1 text-xs text-primary"><Icon name="check_circle" />Daily phrase captured</p>}
          {memo && retaking && !capturing && <button onClick={() => setRetaking(false)} className="min-h-10 text-sm text-primary underline underline-offset-4">Keep my previous recording</button>}
          {rec.error && <p role="alert" className="max-w-xs text-sm text-terracotta">{rec.error}</p>}
        </Card>}
      </> : <Card className="flex flex-col gap-4">
        <label htmlFor="daily-diary" className="text-lg font-semibold">A few lines about today</label>
        <textarea id="daily-diary" value={text} onChange={(event) => { textRequestId.current = null; setText(event.target.value) }} rows={7} maxLength={10000} disabled={busy}
          placeholder="What has been on your mind? A small moment, something you enjoyed, or something that felt hard."
          className="w-full resize-none rounded-2xl bg-surface p-4 text-base leading-relaxed outline-none focus:ring-2 focus:ring-primary/40 disabled:opacity-60" />
        <p className="text-xs text-ink-soft">{words} words · English works best for the text model. A written check-in of 20 words earns the daily reward; shorter entries are welcome too.</p>
        <button onClick={() => { void send(() => {
          textRequestId.current ??= newRequestId()
          return api.checkInText(PID, slot(), text.trim(), textRequestId.current)
        }) }} disabled={busy || words < 3} className={primaryButton}>{busy ? 'Saving your check-in…' : 'Save my check-in'}</button>
      </Card>}

      {error && <p role="alert" className="rounded-2xl bg-terracotta-soft px-4 py-3 text-sm text-terracotta">{error}</p>}
      {mode === 'voice' && <p className="text-center text-xs text-ink-soft">At least 10 seconds of speech earns the daily check-in reward, once each day. The phrase is separate.</p>}
      <p className="flex items-start justify-center gap-2 pb-2 text-center text-xs leading-relaxed text-ink-soft"><Icon name="lock" className="shrink-0 text-base" />{mode === 'voice' ? 'Saved recordings are analysed, then deleted. Your transcript and measurements stay in your care record.' : 'Your words stay in your care record. Your circle never sees your diary.'}</p>
    </div>
  )
}

function Support() {
  return (
    <div className="flex flex-col gap-6 pt-12 text-center">
      <span className="mx-auto grid h-16 w-16 place-items-center rounded-full bg-terracotta-soft text-terracotta"><Icon name="favorite" fill className="text-3xl" /></span>
      <h1 className="text-[28px] font-semibold leading-tight">Thank you for telling us.</h1>
      <p className="text-lg text-ink-soft">Your check-in is saved and flagged for your care team to review. You deserve support from someone you trust.</p>
      <Card className="text-left text-sm leading-relaxed"><p>If you feel in immediate danger, call your local emergency number or go to the nearest emergency department.</p><p className="mt-3 text-ink-soft">Steady does not contact emergency services or guarantee when your care team will respond. Please reach out to someone directly if you need help now.</p></Card>
      <Link to="/friends" className={primaryButton}><Icon name="group" />Find my support contacts</Link>
      <Link to="/" className="text-primary underline underline-offset-4">Back to today</Link>
    </div>
  )
}
