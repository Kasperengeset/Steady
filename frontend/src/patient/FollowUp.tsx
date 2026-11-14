// Follow-up: did you do it, did it help? The answer is the bandit's reward (engine/bandit.py).
import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { api } from '../lib/api'
import { views, type Today } from '../lib/views'
import { Card, Icon } from '../ui/kit'
import { PID } from './Shell'

const SCALE = ['Not really', 'A little', 'Somewhat', 'Quite a bit', 'A lot']

export default function FollowUp() {
  const nav = useNavigate()
  const [t, setT] = useState<Today | null>(null)
  const [done, setDone] = useState<boolean | null>(null)
  const [rating, setRating] = useState<number | null>(null)
  const [saved, setSaved] = useState(false)

  useEffect(() => { views.today(PID).then(setT) }, [])
  const p = t?.pending_followup
  if (!t) return <p className="pt-8 text-ink-soft">Loading…</p>
  if (!p) return (
    <div className="flex flex-col items-center gap-4 pt-16 text-center">
      <h1 className="text-2xl font-semibold">Nothing to follow up right now.</h1>
      <Link to="/" className="text-primary underline underline-offset-4">Back to today</Link>
    </div>
  )

  const save = async () => {
    await api.feedback(PID, p.date, Boolean(done), done ? rating : null)
    setSaved(true)
    setTimeout(() => nav('/'), 1200)
  }
  if (saved) return (
    <div className="flex flex-col items-center gap-4 pt-20 text-center">
      <span className="grid h-16 w-16 place-items-center rounded-full bg-sage-soft text-primary"><Icon name="eco" className="text-3xl" /></span>
      <h1 className="text-2xl font-semibold">Thanks. Steady will remember what helps you.</h1>
    </div>
  )

  return (
    <div className="flex flex-col gap-5 pt-4">
      <Card className="p-6">
        <h1 className="text-[26px] font-semibold leading-tight">Earlier you planned: {p.label.toLowerCase()}. Did you do it?</h1>
        <p className="mt-2 text-ink-soft">Whatever happened is okay.</p>
        <div className="mt-5 grid grid-cols-2 gap-3">
          <button onClick={() => setDone(true)} className={`flex h-14 items-center justify-center gap-2 rounded-full text-lg font-semibold ${done === true ? 'bg-primary text-on-primary' : 'bg-surface-high'}`}>
            <Icon name="check_circle" /> Yes
          </button>
          <button onClick={() => { setDone(false); setRating(null) }} className={`flex h-14 items-center justify-center gap-2 rounded-full text-lg ${done === false ? 'bg-primary font-semibold text-on-primary' : 'bg-surface-high'}`}>
            <Icon name="schedule" /> Not today
          </button>
        </div>
      </Card>

      {done && (
        <Card className="p-6">
          <h2 className="text-xl font-semibold">Did it help?</h2>
          <div className="mt-4 grid grid-cols-5 gap-2">
            {SCALE.map((label, i) => (
              <button key={label} onClick={() => setRating(i + 1)}
                      className={`flex h-20 flex-col items-center justify-center gap-1 rounded-2xl text-xs ${rating === i + 1 ? 'bg-primary text-on-primary' : 'bg-surface-high text-ink-soft'}`}>
                <span className="text-xl font-semibold">{i + 1}</span>{label}
              </button>
            ))}
          </div>
        </Card>
      )}

      <button onClick={save} disabled={done === null || (done && rating === null)}
              className="flex h-14 items-center justify-center gap-2 rounded-full bg-primary text-lg font-semibold text-on-primary disabled:opacity-40">
        <Icon name="check" /> Save
      </button>
      <Link to="/" className="text-center text-ink-soft">Skip for now</Link>
    </div>
  )
}
