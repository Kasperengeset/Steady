import { useState } from 'react'
import { engagementApi, type Earned, type HabitWeek, type Rewards } from '../../lib/api_engagement'
import AvatarSlot from './AvatarSlot'
import CompanionCard, { CompanionDemoButton } from './CompanionCard'
import { DEMO_PATIENT_ID, useLoad } from './useLoad'
import './engagement.css'

const WEEKDAY = new Intl.DateTimeFormat('en', { weekday: 'narrow' })

/** Habits (#24): the week of built-in habits, points, the gentle streak, and today's small thing. */
export default function HabitsPage({ patientId = DEMO_PATIENT_ID }: { patientId?: number }) {
  const rewards = useLoad(() => engagementApi.rewards(patientId), { refreshOnFocus: true })
  const week = useLoad(() => engagementApi.week(patientId), { refreshOnFocus: true })
  const [companionKey, setCompanionKey] = useState(0)
  const refresh = () => {
    rewards.reload()
    week.reload()
  }

  return (
    <section className="view eng">
      <h1>Habits</h1>
      <CompanionCard key={companionKey} patientId={patientId} onCheckedIn={refresh} />
      {rewards.data && <Summary r={rewards.data} />}
      {week.data && <TodayAction w={week.data} patientId={patientId} onDone={refresh} />}
      {week.data && <WeekGrid w={week.data} />}
      {rewards.data && rewards.data.ledger.length > 0 && <Ledger r={rewards.data} />}
      {(rewards.error || week.error) && <p className="error">{rewards.error || week.error}</p>}
      <CompanionDemoButton patientId={patientId} onDone={() => setCompanionKey((k) => k + 1)} />
    </section>
  )
}

function Summary({ r }: { r: Rewards }) {
  const { level, xp, next_level_xp } = r.level_hint
  const pct = next_level_xp ? Math.min(100, Math.round((xp / next_level_xp) * 100)) : 100
  return (
    <div className="card stack">
      <div className="row">
        <AvatarSlot name="Your companion" level={level} state={r.checked_in_today ? 'happy' : 'resting'} />
        <div className="grow stack">
          <p>
            <strong>Level {level}</strong>
            <span className="muted small"> · {next_level_xp ? `${next_level_xp - xp} points to the next level` : 'top level'}</span>
          </p>
          <div className="progress" aria-label={`Level progress ${pct}%`}>
            <span style={{ width: `${pct}%` }} />
          </div>
        </div>
      </div>
      <div className="stats">
        <div className="stat">
          <b>{r.balance}</b>
          <span className="muted small">points</span>
        </div>
        <div className="stat">
          <b>{r.streak}</b>
          <span className="muted small">day streak</span>
        </div>
        <div className="stat">
          <b>{r.longest_streak}</b>
          <span className="muted small">longest</span>
        </div>
      </div>
      <p className="fine">
        {r.checked_in_today
          ? 'Checked in today. Every check-in counts the same, whatever kind of day it was.'
          : 'Your companion is resting until your check-in. Rest days are fine: one a week keeps your streak.'}
      </p>
    </div>
  )
}

function TodayAction({ w, patientId, onDone }: { w: HabitWeek; patientId: number; onDone: () => void }) {
  const [earned, setEarned] = useState<Earned[] | null>(null)
  const [rating, setRating] = useState<number | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const a = w.today_action

  async function mark(done: boolean) {
    setBusy(true)
    setError(null)
    try {
      const r = await engagementApi.markAction(patientId, done, done ? rating : null)
      setEarned(r.earned)
      onDone()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card stack" id="today-action">
      <h2>Today’s small thing</h2>
      {!a ? (
        <p className="muted">Check in first, and today’s small thing appears here.</p>
      ) : (
        <>
          <p className="intervention">{a.label}</p>
          {a.answered && !earned ? (
            <p className="muted">{a.done ? 'Done today ✓' : 'Noted: not today. That’s an honest answer, and it helps.'}</p>
          ) : earned ? (
            <p>
              {earned.length ? earned.map((e) => `+${e.amount} ${e.label}`).join(' · ') : 'Saved.'}
            </p>
          ) : (
            <>
              <div className="chips" role="group" aria-label="How did it feel? (optional)">
                {[1, 2, 3, 4, 5].map((n) => (
                  <button key={n} className="chip" aria-pressed={rating === n} onClick={() => setRating(rating === n ? null : n)}>
                    {n}
                  </button>
                ))}
                <span className="muted small">how it felt (optional)</span>
              </div>
              <div className="row">
                <button className="btn primary" disabled={busy} onClick={() => mark(true)}>
                  Done
                </button>
                <button className="btn" disabled={busy} onClick={() => mark(false)}>
                  Not today
                </button>
              </div>
              <p className="fine">Both answers earn points. Honest answers help pick better suggestions.</p>
            </>
          )}
        </>
      )}
      {error && <p className="error small">{error}</p>}
    </div>
  )
}

function WeekGrid({ w }: { w: HabitWeek }) {
  const last = w.days.length - 1
  return (
    <div className="card stack">
      <h2>This week</h2>
      <table className="week">
        <thead>
          <tr>
            <th />
            {w.days.map((d, i) => (
              <th key={d} className={i === last ? 'today' : undefined} title={d}>
                {i === last ? 'Today' : WEEKDAY.format(new Date(`${d}T12:00:00`))}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {w.habits.map((h) => (
            <tr key={h.key}>
              <td>{h.label}</td>
              {h.done.map((ok, i) => (
                <td key={w.days[i]}>
                  <span className={`dot ${ok ? 'done' : 'rest'}`} aria-label={ok ? 'done' : 'rest'} />
                </td>
              ))}
            </tr>
          ))}
          <tr>
            <td className="muted">Points</td>
            {w.earned.map((n, i) => (
              <td key={w.days[i]} className="muted">
                {n || ''}
              </td>
            ))}
          </tr>
        </tbody>
      </table>
      <p className="fine">Empty circles are rest days. Nothing you earned is ever taken away.</p>
    </div>
  )
}

function Ledger({ r }: { r: Rewards }) {
  return (
    <div className="card stack">
      <h2>Recent points</h2>
      <ul className="ledger">
        {r.ledger.slice(0, 8).map((row, i) => (
          <li key={`${row.ts}-${row.reason}-${i}`}>
            <span>
              {row.label} <span className="muted small">{row.date}</span>
            </span>
            <span className={row.amount > 0 ? 'plus' : 'muted'}>
              {row.amount > 0 ? `+${row.amount}` : row.amount}
            </span>
          </li>
        ))}
      </ul>
    </div>
  )
}
