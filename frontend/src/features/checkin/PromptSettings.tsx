import { useState } from 'react'
import type { PromptSettingsIn, PromptStatus } from '../../lib/api_checkin'
import './checkin.css'

interface Props {
  status: PromptStatus | null
  onChange: (body: PromptSettingsIn) => Promise<void>
  onSendNow: () => Promise<void>
}

/** Reminder settings (#22): quiet hours, pause today, turn off, plus the demo trigger. */
export default function PromptSettings({ status, onChange, onSendNow }: Props) {
  const st = status?.settings
  const [quiet, setQuiet] = useState<{ start: string; end: string } | null>(null)
  const [msg, setMsg] = useState<string | null>(null)
  if (!st) return null
  const q = quiet ?? { start: st.quiet_start ?? '', end: st.quiet_end ?? '' }

  async function run(fn: () => Promise<void>, done: string) {
    setMsg(null)
    try {
      await fn()
      setMsg(done)
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e))
    }
  }

  return (
    <div className="prompt-settings">
      <details className="card">
        <summary>Reminders</summary>
        <p className="muted">Once a day, at a random time between {st.window_start} and {st.window_end}.</p>
        <label className="row">
          <input type="checkbox" checked={st.enabled}
            onChange={(e) => run(() => onChange({ enabled: e.target.checked }), 'Saved')} />
          Daily reminder
        </label>
        <label className="row">
          <input type="checkbox" checked={st.paused_today} disabled={!st.enabled}
            onChange={(e) => run(() => onChange({ pause_today: e.target.checked }), 'Saved')} />
          Pause for today
        </label>
        <div className="row">
          <span>Quiet hours</span>
          <input type="time" value={q.start} onChange={(e) => setQuiet({ ...q, start: e.target.value })} aria-label="Quiet from" />
          <span>–</span>
          <input type="time" value={q.end} onChange={(e) => setQuiet({ ...q, end: e.target.value })} aria-label="Quiet until" />
          <button className="pill small" onClick={() => run(async () => {
            await onChange({ quiet_start: q.start || null, quiet_end: q.end || null })
            setQuiet(null)
          }, 'Saved')}>Save</button>
        </div>
        {msg && <p className="muted">{msg}</p>}
      </details>
      <button className="link demo" onClick={() => run(onSendNow, "Today's prompt sent")}>
        Demo: send today's prompt now
      </button>
    </div>
  )
}
