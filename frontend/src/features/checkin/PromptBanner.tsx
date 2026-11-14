import type { PromptStatus } from '../../lib/api_checkin'
import { formatLeft } from './format'
import './checkin.css'

interface Props {
  status: PromptStatus | null
  secondsLeft: number | null
  onStart: () => void
}

/**
 * Today banner for the daily prompt (#22). Never punishes: after the on-time window a check-in still
 * counts fully, only the on-time bonus (#24) is gone, and the text says so kindly.
 */
export default function PromptBanner({ status, secondsLeft, onStart }: Props) {
  if (!status) return null
  if (status.open) {
    return (
      <div className="card prompt-banner" role="status">
        <div>
          <strong>Your check-in is ready</strong>
          <p className="muted">
            {secondsLeft && secondsLeft > 0
              ? `${formatLeft(secondsLeft)} left in the on-time window`
              : 'It still counts fully whenever you do it.'}
          </p>
        </div>
        <button className="pill" onClick={onStart}>Start</button>
      </div>
    )
  }
  if (status.state === 'answered') {
    return <p className="prompt-note muted">✓ Checked in today{status.on_time ? ', right on time' : ''}</p>
  }
  if (status.state === 'scheduled') return <p className="prompt-note muted">Today's check-in will pop up at a random time.</p>
  if (status.state === 'paused') return <p className="prompt-note muted">Reminders are paused for today.</p>
  return null
}
