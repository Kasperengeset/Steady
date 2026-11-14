import { useState } from 'react'
import { engagementApi, type Profile } from '../../lib/api_engagement'
import CircleFeed from './CircleFeed'
import ProfileForm from './ProfileForm'
import { DEMO_PATIENT_ID, useLoad } from './useLoad'
import './engagement.css'

/** The patient's circle (#26): own consent first, then create or join, invite by code, the feed. */
export default function CirclePage({ patientId = DEMO_PATIENT_ID }: { patientId?: number }) {
  const profile = useLoad(() => engagementApi.profile(patientId))
  const circle = useLoad(() => engagementApi.myCircle(patientId), { refreshOnFocus: true })
  const [editing, setEditing] = useState(false)

  if (profile.error || circle.error) return <p className="error">{profile.error || circle.error}</p>
  if (!profile.data || !circle.loaded) return <p className="muted">Loading…</p>
  const p = profile.data

  const saved = (next: Profile) => {
    profile.setData(next)
    setEditing(false)
    circle.reload()
  }

  return (
    <section className="view eng">
      <h1>Circle</h1>
      {!p.social_consent ? (
        <>
          <p className="muted">
            Share your avatar and streak with a few close people, and get small reactions back. Only what you pick is
            shared, never how you are doing.
          </p>
          <ProfileForm patientId={patientId} profile={p} onSaved={saved} submitLabel="Start my circle" />
        </>
      ) : !circle.data ? (
        <StartCircle patientId={patientId} onDone={circle.reload} />
      ) : (
        <>
          <div className="card stack">
            <h2>Invite</h2>
            <p className="code">{circle.data.invite_code}</p>
            <p className="fine">
              Friends join at <strong>/circle/friend</strong> with this code. They don’t need the health features.
            </p>
          </div>
          {circle.data.nudge && (
            <div className="card stack">
              <p>{circle.data.nudge.message}</p>
              {circle.data.nudge.pending && circle.data.nudge.can_cancel && (
                <div className="row">
                  <button className="btn" onClick={() => engagementApi.cancelNudge(patientId).then(circle.reload)}>
                    Cancel the card
                  </button>
                </div>
              )}
            </div>
          )}
          <CircleFeed token={circle.data.token} />
          {editing ? (
            <ProfileForm patientId={patientId} profile={p} onSaved={saved} submitLabel="Save" />
          ) : (
            <button className="link" onClick={() => setEditing(true)}>
              How I appear and what I share
            </button>
          )}
        </>
      )}
    </section>
  )
}

function StartCircle({ patientId, onDone }: { patientId: number; onDone: () => void }) {
  const [code, setCode] = useState('')
  const [error, setError] = useState<string | null>(null)
  const run = (fn: () => Promise<unknown>) => fn().then(onDone, (e: unknown) => setError(String(e)))
  return (
    <div className="card stack">
      <h2>Your circle</h2>
      <div className="row">
        <button className="btn primary" onClick={() => run(() => engagementApi.createCircle(patientId))}>
          Create my circle
        </button>
      </div>
      <p className="muted small">or join someone else’s with their code</p>
      <div className="row">
        <input
          className="grow"
          type="text"
          value={code}
          onChange={(e) => setCode(e.target.value)}
          placeholder="Invite code"
          aria-label="Invite code"
        />
        <button className="btn" disabled={!code.trim()} onClick={() => run(() => engagementApi.join(code, null, patientId))}>
          Join
        </button>
      </div>
      {error && <p className="error small">{error}</p>}
    </div>
  )
}
