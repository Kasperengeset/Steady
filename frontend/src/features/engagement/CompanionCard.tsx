import { useEffect, useState } from 'react'
import { api, type Slot } from '../../lib/api'
import { engagementApi, type Companion } from '../../lib/api_engagement'
import AvatarSlot from './AvatarSlot'
import SupportCard from './SupportCard'
import { DEMO_PATIENT_ID, useLoad } from './useLoad'

/**
 * The avatar's kind check-in (#25): at most one a day, from the day's prompt time on.
 * Self-contained, so it can sit on Today in the prototype as well: <CompanionCard patientId={id} />.
 */
export default function CompanionCard({
  patientId = DEMO_PATIENT_ID,
  avatarName = 'Your companion',
  onCheckedIn,
}: {
  patientId?: number
  avatarName?: string
  onCheckedIn?: () => void
}) {
  const { data } = useLoad(() => engagementApi.companion(patientId))
  const [writing, setWriting] = useState(false)
  const [text, setText] = useState('')
  const [busy, setBusy] = useState(false)
  const [done, setDone] = useState<string | null>(null)
  const [crisis, setCrisis] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (data?.id) engagementApi.companionSeen(patientId, data.id).catch(() => {})
  }, [data?.id, patientId])

  if (crisis) return <SupportCard />
  if (!data?.message || !data.id) return null
  const msg: Companion & { id: number } = { ...data, id: data.id }

  async function sendWritten() {
    if (!text.trim()) return
    setBusy(true)
    setError(null)
    try {
      const slot: Slot = new Date().getHours() < 14 ? 'morning' : 'evening'
      const r = await api.checkInText(patientId, slot, text.trim()) // the check-in path runs its own crisis check
      await engagementApi.companionReply(patientId, msg.id, null)
      if (r.crisis) setCrisis(true)
      setDone('Thank you. That counts as today’s check-in.')
      setWriting(false)
      onCheckedIn?.()
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  async function sayHi() {
    const r = await engagementApi.companionReply(patientId, msg.id, null)
    if (r.crisis) setCrisis(true)
    setDone('👋')
  }

  return (
    <div className="card companion" aria-live="polite">
      <AvatarSlot name={avatarName} state={msg.glad_to_see_you ? 'glad' : 'calm'} size={48} />
      <div className="grow stack">
        <p className="bubble">{msg.message}</p>
        {done ? (
          <p className="muted small">{done}</p>
        ) : writing ? (
          <div className="stack">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="A few words about today…"
              aria-label="Write a few words about today"
            />
            <div className="row">
              <button className="btn primary" onClick={sendWritten} disabled={busy || !text.trim()}>
                {busy ? '…' : 'Send'}
              </button>
              <button className="link" onClick={() => setWriting(false)}>
                Not now
              </button>
            </div>
          </div>
        ) : (
          <div className="row">
            {msg.offer_written && (
              <button className="btn" onClick={() => setWriting(true)}>
                Write instead
              </button>
            )}
            {msg.offer_action && (
              <a className="btn" href="#today-action">
                Today’s small thing
              </a>
            )}
            <button className="link" onClick={sayHi}>
              Say hi back
            </button>
          </div>
        )}
        {error && <p className="error small">{error}</p>}
      </div>
    </div>
  )
}

/** Demo only: make today's check-in appear now instead of at the prompt time. */
export function CompanionDemoButton({ patientId = DEMO_PATIENT_ID, onDone }: { patientId?: number; onDone: () => void }) {
  return (
    <button className="link small" onClick={() => engagementApi.companionDemo(patientId).then(onDone, onDone)}>
      Demo: show the avatar’s check-in now
    </button>
  )
}
