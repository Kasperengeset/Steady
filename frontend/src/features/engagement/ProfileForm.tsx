import { useState } from 'react'
import { engagementApi, type Profile, type ProfileIn } from '../../lib/api_engagement'
import SupportCard from './SupportCard'

const EMOJI = ['🌿', '☀️', '🌧️', '🙂', '😴', '💪', '🎧', '🌙']

/**
 * Profile or anonymous, what to share, and the circle's own consent (#26).
 * Health data is never shared; this only controls the avatar, level, streak and "checked in today".
 */
export default function ProfileForm({
  patientId,
  profile,
  onSaved,
  submitLabel,
}: {
  patientId: number
  profile: Profile
  onSaved: (p: Profile) => void
  submitLabel: string
}) {
  const [p, setP] = useState<Profile>(profile)
  const [consent, setConsent] = useState(profile.social_consent)
  const [busy, setBusy] = useState(false)
  const [crisis, setCrisis] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const set = (patch: Partial<Profile>) => setP((x) => ({ ...x, ...patch }))

  async function save() {
    setBusy(true)
    setError(null)
    try {
      const body: ProfileIn = {
        mode: p.mode,
        display_name: p.display_name || null,
        pseudonym: p.pseudonym || null,
        emoji: p.emoji,
        share_avatar: p.share_avatar,
        share_level: p.share_level,
        share_streak: p.share_streak,
        share_checked_in: p.share_checked_in,
        social_consent: consent,
      }
      const r = await engagementApi.saveProfile(patientId, body)
      setCrisis(r.crisis)
      onSaved(r)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="card stack">
      <h2>How you appear</h2>
      <div className="chips" role="radiogroup" aria-label="Identity">
        <button className="chip" aria-pressed={p.mode === 'anonymous'} onClick={() => set({ mode: 'anonymous' })}>
          Anonymous
        </button>
        <button className="chip" aria-pressed={p.mode === 'profile'} onClick={() => set({ mode: 'profile' })}>
          With my name
        </button>
      </div>
      {p.mode === 'anonymous' ? (
        <input
          type="text"
          maxLength={40}
          value={p.pseudonym ?? ''}
          onChange={(e) => set({ pseudonym: e.target.value })}
          placeholder="A pseudonym, e.g. Quiet Otter"
          aria-label="Pseudonym"
        />
      ) : (
        <input
          type="text"
          maxLength={40}
          value={p.display_name ?? ''}
          onChange={(e) => set({ display_name: e.target.value })}
          placeholder="Your name as friends know it"
          aria-label="Display name"
        />
      )}
      <p className="fine">
        {p.mode === 'anonymous'
          ? 'Nobody in the circle sees your real name. Your care team always knows who you are.'
          : 'Friends see this name.'}
      </p>

      <h2>Status (optional)</h2>
      <div className="chips" role="radiogroup" aria-label="Emoji status">
        {EMOJI.map((e) => (
          <button key={e} className="chip" aria-pressed={p.emoji === e} onClick={() => set({ emoji: p.emoji === e ? null : e })}>
            {e}
          </button>
        ))}
      </div>

      <h2>What friends can see</h2>
      {(
        [
          ['share_avatar', 'My avatar'],
          ['share_level', 'My level'],
          ['share_streak', 'My streak'],
          ['share_checked_in', '“Checked in today”'],
        ] as const
      ).map(([k, label]) => (
        <label key={k} className="toggle">
          <input type="checkbox" checked={p[k]} onChange={(e) => set({ [k]: e.target.checked })} />
          {label}
        </label>
      ))}
      <p className="fine">
        Never shared: how you are doing, scores, your diary, your voice, transcripts, health or Fitbit data.
      </p>

      <label className="toggle">
        <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} />
        I agree to share the items I picked above with the people in my circle. I can change or withdraw this at any time.
      </label>
      <div className="row">
        <button className="btn primary" disabled={busy} onClick={save}>
          {busy ? '…' : submitLabel}
        </button>
      </div>
      {crisis && <SupportCard />}
      {error && <p className="error small">{error}</p>}
    </div>
  )
}
