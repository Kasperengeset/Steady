import { useState } from 'react'
import { engagementApi, type MemberCard } from '../../lib/api_engagement'
import AvatarSlot from './AvatarSlot'
import { useLoad } from './useLoad'

/** The circle feed (#26) for one member token: opt-in cards, preset reactions, discreet reach-out cards. */
export default function CircleFeed({ token }: { token: string }) {
  const feed = useLoad(() => engagementApi.feed(token), { refreshOnFocus: true })
  const presets = useLoad(() => engagementApi.presets())
  const [sent, setSent] = useState<Record<number, string>>({})

  if (feed.error) return <p className="error">{feed.error}</p>
  if (!feed.data) return <p className="muted">Loading the circle…</p>
  const f = feed.data
  const others = f.members.filter((m) => !m.is_you)

  async function react(m: MemberCard, kind: string, label: string) {
    await engagementApi.react(token, m.member_id, kind)
    setSent((s) => ({ ...s, [m.member_id]: label }))
    feed.reload()
  }

  return (
    <div className="stack">
      {f.reach_out_cards.map((c) => (
        <div key={c.to_member_id} className="card reach-out">
          <p>{c.message}</p>
        </div>
      ))}

      <div className="card stack">
        <h2>{f.circle.name}</h2>
        {others.length === 0 ? (
          <p className="muted">Nobody here yet. Share the invite code with someone close.</p>
        ) : (
          <ul className="members">
            {others.map((m) => (
              <li key={m.member_id}>
                {m.role === 'patient' ? (
                  <AvatarSlot
                    name={m.avatar?.name || m.name}
                    colour={m.avatar?.colour}
                    level={m.level}
                    state={m.checked_in_today ? 'happy' : 'calm'}
                    size={44}
                  />
                ) : (
                  <AvatarSlot name={m.name} size={44} colour="var(--muted)" />
                )}
                <div className="grow stack">
                  <p>
                    <strong>{m.name}</strong> {m.emoji && <span aria-label="status">{m.emoji}</span>}
                  </p>
                  <p>
                    {m.streak !== null && m.streak > 0 && <span className="tag">{m.streak}-day streak</span>}
                    {m.checked_in_today && <span className="tag">checked in today ✓</span>}
                    {m.role === 'friend' && <span className="tag">friend</span>}
                  </p>
                  {sent[m.member_id] ? (
                    <p className="muted small">Sent: {sent[m.member_id]}</p>
                  ) : (
                    <div className="chips" aria-label={`React to ${m.name}`}>
                      {presets.data?.reactions.map((p) => (
                        <button key={p.kind} className="chip" onClick={() => react(m, p.kind, p.label)}>
                          {p.label}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              </li>
            ))}
          </ul>
        )}
      </div>

      {f.reactions.length > 0 && (
        <div className="card stack">
          <h2>Recent</h2>
          <ul className="ledger">
            {f.reactions.slice(0, 10).map((r) => {
              const to = f.members.find((m) => m.member_id === r.to_member_id)
              const toName = r.to_member_id === f.you ? 'you' : to?.name || 'a member'
              return (
                <li key={r.id}>
                  <span>
                    <strong>{r.from_member_id === f.you ? 'You' : r.from_name}</strong> → {toName}: {r.label}
                  </span>
                </li>
              )
            })}
          </ul>
        </div>
      )}
    </div>
  )
}
