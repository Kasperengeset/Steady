// Placeholder for Mahdan's avatar (#20, #23): a coloured circle with initials, a level badge and a state ring.
// Replace the inside of this component with the character assets from public/avatar/; keep the props.

export type AvatarState = 'calm' | 'happy' | 'celebrating' | 'resting' | 'glad'

const PALETTE = ['#2f6f62', '#7a5c99', '#b7791f', '#3d6fa3', '#a3543d', '#4f7d3a']

function initials(name: string) {
  const parts = name.trim().split(/\s+/).filter(Boolean)
  return (parts.length > 1 ? parts[0][0] + parts[1][0] : name.slice(0, 2)).toUpperCase() || '?'
}

function colourFor(name: string) {
  let h = 0
  for (const ch of name) h = (h * 31 + ch.charCodeAt(0)) >>> 0
  return PALETTE[h % PALETTE.length]
}

export default function AvatarSlot({
  name,
  level,
  state = 'calm',
  colour,
  size = 56,
}: {
  name: string
  level?: number | null
  state?: AvatarState
  colour?: string | null
  size?: number
}) {
  return (
    <span
      className={`avatar-slot state-${state}`}
      style={{ width: size, height: size, background: colour || colourFor(name), fontSize: size * 0.36 }}
      role="img"
      aria-label={`${name}'s avatar${level ? `, level ${level}` : ''}`}
    >
      {initials(name)}
      {level ? <span className="avatar-level">{level}</span> : null}
      {state === 'resting' && <span className="avatar-badge" aria-hidden>z</span>}
      {(state === 'celebrating' || state === 'glad') && <span className="avatar-badge" aria-hidden>✦</span>}
    </span>
  )
}
