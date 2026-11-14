// Shared character artwork, preserving the engagement feature's public props.
import { Appie } from '../../ui/Appie'

export type AvatarState = 'calm' | 'happy' | 'celebrating' | 'resting' | 'glad'

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
      style={{ width: size, height: size, background: colour || '#dfeee3', fontSize: size * 0.36 }}
      role="img"
      aria-label={`${name}'s avatar${level ? `, level ${level}` : ''}`}
    >
      <span aria-hidden style={{ display: 'block', width: size, height: size }}><Appie name={name} mood={{ calm: 'calm', happy: 'happy', celebrating: 'celebrate', resting: 'sleep', glad: 'wave' }[state]} tier={level && level >= 5 ? 3 : level && level >= 3 ? 2 : 1} className="h-full w-full" /></span>
      {level ? <span className="avatar-level">{level}</span> : null}
      {state === 'resting' && <span className="avatar-badge" aria-hidden>z</span>}
      {(state === 'celebrating' || state === 'glad') && <span className="avatar-badge" aria-hidden>✦</span>}
    </span>
  )
}
