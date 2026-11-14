import { useEffect, useState } from 'react'
import { api, type FriendHint } from '../lib/api'

export default function FriendView() {
  const [hint, setHint] = useState<FriendHint | null>(null)

  useEffect(() => {
    api.friendHint(1).then(setHint).catch(() => setHint(null))
  }, [])

  return (
    <section className="view">
      <h1>Friends</h1>
      {hint?.show_hint ? (
        <div className="card nudge">
          <p>{hint.message}</p>
          <button>Send a message</button>
        </div>
      ) : (
        <p className="muted">No nudge today. You never see health data – only a hint when reaching out could help.</p>
      )}
    </section>
  )
}
