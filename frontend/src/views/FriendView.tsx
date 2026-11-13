import { useEffect, useState } from 'react'
import { api, type FriendHint } from '../lib/api'

export default function FriendView() {
  const [hint, setHint] = useState<FriendHint | null>(null)

  useEffect(() => {
    api.friendHint(1).then(setHint).catch(() => setHint(null))
  }, [])

  return (
    <section className="view">
      <h1>Venner</h1>
      {hint?.show_hint ? (
        <div className="card nudge">
          <p>{hint.message}</p>
          <button>Send en melding</button>
        </div>
      ) : (
        <p className="muted">Ingen hint i dag. Du ser aldri helsedata – bare et vink når kontakt kan hjelpe.</p>
      )}
    </section>
  )
}
