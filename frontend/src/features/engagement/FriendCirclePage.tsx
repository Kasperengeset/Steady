import { useState } from 'react'
import { engagementApi } from '../../lib/api_engagement'
import CircleFeed from './CircleFeed'
import { store } from './useLoad'
import './engagement.css'

const KEY = 'circle-friend-token'

/** A light friend account (#26): join with a code and a name, then see the feed. No health features. */
export default function FriendCirclePage() {
  const [token, setToken] = useState<string | null>(() => store.get(KEY))
  const [code, setCode] = useState(() => new URLSearchParams(window.location.search).get('code') ?? '')
  const [name, setName] = useState('')
  const [error, setError] = useState<string | null>(null)

  async function join() {
    setError(null)
    try {
      const r = await engagementApi.join(code, name)
      store.set(KEY, r.token)
      setToken(r.token)
    } catch (e) {
      setError(e instanceof Error && e.message.startsWith('404') ? 'That code was not found.' : String(e))
    }
  }

  async function leave() {
    if (token) await engagementApi.leave(token).catch(() => {})
    store.set(KEY, null)
    setToken(null)
  }

  return (
    <section className="view eng">
      <h1>Circle</h1>
      {token ? (
        <>
          <CircleFeed token={token} />
          <button className="link" onClick={leave}>
            Leave this circle
          </button>
        </>
      ) : (
        <div className="card stack">
          <h2>Join a friend’s circle</h2>
          <input type="text" value={code} onChange={(e) => setCode(e.target.value)} placeholder="Invite code" aria-label="Invite code" />
          <input type="text" maxLength={40} value={name} onChange={(e) => setName(e.target.value)} placeholder="Your name" aria-label="Your name" />
          <div className="row">
            <button className="btn primary" disabled={!code.trim() || !name.trim()} onClick={join}>
              Join
            </button>
          </div>
          <p className="fine">You’ll see their avatar and the things they chose to share, never health data.</p>
          {error && <p className="error small">{error}</p>}
        </div>
      )}
    </section>
  )
}
