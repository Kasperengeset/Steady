// What a close friend sees: a gentle nudge, never scores or health data (backend friends router).
import { useEffect, useState } from 'react'
import { useParams } from 'react-router-dom'
import { api, type FriendHint } from '../lib/api'
import { Icon } from '../ui/kit'

export default function FriendNudge() {
  const pid = Number(useParams().pid ?? 1)
  const [h, setH] = useState<FriendHint | null>(null)
  const [name, setName] = useState('')
  useEffect(() => {
    api.friendHint(pid).then(setH)
    api.patient(pid).then((p) => setName(String(p.patient.name ?? '')))
  }, [pid])

  return (
    <div className="mx-auto flex min-h-full max-w-[440px] flex-col gap-6 bg-surface px-5 pt-16 text-center">
      <span className="mx-auto grid h-16 w-16 place-items-center rounded-full bg-sand text-secondary"><Icon name="favorite" className="text-3xl" /></span>
      {h?.show_hint ? (
        <>
          <h1 className="text-[28px] font-semibold leading-tight">{name} might like a check-in today.</h1>
          <p className="text-lg text-ink-soft">A short message or a call is enough. No health details are shared.</p>
          <a href="sms:" className="flex h-14 items-center justify-center gap-2 rounded-full bg-primary text-lg font-semibold text-on-primary"><Icon name="chat" /> Send a message</a>
          <a href="tel:" className="flex h-14 items-center justify-center gap-2 rounded-full bg-surface-high text-lg"><Icon name="call" /> Call</a>
          <div className="rounded-3xl bg-surface-low p-5 text-left text-sm text-ink-soft">
            <p className="font-semibold text-ink">How to check in</p>
            <p className="mt-2">Ask how their week has been, and listen. Suggest something small together, like a walk.</p>
          </div>
        </>
      ) : (
        <>
          <h1 className="text-[28px] font-semibold">Nothing needed right now.</h1>
          <p className="text-lg text-ink-soft">You'll get a gentle note here if {name || 'your friend'} could use a check-in.</p>
        </>
      )}
    </div>
  )
}
