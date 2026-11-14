// The support directory is separate from the opt-in Appie circle. Contacts have no phone data here.
import { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import { views, type Contact } from '../lib/views'
import { Card, Icon } from '../ui/kit'
import { LoadError } from '../ui/Appie'
import { PID } from './Shell'

export default function Friends() {
  const [contacts, setContacts] = useState<Contact[] | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const mounted = useRef(true)
  const load = useCallback(async () => {
    setLoading(true)
    setError('')
    try {
      const people = await views.contacts(PID)
      if (mounted.current) setContacts(people)
    } catch {
      if (mounted.current) setError('Your support contacts could not load. Please check your connection and try again.')
    } finally { if (mounted.current) setLoading(false) }
  }, [])
  useEffect(() => {
    mounted.current = true
    void load()
    return () => { mounted.current = false }
  }, [load])

  const friends = contacts?.filter((person) => person.kind === 'friend') ?? []
  const clinicians = contacts?.filter((person) => person.kind === 'clinician') ?? []
  const people = (list: Contact[], kind: string) => <ul className="mt-4 flex flex-col gap-4">
    {list.map((person) => <li key={person.id} className="flex items-center gap-3">
      <span aria-hidden className="grid h-11 w-11 shrink-0 place-items-center rounded-full bg-sand font-semibold text-secondary">{person.name.trim()[0] || <Icon name="person" />}</span>
      <div className="min-w-0"><p className="break-words font-semibold">{person.name}</p><p className="mt-0.5 text-xs text-ink-soft">{kind}</p></div>
    </li>)}
  </ul>

  return <div className="flex flex-col gap-5 pt-3">
    <div><p className="eyebrow">YOU HAVE PEOPLE IN YOUR CORNER</p><h1 className="mt-1 text-3xl font-semibold">Your support contacts</h1><p className="mt-2 text-sm leading-relaxed text-ink-soft">A familiar person can make a little space for you. Reach out using your usual phone or messaging app.</p></div>
    {loading ? <p role="status" className="rounded-3xl bg-surface-low p-5 text-sm text-ink-soft">Loading the people in your corner…</p>
      : error ? <LoadError error={error} retry={() => { void load() }} /> : <>
        <Card><h2 className="flex items-center gap-2 font-semibold"><Icon name="diversity_1" className="text-primary" />Close friends</h2>{friends.length ? people(friends, 'Someone in your support network') : <p className="mt-3 text-sm leading-relaxed text-ink-soft">No close friends are listed here yet. You can still reach out to someone you trust.</p>}</Card>
        <Card><h2 className="flex items-center gap-2 font-semibold"><Icon name="health_and_safety" className="text-primary" />Your care team</h2>{clinicians.length ? people(clinicians, 'Care team contact') : <p className="mt-3 text-sm leading-relaxed text-ink-soft">No care team contact is listed here. Use the contact details your clinic gave you.</p>}</Card>
      </>}
    <Card className="bg-sage-soft/60"><h2 className="flex items-center gap-2 font-semibold text-primary"><Icon name="lock" />How reminders work</h2><p className="mt-3 text-sm leading-relaxed text-ink-soft">The friend view can show a gentle reminder to check in with you. It includes no voice recording, diary, measurements or scores.</p><p className="mt-3 text-sm leading-relaxed text-ink-soft">When a check-in is flagged for care team review, that flag is recorded in the clinic view. Steady does not send a text, email or emergency call, and cannot guarantee a response time.</p></Card>
    <div className="rounded-3xl bg-sand px-5 py-4 text-sm leading-relaxed"><p className="font-semibold">Need someone now?</p><p className="mt-2 text-ink-soft">Please contact someone directly. If you are in immediate danger, call your local emergency number or go to the nearest emergency department.</p></div>
    <Link to="/circle" className="flex min-h-14 items-center justify-between gap-3 rounded-3xl bg-surface-low px-5 py-4 text-sm font-semibold text-primary"><span className="flex items-center gap-2"><Icon name="pets" />Visit your Appie circle</span><Icon name="arrow_forward" /></Link>
    <Link to="/" className="py-2 text-center text-sm text-primary underline underline-offset-4">Back to today</Link>
  </div>
}
