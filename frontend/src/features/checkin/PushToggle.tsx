import { useEffect, useState } from 'react'
import { checkinApi, type PushStatus } from '../../lib/api_checkin'

/** VAPID public key (base64url) → the bytes pushManager.subscribe() wants. */
function keyBytes(b64url: string): Uint8Array {
  const b64 = (b64url + '='.repeat((4 - (b64url.length % 4)) % 4)).replace(/-/g, '+').replace(/_/g, '/')
  return Uint8Array.from(atob(b64), (ch) => ch.charCodeAt(0))
}

const supported = () => 'serviceWorker' in navigator && 'PushManager' in window && 'Notification' in window
const isIos = () => /iphone|ipad|ipod/i.test(navigator.userAgent)
const standalone = () => window.matchMedia('(display-mode: standalone)').matches

async function registration(): Promise<ServiceWorkerRegistration> {
  // The PWA plugin registers the service worker in production builds. In dev there is none, so say so.
  const reg = await navigator.serviceWorker.getRegistration()
  if (!reg) throw new Error('No service worker yet – use the installed app or a production build (npm run build && npm run preview).')
  return navigator.serviceWorker.ready
}

/**
 * "Turn on notifications" for the daily prompt (#22, P2). Without it the app shows the prompt when opened.
 * iPhone: only after "Add to Home Screen" (iOS 16.4+).
 */
export default function PushToggle({ patientId }: { patientId: number }) {
  const [server, setServer] = useState<PushStatus | null>(null)
  const [on, setOn] = useState(false)
  const [busy, setBusy] = useState(false)
  const [msg, setMsg] = useState<string | null>(null)

  useEffect(() => {
    if (!supported()) return
    checkinApi.push(patientId).then(setServer).catch(() => setServer(null))
    navigator.serviceWorker.getRegistration()
      .then((reg) => reg?.pushManager.getSubscription())
      .then((sub) => setOn(!!sub))
      .catch(() => setOn(false))
  }, [patientId])

  if (!supported()) {
    return <p className="muted">{isIos() && !standalone()
      ? 'To get reminders on iPhone, add this app to your Home Screen first (Share → Add to Home Screen).'
      : 'This browser cannot show notifications. The check-in still appears when you open the app.'}</p>
  }
  if (!server?.enabled) return null // the server has no VAPID keys: in-app prompt only

  async function run(fn: () => Promise<string>) {
    setBusy(true)
    setMsg(null)
    try {
      setMsg(await fn())
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e))
    } finally {
      setBusy(false)
    }
  }

  const turnOn = () => run(async () => {
    if ((await Notification.requestPermission()) !== 'granted') return 'Notifications are blocked in the browser settings.'
    const reg = await registration()
    const sub = await reg.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: keyBytes(server.public_key ?? '') })
    await checkinApi.pushSubscribe(patientId, sub.toJSON())
    setOn(true)
    return 'Notifications are on.'
  })

  const turnOff = () => run(async () => {
    const sub = await (await navigator.serviceWorker.getRegistration())?.pushManager.getSubscription()
    if (sub) {
      await checkinApi.pushUnsubscribe(patientId, sub.endpoint)
      await sub.unsubscribe()
    }
    setOn(false)
    return 'Notifications are off.'
  })

  const test = () => run(async () => {
    const r = await checkinApi.pushTest(patientId)
    return r.sent ? 'Test sent.' : 'No device accepted the test.'
  })

  return (
    <div className="row">
      {on ? (
        <>
          <span>Notifications on</span>
          <button className="pill small" onClick={test} disabled={busy}>Send a test</button>
          <button className="link" onClick={turnOff} disabled={busy}>Turn off</button>
        </>
      ) : (
        <button className="pill small" onClick={turnOn} disabled={busy}>Turn on notifications</button>
      )}
      {msg && <span className="muted">{msg}</span>}
    </div>
  )
}
