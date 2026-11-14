import { useCallback, useEffect, useState } from 'react'
import { community, type PromptStatus, type PushStatus } from '../lib/community'
import { Card, Icon } from '../ui/kit'
import { PID } from './Shell'

function applicationKey(value: string): Uint8Array<ArrayBuffer> {
  const base64 = value.replace(/-/g, '+').replace(/_/g, '/')
  const raw = atob(base64 + '='.repeat((4 - base64.length % 4) % 4))
  const result = new Uint8Array(raw.length)
  for (let i = 0; i < raw.length; i++) result[i] = raw.charCodeAt(i)
  return result
}

export default function PromptSettings() {
  const [prompt, setPrompt] = useState<PromptStatus | null>(null)
  const [push, setPush] = useState<PushStatus | null>(null)
  const [tz, setTz] = useState('')
  const [enabled, setEnabled] = useState(true)
  const [quiet, setQuiet] = useState(false)
  const [start, setStart] = useState('22:00')
  const [end, setEnd] = useState('08:00')
  const [endpoint, setEndpoint] = useState<string | null>(null)
  const [busy, setBusy] = useState('')
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const supported = window.isSecureContext && 'Notification' in window && 'serviceWorker' in navigator && 'PushManager' in window
  const load = useCallback(async () => {
    try {
      const value = await community.prompt(PID)
      setPrompt(value); setTz(value.settings.tz); setEnabled(value.settings.enabled)
      setQuiet(Boolean(value.settings.quiet_start && value.settings.quiet_end))
      setStart(value.settings.quiet_start || '22:00'); setEnd(value.settings.quiet_end || '08:00'); setError('')
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not load your daily moment settings.') }
    try {
      setPush(await community.push(PID))
      if (supported) {
        const registration = await navigator.serviceWorker.getRegistration()
        setEndpoint((await registration?.pushManager.getSubscription())?.endpoint || null)
      }
    } catch { /* Prompt preferences remain available when notification setup is unavailable. */ }
  }, [supported])
  useEffect(() => { void load() }, [load])
  const run = async (key: string, action: () => Promise<void>) => {
    setBusy(key); setError(''); setNotice('')
    try { await action() } catch (e) { setError(e instanceof Error ? e.message : 'Could not save that. Please try again.') }
    finally { setBusy('') }
  }
  const save = (e: React.FormEvent) => {
    e.preventDefault()
    void run('save', async () => {
      const value = await community.promptSettings(PID, { tz: tz.trim(), enabled, quiet_start: quiet ? start : null, quiet_end: quiet ? end : null })
      setPrompt(value); setNotice('Your daily moment settings are saved.')
    })
  }
  const subscribe = () => run('push', async () => {
    if (!supported || !push?.public_key || !push.enabled) throw new Error('Notifications are not available on this device or server yet.')
    const registration = await navigator.serviceWorker.getRegistration()
    if (!registration?.active) throw new Error('The app is still getting ready for notifications. Open the installed app or refresh and try again.')
    const permission = await Notification.requestPermission()
    if (permission !== 'granted') throw new Error(permission === 'denied' ? 'Notifications are blocked in your browser settings. You can turn them on there whenever you like.' : 'No problem. You can turn notifications on later.')
    const subscription = await registration.pushManager.getSubscription() || await registration.pushManager.subscribe({ userVisibleOnly: true, applicationServerKey: applicationKey(push.public_key) })
    setPush(await community.subscribe(PID, subscription.toJSON())); setEndpoint(subscription.endpoint); setNotice('Notifications are on for this device.')
  })
  const unsubscribe = () => run('push', async () => {
    if (!endpoint) return
    setPush(await community.unsubscribe(PID, endpoint))
    const registration = await navigator.serviceWorker.getRegistration()
    await (await registration?.pushManager.getSubscription())?.unsubscribe()
    setEndpoint(null); setNotice('Notifications are off on this device. You can still check in anytime.')
  })
  return <Card>
    <div className="flex items-center gap-2"><Icon name="notifications_none" className="text-primary" /><h2 className="text-lg font-semibold">Your daily moment</h2></div>
    <p className="mt-2 text-sm leading-relaxed text-ink-soft">A surprise prompt between 09:00 and 21:00 in your time zone. Choose when it can reach you.</p>
    {prompt ? <form onSubmit={save} className="mt-4 flex flex-col gap-4">
      <label className="flex items-center gap-3 text-sm"><input type="checkbox" checked={enabled} onChange={(e) => setEnabled(e.target.checked)} className="h-4 w-4 accent-primary" />Daily prompts on</label>
      <div><label htmlFor="prompt-timezone" className="text-xs font-semibold text-ink-soft">Time zone</label><input id="prompt-timezone" value={tz} onChange={(e) => setTz(e.target.value)} required list="prompt-timezones" className="form-input mt-2" placeholder="Europe/Oslo" /><datalist id="prompt-timezones">{Array.from(new Set([Intl.DateTimeFormat().resolvedOptions().timeZone, 'Europe/Oslo', 'Europe/Helsinki', 'Europe/London', 'America/New_York', 'UTC'])).map((zone) => <option key={zone} value={zone} />)}</datalist><button type="button" onClick={() => setTz(Intl.DateTimeFormat().resolvedOptions().timeZone)} className="mt-2 text-xs font-semibold text-primary">Use this device's time zone</button></div>
      <label className="flex items-center gap-3 text-sm"><input type="checkbox" checked={quiet} onChange={(e) => setQuiet(e.target.checked)} className="h-4 w-4 accent-primary" />Keep some hours quiet</label>
      {quiet && <div className="grid grid-cols-2 gap-3"><label className="text-xs text-ink-soft">Quiet from<input type="time" value={start} onChange={(e) => setStart(e.target.value)} required className="form-input mt-2" /></label><label className="text-xs text-ink-soft">Until<input type="time" value={end} onChange={(e) => setEnd(e.target.value)} required className="form-input mt-2" /></label></div>}
      <button disabled={Boolean(busy) || !tz.trim()} className="button-soft">{busy === 'save' ? 'Saving…' : 'Save preferences'}</button>
      {prompt.settings.enabled && <button type="button" disabled={Boolean(busy)} onClick={() => run('pause', async () => { const value = await community.promptSettings(PID, { pause_today: !prompt.settings.paused_today }); setPrompt(value); setNotice(value.settings.paused_today ? 'A quiet day today. Your prompt returns tomorrow.' : 'Your daily prompt is back on today.') })} className="text-xs font-medium text-primary underline underline-offset-4">{prompt.settings.paused_today ? 'Resume today’s prompt' : 'Pause just for today'}</button>}
      <p className="text-xs leading-relaxed text-ink-soft">{prompt.state === 'off' ? 'Daily prompts are off. You can still check in whenever you like.' : prompt.state === 'paused' ? 'Your prompt is paused today.' : prompt.state === 'answered' ? 'You have answered today’s prompt.' : prompt.at_local ? `Today’s prompt: ${prompt.at_local} in ${prompt.settings.tz}.` : 'No prompt is scheduled for today.'}</p>
    </form> : <button onClick={load} className="mt-4 text-sm font-semibold text-primary">{error ? 'Retry settings' : 'Loading preferences…'}</button>}
    <div className="mt-5 border-t border-outline-soft/35 pt-5"><h3 className="text-sm font-semibold">Notifications on this device</h3><p className="mt-2 text-xs leading-relaxed text-ink-soft">{!supported ? 'Push notifications need a supported browser in a secure context. The in-app prompt still works.' : push?.enabled ? 'We only ask for browser permission when you choose to turn these on.' : 'Push notifications need the team’s server setup. Your preferences and in-app prompt still work.'}</p>
      <div className="mt-3 flex flex-wrap gap-2"><button type="button" disabled={Boolean(busy) || !supported || !push?.enabled} onClick={endpoint ? unsubscribe : subscribe} className="button-soft text-xs">{busy === 'push' ? 'Working…' : endpoint ? 'Turn off on this device' : 'Turn on notifications'}</button>{endpoint && <button disabled={Boolean(busy)} onClick={() => run('test', async () => { const result = await community.testPush(PID); setNotice(result.sent ? 'A test notification was sent to your registered devices.' : 'No device received the test. Try reconnecting notifications.') })} className="button-soft text-xs">Send a test</button>}</div>
    </div>
    <details className="mt-5 text-xs text-ink-soft"><summary className="cursor-pointer font-medium">Demo controls</summary><button type="button" disabled={Boolean(busy)} onClick={() => run('demo', async () => { const value = await community.promptNow(PID); setPrompt(value); window.dispatchEvent(new Event('steady:engagement')); setNotice('Today’s prompt is ready. Open Check in to read the phrase and record your day.') })} className="button-soft mt-3 text-xs">Send today’s prompt now</button></details>
    {notice && <p role="status" className="mt-4 text-sm text-primary">{notice}</p>}{error && <p role="alert" className="mt-4 text-sm text-terracotta">{error}</p>}
  </Card>
}
