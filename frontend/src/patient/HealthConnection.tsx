import { useCallback, useEffect, useState } from 'react'
import { Icon } from '../ui/kit'
import { PID } from './Shell'

interface HealthStatus { connected: boolean; configured: boolean; last_sync_at: string | null; last_day: string | null; source: 'google_health' | 'demo' | null; days_synced: number; error: string | null }
interface SyncResult { days_synced: number; skipped: number; rewards: { amount: number }[]; warnings: string[]; status: HealthStatus }
export default function HealthConnection() {
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const call = useCallback(async <T,>(path = '', method = 'GET'): Promise<T> => {
    const res = await fetch(`/api/patients/${PID}/health${path}`, { method })
    const out = await res.json()
    if (!res.ok) throw new Error(typeof out.detail === 'string' ? out.detail : 'Could not connect to your health data.')
    return out as T
  }, [])
  const refresh = useCallback(() => { call<HealthStatus>().then((status) => { setHealth(status); setError('') }).catch((e) => setError(String(e.message))) }, [call])
  useEffect(() => {
    refresh()
    const result = new URLSearchParams(window.location.search).get('health')
    if (result === 'connected') setNotice('Google Health connected. Tap Sync now to bring your latest completed Fitbit day into your habits.')
    if (result === 'cancelled') setNotice('Google Health sign-in was cancelled. You can connect whenever you are ready.')
    if (result === 'error') setNotice('Google Health sign-in could not finish. Try connecting again below.')
    if (result) {
      const url = new URL(window.location.href)
      url.searchParams.delete('health')
      window.history.replaceState(window.history.state, '', url.pathname + url.search + url.hash)
    }
  }, [refresh])
  const run = async (action: 'connect' | 'sync' | 'demo' | 'disconnect') => {
    setBusy(true); setError(''); setNotice('')
    try {
      if (action === 'connect') {
        const r = await call<{ authorization_url: string }>('/connect', 'POST')
        window.location.assign(r.authorization_url)
        return
      }
      if (action === 'disconnect') { await call('/connection', 'DELETE'); refresh(); setNotice('Google Health disconnected. Your recorded habits stay yours.') }
      else {
        const r = await call<SyncResult>(`/${action}`, 'POST')
        setHealth(r.status)
        const food = r.rewards.reduce((n, x) => n + x.amount, 0)
        setNotice(`${action === 'demo' ? 'Demo day added' : r.days_synced ? `${r.days_synced} health day${r.days_synced > 1 ? 's' : ''} synced` : 'You’re up to date'}.${food ? ` +${food} pet food.` : ''}${r.warnings.length ? ` ${r.warnings.join(' ')}` : ''}`)
        window.dispatchEvent(new Event('steady:engagement'))
      }
    } catch (e) { setError(e instanceof Error ? e.message : 'Could not sync. Please try again.') }
    finally { setBusy(false) }
  }
  return <section className="rounded-[1.75rem] bg-sage-soft/60 p-5">
    <div className="flex items-center gap-3"><span className="grid h-11 w-11 place-items-center rounded-2xl bg-white/70 text-primary"><Icon name="watch" /></span><div><h2 className="font-semibold">{health?.connected ? 'Your Fitbit, connected' : 'Your Fitbit'}</h2><p className="mt-0.5 text-xs text-ink-soft">Sleep, steps and active minutes</p></div></div>
    <p className="mt-3 text-sm leading-relaxed text-ink-soft">{health?.connected ? 'Your latest completed Fitbit day syncs into your habits through Google Health.' : 'Connect through Google Health to let your wearable check off the little things.'}</p>
    {health?.last_day && <p className="mt-3 text-xs text-primary">{health.source === 'demo' ? 'Demo health data' : 'Latest health day'} · {new Date(`${health.last_day}T12:00:00`).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })}</p>}
    {health ? <>
      {health.connected ? <div className="mt-4 flex gap-2"><button disabled={busy} onClick={() => run('sync')} className="button-primary flex-1">{busy ? 'Working…' : 'Sync now'}</button><button disabled={busy} onClick={() => run('disconnect')} className="button-soft">Disconnect</button></div> : <button disabled={busy || !health.configured} onClick={() => run('connect')} className="button-primary mt-4 w-full">{busy ? 'Connecting…' : 'Connect Google Health'}</button>}
      {!health.connected && !health.configured && <p className="mt-2 text-xs text-ink-soft">Google Health sign-in needs the team's OAuth setup. You can still track hobbies and check in.</p>}
      <details className="mt-4 text-xs text-ink-soft"><summary className="cursor-pointer font-medium">Try the demo</summary><p className="mt-2 leading-relaxed">Adds a generated wearable day so you can try automatic habits. It is labelled as demo data.</p><button disabled={busy} onClick={() => run('demo')} className="mt-3 rounded-full border border-primary/25 px-4 py-2 font-semibold text-primary">Try demo health data</button></details>
    </> : !error ? <p className="mt-4 text-sm text-ink-soft">Checking your connection…</p> : <button onClick={refresh} className="mt-3 text-sm font-semibold text-primary">Retry connection status</button>}
    {notice && <p role="status" className="mt-4 text-sm font-medium text-primary">{notice}</p>}
    {(error || health?.error) && <p role="alert" className="mt-3 text-sm text-terracotta">{error || health?.error}</p>}
  </section>
}
