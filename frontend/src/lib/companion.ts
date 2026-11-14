import { useCallback, useEffect, useState } from 'react'

export interface Reward { reason: string; label: string; amount: number }
export interface Habit { key: string; label: string; icon: string; done: boolean; auto: boolean; food: number; hint: string }
export interface CompanionView {
  patient: { id: number; name: string }
  name: string; mood: string; message: string; message_kind: string
  daily_phrase?: string
  level: number; title: string; tier: number; xp: number; xp_level_start: number; xp_next: number | null
  next_unlock: string | null; food: number
  streak: { current: number; longest: number; today: boolean; days_since_last: number | null }
  prompt: { at: string | null; answered: boolean; open: boolean; on_time_until: string | null; bonus_available: boolean; minutes_left: number; done_before_prompt: boolean; state?: string; settings?: { tz: string }; phrase?: string }
  habits: Habit[]
  week: { date: string; checked_in: boolean; done: number; total: number }[]
  earned_today: Reward[]; earned_today_total: number
  profile: { display_name: string; anonymous: boolean }
}
export interface FeedResult { fed: number; food: number; xp: number; level: number; leveled_up: boolean; tier: number; tier_up: boolean; title: string }
export interface CircleView {
  circle: string | null
  members: { id: number; me: boolean; name: string; anonymous: boolean; avatar_name: string; level: number; tier: number; streak: number; checked_in_today: boolean; mood: string; cheered_today: boolean }[]
  cheers: Record<string, string>
  inbox: { from: string; kind: string; text: string; ts: string }[]
}

async function request<T>(pid: number, path: string, method = 'GET', body?: unknown): Promise<T> {
  const response = await fetch(`/api/patients/${pid}${path}`, {
    method, ...(body === undefined ? {} : { headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }),
  })
  if (!response.ok) {
    if (response.status === 404 && path === '/appie') throw new Error('Appie is not available from this backend. Restart Steady with scripts/run_local.ps1.')
    const error = await response.json().catch(() => null)
    throw new Error(typeof error?.detail === 'string' ? error.detail : `Could not save that (${response.status}). Please try again.`)
  }
  const data = await response.json() as T
  if (method !== 'GET') window.dispatchEvent(new Event('steady:engagement'))
  return data
}

export const engagement = {
  companion: (pid: number) => request<CompanionView>(pid, '/appie'),
  feed: (pid: number, amount: number, request_id?: string) => request<FeedResult>(pid, '/appie/feed', 'POST', { amount, request_id }),
  addHabit: (pid: number, label: string) => request<{ key: string; label: string }>(pid, '/appie/habits', 'POST', { label }),
  renameHabit: (pid: number, key: string, label: string) => request<{ key: string; label: string }>(pid, `/appie/habits/${encodeURIComponent(key)}`, 'PUT', { label }),
  tickHabit: (pid: number, key: string) => request<{ ok: boolean; reward: Reward | null }>(pid, `/appie/habits/${encodeURIComponent(key)}/tick`, 'POST'),
  removeHabit: (pid: number, key: string) => request<{ ok: boolean }>(pid, `/appie/habits/${encodeURIComponent(key)}`, 'DELETE'),
  profile: (pid: number, body: { avatar_name?: string }) => request<{ avatar_name: string }>(pid, '/appie/profile', 'PUT', body),
}

export function useCompanion(pid: number) {
  const [data, setData] = useState<CompanionView | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const refresh = useCallback(async () => {
    try { setData(await engagement.companion(pid)); setError(null) }
    catch (e) { setError(e instanceof Error ? e.message : 'Could not reach Appie. Please try again.') }
    finally { setLoading(false) }
  }, [pid])
  useEffect(() => {
    void refresh()
    const handler = () => { void refresh() }
    window.addEventListener('steady:engagement', handler)
    const timer = window.setInterval(handler, 60000)
    return () => { window.removeEventListener('steady:engagement', handler); window.clearInterval(timer) }
  }, [refresh])
  return { data, error, loading, refresh }
}

export function localTime(iso: string | null, timeZone?: string) {
  if (!iso) return 'when you choose'
  return new Date(iso).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', ...(timeZone ? { timeZone } : {}) })
}

export function promptTitle(prompt: CompanionView['prompt']) {
  if (prompt.answered) return 'Your daily moment is complete'
  if (prompt.state === 'off') return 'Daily prompts are off'
  if (prompt.state === 'paused') return 'Your prompt is paused today'
  if (prompt.open) return 'Your daily moment is here'
  if (!prompt.at) return 'No prompt is scheduled for today'
  return `Your daily moment arrives at ${localTime(prompt.at, prompt.settings?.tz)}`
}
