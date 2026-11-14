export interface SocialProfile {
  mode: 'profile' | 'anonymous'; display_name: string | null; pseudonym: string | null
  photo_url: string | null; emoji: string | null
  share_avatar: boolean; share_level: boolean; share_streak: boolean; share_checked_in: boolean
  social_consent: boolean; crisis?: boolean
}
export interface CircleSummary {
  circle_id: number; name: string; invite_code: string; member_id: number; token: string
  nudge?: { pending: boolean; visible_from: string; can_cancel: boolean; message: string } | null
}
export interface CircleMember {
  member_id: number; role: 'patient' | 'friend'; name: string; is_you: boolean
  photo_url: string | null; emoji: string | null
  avatar: { patient_id: number; name?: string | null; colour?: string | null; items_json?: string | null; tier?: number } | null
  level: number | null; streak: number | null; checked_in_today: boolean | null
}
export interface CircleFeed {
  circle: { id: number; name: string; invite_code: string }; you: number
  members: CircleMember[]
  reactions: { id: number; from_member_id: number; from_name: string; to_member_id: number; kind: string; label: string; ts: string }[]
  reach_out_cards: { to_member_id: number; message: string }[]
}
export interface PromptStatus {
  date: string; state: 'off' | 'paused' | 'none' | 'scheduled' | 'open' | 'answered'
  open: boolean; at: string | null; at_local: string | null; window_minutes: number
  on_time_left_s: number | null; answered_at: string | null; on_time: boolean | null; source: string | null
  settings: { tz: string; window_start: string; window_end: string; quiet_start: string | null; quiet_end: string | null; enabled: boolean; paused_today: boolean }
}
export interface PushStatus { enabled: boolean; public_key: string | null; devices?: number }

export async function communityRequest<T>(path: string, method = 'GET', body?: unknown): Promise<T> {
  const res = await fetch(`/api${path}`, { method, ...(body === undefined ? {} : { headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) }) })
  const out = await res.json().catch(() => null)
  if (!res.ok) throw new Error(typeof out?.detail === 'string' ? out.detail : `Could not save that (${res.status}). Please try again.`)
  if (method !== 'GET') window.dispatchEvent(new Event('steady:engagement'))
  return out as T
}
const patient = (pid: number, path: string) => `/patients/${pid}${path}`
export const community = {
  profile: (pid: number) => communityRequest<SocialProfile>(patient(pid, '/profile')),
  saveProfile: (pid: number, body: Partial<SocialProfile>) => communityRequest<SocialProfile>(patient(pid, '/profile'), 'PUT', body),
  presets: () => communityRequest<{ reactions: { kind: string; label: string }[] }>('/social/presets'),
  circle: async (pid: number): Promise<CircleSummary | null> => {
    const res = await fetch(`/api${patient(pid, '/circle')}`)
    if (res.status === 404) return null
    if (!res.ok) throw new Error('Could not load your circle. Please try again.')
    return res.json() as Promise<CircleSummary>
  },
  create: (pid: number, name?: string) => communityRequest<CircleSummary>(patient(pid, '/circle'), 'POST', { name }),
  join: (code: string, identity: { patient_id: number } | { name: string }) => communityRequest<CircleSummary>('/circles/join', 'POST', { code, ...identity }),
  leave: (token: string) => communityRequest<{ ok: boolean }>('/circles/leave', 'POST', { token }),
  feed: (token: string) => communityRequest<CircleFeed>(`/circles/feed?token=${encodeURIComponent(token)}`),
  react: (token: string, to_member_id: number, kind: string) => communityRequest<{ ok: boolean }>('/circles/react', 'POST', { token, to_member_id, kind }),
  cancelNudge: (pid: number) => communityRequest<{ ok: boolean }>(patient(pid, '/circle/nudge/cancel'), 'POST'),
  prompt: (pid: number) => communityRequest<PromptStatus>(patient(pid, '/prompt')),
  promptNow: (pid: number) => communityRequest<PromptStatus>(patient(pid, '/prompt/now'), 'POST'),
  promptSettings: (pid: number, body: { tz?: string; quiet_start?: string | null; quiet_end?: string | null; enabled?: boolean; pause_today?: boolean }) => communityRequest<PromptStatus>(patient(pid, '/prompt/settings'), 'PUT', body),
  push: (pid: number) => communityRequest<PushStatus>(patient(pid, '/push')),
  subscribe: (pid: number, subscription: PushSubscriptionJSON) => communityRequest<PushStatus>(patient(pid, '/push/subscribe'), 'POST', subscription),
  unsubscribe: (pid: number, endpoint: string) => communityRequest<PushStatus>(patient(pid, '/push/unsubscribe'), 'POST', { endpoint }),
  testPush: (pid: number) => communityRequest<{ sent: number }>(patient(pid, '/push/test'), 'POST'),
}
