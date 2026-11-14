// Rewards and habits (#24), the circle (#26) and the avatar's check-ins (#25).
// Mirrors backend/app/rewards.py, routers/rewards.py, routers/social.py and routers/companion.py.
// No health data travels through any of these: no scores, features, diaries or transcripts.

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`)
  return res.json() as Promise<T>
}

const post = (url: string, body?: unknown) =>
  fetch(url, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  })

// ---------------------------------------------------------------- rewards and habits

export interface LedgerRow { ts: string; date: string; reason: string; label: string; amount: number }
export interface LevelHint { level: number; xp: number; next_level_xp: number | null }
export interface Rewards {
  balance: number
  xp: number // total ever earned; spending never lowers it
  level_hint: LevelHint
  streak: number // gentle: one rest day a week keeps it
  longest_streak: number // never lost
  checked_in_today: boolean
  days_since_last_checkin: number | null
  ledger: LedgerRow[]
}
export interface HabitRow { key: 'memo' | 'action' | 'followup'; label: string; done: boolean[] }
export interface TodayAction { arm_key: string; label: string; answered: boolean; done: boolean }
export interface HabitWeek {
  days: string[] // ISO dates, oldest first, today last
  habits: HabitRow[]
  earned: number[]
  today_action: TodayAction | null // null until today's first check-in picks it
}
export interface Earned { reason: string; label: string; amount: number }
export interface ActionResult { bandit_reward: number; earned: Earned[]; balance: number }

// ---------------------------------------------------------------- the avatar's check-ins

export type CompanionKind = 'missed_one_after_streak' | 'missed_few' | 'short_memos' | 'back_after_break'
export interface Companion {
  message: string | null // at most one a day; null when there is nothing to say
  kind: CompanionKind | null
  at: string | null
  id: number | null
  offer_written: boolean // show "type instead"
  offer_action: boolean // point to today's small thing
  glad_to_see_you: boolean // the avatar's "glad to see you" state (#23)
}

// ---------------------------------------------------------------- the circle

export type IdentityMode = 'profile' | 'anonymous'
export interface Profile {
  mode: IdentityMode
  display_name: string | null
  pseudonym: string | null
  photo_url: string | null
  emoji: string | null
  share_avatar: boolean
  share_level: boolean
  share_streak: boolean
  share_checked_in: boolean
  social_consent: boolean
}
export type ProfileIn = Partial<Omit<Profile, 'social_consent'>> & { social_consent?: boolean }
export interface MyCircle {
  circle_id: number
  name: string
  invite_code: string
  member_id: number
  token: string
  nudge: { pending: boolean; visible_from: string; can_cancel: boolean; message: string } | null
}
export interface AvatarRef { patient_id: number; name?: string | null; colour?: string | null; items_json?: string | null }
export interface MemberCard {
  member_id: number
  role: 'patient' | 'friend'
  name: string
  is_you: boolean
  photo_url: string | null
  emoji: string | null
  // null = not shared (each is opt-in)
  avatar: AvatarRef | null
  level: number | null
  streak: number | null
  checked_in_today: boolean | null
}
export interface ReactionOut {
  id: number
  from_member_id: number
  from_name: string
  to_member_id: number
  kind: string
  label: string
  ts: string
}
export interface Feed {
  circle: { id: number; name: string; invite_code: string }
  you: number
  members: MemberCard[]
  reactions: ReactionOut[]
  reach_out_cards: { to_member_id: number; message: string }[]
}
export interface Preset { kind: string; label: string }
export interface Joined { circle_id: number; name: string; member_id: number; token: string }

export const engagementApi = {
  rewards: (pid: number) => fetch(`/api/patients/${pid}/rewards`).then(json<Rewards>),
  week: (pid: number, days = 7) => fetch(`/api/patients/${pid}/habits/week?days=${days}`).then(json<HabitWeek>),
  /** Today's small thing. done=false is "not today": an honest answer that still earns. Feeds the bandit. */
  markAction: (pid: number, done: boolean, rating: number | null = null) =>
    post(`/api/patients/${pid}/habits/action`, { done, rating }).then(json<ActionResult>),

  companion: (pid: number) => fetch(`/api/patients/${pid}/companion`).then(json<Companion>),
  companionDemo: (pid: number) => post(`/api/patients/${pid}/companion/demo`).then(json<Companion>),
  companionSeen: (pid: number, messageId: number) =>
    post(`/api/patients/${pid}/companion/seen`, { message_id: messageId }).then(json<{ ok: boolean }>),
  /** Crisis words in `text` go to the clinician at once; open the support screen when crisis is true. */
  companionReply: (pid: number, messageId: number, text: string | null) =>
    post(`/api/patients/${pid}/companion/reply`, { message_id: messageId, text }).then(
      json<{ ok: boolean; crisis: boolean }>,
    ),

  profile: (pid: number) => fetch(`/api/patients/${pid}/profile`).then(json<Profile>),
  saveProfile: (pid: number, body: ProfileIn) =>
    fetch(`/api/patients/${pid}/profile`, {
      method: 'PUT',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    }).then(json<Profile & { crisis: boolean }>),
  /** 404 when the person has no circle yet. */
  myCircle: async (pid: number): Promise<MyCircle | null> => {
    const r = await fetch(`/api/patients/${pid}/circle`)
    return r.status === 404 ? null : json<MyCircle>(r)
  },
  createCircle: (pid: number, name?: string) => post(`/api/patients/${pid}/circle`, { name }).then(json<MyCircle>),
  cancelNudge: (pid: number) => post(`/api/patients/${pid}/circle/nudge/cancel`).then(json<{ ok: boolean }>),
  join: (code: string, name: string | null, patientId: number | null = null) =>
    post('/api/circles/join', { code, name, patient_id: patientId }).then(json<Joined>),
  leave: (token: string) => post('/api/circles/leave', { token }).then(json<{ ok: boolean }>),
  feed: (token: string) => fetch(`/api/circles/feed?token=${encodeURIComponent(token)}`).then(json<Feed>),
  react: (token: string, toMemberId: number, kind: string) =>
    post('/api/circles/react', { token, to_member_id: toMemberId, kind }).then(json<{ ok: boolean; id: number }>),
  presets: () => fetch('/api/social/presets').then(json<{ reactions: Preset[] }>),
}
