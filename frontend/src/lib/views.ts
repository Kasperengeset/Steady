// Read endpoints for the patient app and the clinician portal (backend/app/routers/views.py, app/insight.py).
import type { BaselineDetail, BaselineOut } from './api'

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`)
  return res.json() as Promise<T>
}
const post = (url: string, body: unknown = {}) =>
  fetch(url, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify(body) })

export type Status = 'building' | 'close' | 'different' | 'care'

export interface Today {
  patient: { id: number; name: string }
  date: string
  status: Status
  status_text: string
  baseline: BaselineOut | null
  series: { date: string; S: number }[]
  action: { date: string; arm_key: string; label: string; reason: string; completed: boolean | null; rating: number | null } | null
  pending_followup: { date: string; arm_key: string; label: string } | null
  last_sync: string | null
}

export interface ArmInsight {
  key: string
  label: string
  suggested: number
  done: number
  skipped: number
  tried: number // answered: done or skipped
  helped: number
  mean_rating: number | null
  learned: number
  fits_when: string[]
  history: { date: string; completed: boolean | null; rating: number | null }[]
}

export interface Contact { id: number; name: string; kind: 'friend' | 'clinician' }

export interface Notification {
  id: string
  patient_id: number
  patient: string
  kind: 'crisis' | 'clinic' | 'friends' | 'baseline'
  level: number
  message: string
  quote: string | null
  ts: string
  handled: boolean
}

export interface Timeline {
  patient: { id: number; name: string; level: number; status: Status; days_since_start: number | null }
  days: { date: string; S: number | null; S_anchored: number | null; S_rolling: number | null; level: number;
          checkin: boolean; action_done: boolean; rating: number | null }[]
  feelings: { ts: string; slot: string; text: string; tone: string | null; crisis: boolean }[]
  baseline: BaselineOut | null
  what_helps: ArmInsight[]
}

export type SignalFormat = 'int' | 'pct' | 'clock' | 'h' | 'bpm' | 'ms' | 'num'
export interface SignalRow {
  feature: string
  name: string
  technical: string
  format: SignalFormat
  status: 'worse' | 'better' | 'same' | 'no_data'
  usual?: number
  now?: number
  z?: number
  series?: number[]
}
export interface SourceExplain {
  source: 'health' | 'acoustic' | 'text'
  label: string
  mean_deviation: number | null
  share: number
  direction: 'worse' | 'better' | 'same' | null
  summary: string
  signals: SignalRow[]
}
export interface Explain {
  available: boolean
  message?: string
  days: number
  summary: string
  data_note: string
  sources: SourceExplain[]
  words: { stand_out: { word: string; date: string }[]; quote: { date: string; text: string; highlights: string[] } | null }
  habits: { per_week_usual: number; per_week_now: number; words_usual: number | null; words_now: number | null; days_since_last_checkin: number }
}

export const views = {
  today: (pid: number) => fetch(`/api/patients/${pid}/today`).then(json<Today>),
  insights: (pid: number) => fetch(`/api/patients/${pid}/insights`).then(json<ArmInsight[]>),
  contacts: (pid: number) => fetch(`/api/patients/${pid}/contacts`).then(json<Contact[]>),
  notifications: (status: 'open' | 'handled' | 'all') =>
    fetch(`/api/clinician/notifications?status=${status}`).then(json<Notification[]>),
  markHandled: (id: string) => post(`/api/clinician/notifications/${id}/handled`).then(json<{ ok: boolean }>),
  timeline: (pid: number, days: number) => fetch(`/api/patients/${pid}/timeline?days=${days}`).then(json<Timeline>),
  explain: (pid: number, days: number) => fetch(`/api/patients/${pid}/explain?days=${days}`).then(json<Explain>),
  confirmBaseline: (pid: number, clinician: string) =>
    post(`/api/patients/${pid}/baseline/confirm`, { clinician }).then(json<BaselineDetail>),
}
