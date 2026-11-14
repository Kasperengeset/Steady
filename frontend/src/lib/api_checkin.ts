// Daily moment (#21 fixed phrase + memo, #22 daily prompt). Mirrors backend/app/schemas_prompt.py and
// the `prompt` / `phrase` fields on CheckInResponse in backend/app/schemas.py – change both together.
import type { CheckInResponse, Slot } from './api'
import { toWav16k } from './wav'

export type PromptState = 'off' | 'paused' | 'none' | 'scheduled' | 'open' | 'answered'

export interface PromptSettings {
  tz: string
  window_start: string // "HH:MM" local
  window_end: string
  quiet_start: string | null
  quiet_end: string | null
  enabled: boolean
  paused_today: boolean
}
export interface PromptStatus {
  date: string
  state: PromptState
  open: boolean // fired and not answered: show "Your check-in is ready"
  at: string | null
  at_local: string | null
  window_minutes: number
  on_time_left_s: number | null // while open: seconds left of the on-time window (0 = late, still counts fully)
  answered_at: string | null
  on_time: boolean | null
  source: 'random' | 'demo' | null
  settings: PromptSettings
}
export interface PromptSettingsIn {
  tz?: string
  quiet_start?: string | null
  quiet_end?: string | null
  enabled?: boolean
  pause_today?: boolean
}
export interface PromptAnswer { answered: boolean; on_time: boolean; prompt_at: string; late_by_s: number }
export interface PhraseCheck {
  phrase_id: string
  ok: boolean
  reason: string | null
  wer?: number | null
  syllables_per_s?: number | null
  span_s?: number
  checked_by?: 'transcript' | 'timing'
}
export type DailyCheckInResponse = CheckInResponse & { prompt: PromptAnswer | null; phrase: PhraseCheck | null }

/** Must match PHRASE in backend/app/pipeline/phrase.py (Rainbow Passage). Never change it: a new phrase means a new baseline. */
export const PHRASE =
  'When the sunlight strikes raindrops in the air, they act as a prism and form a rainbow. ' +
  'The rainbow is a division of white light into many beautiful colors.'

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`)
  return res.json() as Promise<T>
}

export interface PushStatus { enabled: boolean; public_key: string | null; devices: number | null }

export const checkinApi = {
  /** Web Push (#22, P2): enabled = the server has VAPID keys. */
  push: (patientId: number) => fetch(`/api/patients/${patientId}/push`).then(json<PushStatus>),
  pushSubscribe: (patientId: number, sub: PushSubscriptionJSON) =>
    fetch(`/api/patients/${patientId}/push/subscribe`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(sub),
    }).then(json<PushStatus>),
  pushUnsubscribe: (patientId: number, endpoint: string) =>
    fetch(`/api/patients/${patientId}/push/unsubscribe`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ endpoint }),
    }).then(json<PushStatus>),
  pushTest: (patientId: number) =>
    fetch(`/api/patients/${patientId}/push/test`, { method: 'POST' }).then(json<{ sent: number }>),

  prompt: (patientId: number) => fetch(`/api/patients/${patientId}/prompt`).then(json<PromptStatus>),

  /** Demo: "Send today's prompt now", so the video does not wait for a random time. */
  promptNow: (patientId: number) =>
    fetch(`/api/patients/${patientId}/prompt/now`, { method: 'POST' }).then(json<PromptStatus>),

  promptSettings: (patientId: number, body: PromptSettingsIn) =>
    fetch(`/api/patients/${patientId}/prompt/settings`, {
      method: 'PUT',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    }).then(json<PromptStatus>),

  /** One upload, two files: `phrase` (the fixed reading, optional) and `audio` (the 30 s memo). */
  checkIn: async (patientId: number, slot: Slot, memo: Blob, phrase: Blob | null) => {
    const fd = new FormData()
    fd.append('audio', await toWav16k(memo), 'memo.wav')
    if (phrase) {
      // A phrase that fails to decode must not cost the person their memo: send the memo alone
      const wav = await toWav16k(phrase).catch(() => null)
      if (wav) fd.append('phrase', wav, 'phrase.wav')
    }
    fd.append('patient_id', String(patientId))
    fd.append('slot', slot)
    return fetch('/api/checkins', { method: 'POST', body: fd }).then(json<DailyCheckInResponse>)
  },
}
