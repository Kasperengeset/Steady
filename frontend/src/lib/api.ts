// Mirrors backend/app/schemas.py – change both together.
import { toWav16k } from './wav'

export type Slot = 'morning' | 'evening'

export interface FeatureOut { name: string; value: number; source: string }
export interface StateOut {
  S: number | null // max(S_anchored, S_rolling); null while the healthy baseline is building
  level: number
  branches: Record<string, number>
  S_anchored: number | null // vs fixed healthy baseline – slow drift
  S_rolling: number | null // vs last 14 days, same slot – abrupt change
  driver: 'anchored' | 'rolling' | null
}
export interface BaselineOut {
  version: number
  status: 'building' | 'active' | 'retired' | 'rejected'
  started_at: string
  locked_at: string | null
  period_days: number
  days_elapsed: number
  checkins: Record<string, number>
  min_checkins: number
  confirmed_by: string | null
}
export interface BaselineDetail {
  active: BaselineOut | null
  building: BaselineOut | null
  stats: Record<string, Record<string, { center: number; scale: number; n: number }>>
}
export interface InterventionOut { arm_key: string; label: string }
export interface CheckInResponse {
  checkin_id: number
  features: FeatureOut[]
  state: StateOut
  baseline: BaselineOut
  intervention: InterventionOut
  crisis: boolean
}
export interface PatientSummary {
  id: number
  name: string
  S: number | null
  level: number
  trend: number[]
  driver: 'anchored' | 'rolling' | null
  baseline_status: 'building' | 'active' | null
}
export interface PatientDetail {
  patient: Record<string, unknown>
  scores: Record<string, unknown>[]
  checkins: Record<string, unknown>[]
  top_features: { branch: string; feature: string; contribution: number }[]
}
export interface FriendHint { show_hint: boolean; message: string }

async function json<T>(res: Response): Promise<T> {
  if (!res.ok) throw new Error(`${res.status} ${await res.text()}`)
  return res.json() as Promise<T>
}

export const api = {
  health: () => fetch('/api/health').then(json<{ ok: boolean }>),

  checkIn: async (patientId: number, slot: Slot, audio: Blob) => {
    const fd = new FormData()
    fd.append('audio', await toWav16k(audio), 'checkin.wav')
    fd.append('patient_id', String(patientId))
    fd.append('slot', slot)
    return fetch('/api/checkins', { method: 'POST', body: fd }).then(json<CheckInResponse>)
  },

  /** Written diary entry – text branch only. English, 50 to 100 words works best. */
  checkInText: (patientId: number, slot: Slot, text: string) =>
    fetch('/api/checkins/text', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ patient_id: patientId, slot, text }),
    }).then(json<CheckInResponse>),

  feedback: (patientId: number, date: string, completed: boolean, rating: number | null) =>
    fetch('/api/interventions/feedback', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ patient_id: patientId, date, completed, rating }),
    }).then(json<{ reward: number }>),

  patients: () => fetch('/api/patients').then(json<PatientSummary[]>),
  patient: (id: number) => fetch(`/api/patients/${id}`).then(json<PatientDetail>),
  baseline: (id: number) => fetch(`/api/patients/${id}/baseline`).then(json<BaselineDetail>),
  baselineAction: (id: number, action: 'start' | 'reject', body: { period_days?: number; min_checkins?: number; note?: string } = {}) =>
    fetch(`/api/patients/${id}/baseline/${action}`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(body),
    }).then(json<BaselineDetail>),
  friendHint: (patientId: number) => fetch(`/api/friends/${patientId}/hint`).then(json<FriendHint>),
  simulate: (days = 70, seed = 0, relapse_day = 40) =>
    fetch('/api/sim/run', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ days, seed, relapse_day }),
    }).then(json<SimResult>),
  simRobustness: (seeds = 50) => fetch(`/api/sim/robustness?seeds=${seeds}`).then(json<SimRobustness>),
}

export interface SimDay {
  day: number
  latent: number
  self_mood: number
  S_rolling: number | null
  S_anchored: number | null
  level_rolling: number
  level_anchored: number
  arm: string
  completed: boolean
  reward: number
  features: Record<string, number>
}
type ByBaseline<T> = { rolling: T; anchored: T }
export interface SimResult {
  days: SimDay[]
  summary: {
    relapse_starts: number
    self_report_flag_day: number | null
    engine_flag_day: ByBaseline<number | null>
    lead_days_vs_self_report: ByBaseline<number | null>
    false_alarm_days_before_relapse: ByBaseline<number>
    arms: Record<string, { label: string; true: number; learned: number; times_chosen: number }>
    bandit_top_arm: string
    true_best_arm: string
  }
  assumptions: Record<string, string>
  weights_trained: boolean
}
export interface SimRobustness {
  seeds: number
  median_lead_days: ByBaseline<number | null>
  missed_relapses: ByBaseline<number>
  mean_false_alarm_days: ByBaseline<number>
  bandit_finds_best_arm: string
}

/** Backend stores dates in UTC. */
export const todayUTC = () => new Date().toISOString().slice(0, 10)
