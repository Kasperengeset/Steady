// Mirrors backend/app/schemas.py – change both together.
import { toWav16k } from './wav'

export type Slot = 'morning' | 'evening'

export interface FeatureOut { name: string; value: number; source: string }
export interface StateOut { S: number | null; level: number; branches: Record<string, number> }
export interface InterventionOut { arm_key: string; label: string }
export interface CheckInResponse {
  checkin_id: number
  features: FeatureOut[]
  state: StateOut
  intervention: InterventionOut
  crisis: boolean
}
export interface PatientSummary { id: number; name: string; S: number | null; level: number; trend: number[] }
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

  feedback: (patientId: number, date: string, completed: boolean, rating: number | null) =>
    fetch('/api/interventions/feedback', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ patient_id: patientId, date, completed, rating }),
    }).then(json<{ reward: number }>),

  patients: () => fetch('/api/patients').then(json<PatientSummary[]>),
  patient: (id: number) => fetch(`/api/patients/${id}`).then(json<PatientDetail>),
  friendHint: (patientId: number) => fetch(`/api/friends/${patientId}/hint`).then(json<FriendHint>),
  simulate: (days = 60, seed = 0) =>
    fetch('/api/sim/run', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ days, seed }),
    }).then(json<{ day: number; S: number; self_mood: number }[]>),
}

/** Backend stores dates in UTC. */
export const todayUTC = () => new Date().toISOString().slice(0, 10)
