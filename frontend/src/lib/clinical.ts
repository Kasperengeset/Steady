import { useCallback, useEffect, useRef, useState } from 'react'
import type { BaselineOut, Explanation } from './api'

export type ClinicalOrigin = 'live' | 'synthetic' | 'imported' | 'legacy' | string
export interface ClinicalSource { key: string; label: string; last_at?: string | null; last_live_at?: string | null; entries?: number; live_entries?: number; synthetic_entries?: number; scored_entries?: number; scored?: number | boolean; feature_count?: number; available?: boolean; reason?: string | null }
export interface ModelRecord {
  key: string; label: string; status: string; artifact_sha256?: string | null
  validation?: Record<string, unknown> | null; limitations?: string[]
  features?: { name: string; weight: number; direction: number }[]; dataset?: string; description?: string; artifact_path?: string
}
export interface CurrentState {
  S: number | null; D?: number | null; level: number; elevated?: boolean; driver?: string | null; date?: string | null
  S_anchored?: number | null; S_rolling?: number | null; branches?: Record<string, number | null>
}
export interface ClinicalPatient {
  id: number; name: string; demo_key?: string | null; data_origin?: ClinicalOrigin; needs_attention?: boolean
  level?: number
  current?: CurrentState | null; baseline?: BaselineOut | null; created_at?: string
  last_entry_at?: string | null; last_live_entry_at?: string | null
  entry_counts?: { voice: number; text: number; wearable: number; total?: number }
  sources?: ClinicalSource[]; days_since_start?: number | null
}
export interface ClinicalOverview {
  generated_at: string
  summary: { patients: number; attention: number; baseline_building: number; entries_today: number; live_entries_today: number }
  patients: ClinicalPatient[]; models: ModelRecord[]; limitations?: string[]
}
export interface ClinicalDay {
  date: string; S: number | null; D?: number | null; S_anchored?: number | null; S_rolling?: number | null
  D_anchored?: number | null; D_rolling?: number | null; level?: number; driver?: string | null
  branches?: Record<string, number | null> | Record<string, Record<string, number | null>>
  coverage?: Record<string, number | boolean>; entries?: number; entry_count?: number
  tracks?: Record<string, Record<string, number | null>>; origins?: string[]
}
export interface ClinicalAction { id?: number; date: string; arm_key?: string; label?: string; completed: boolean | null; rating?: number | null; reward?: number | null }
export interface ClinicalWorkspace {
  generated_at?: string; patient: ClinicalPatient; current?: CurrentState
  baseline: { active: (BaselineOut & { id?: number; note?: string | null; origin?: string }) | null; building: (BaselineOut & { id?: number; note?: string | null; origin?: string }) | null; stats: { slot: string; feature: string; center: number; scale: number; n: number; branch: string }[]; origin?: string }
  series: ClinicalDay[]; coverage?: Record<string, unknown>; sources: ClinicalSource[]
  explanation: Explanation | null; actions: ClinicalAction[]; models: ModelRecord[]
  origins?: Record<string, number>; limitations?: string[]
  analysis?: EntryAnalysis | null
}
export interface ClinicalEntry {
  id: number; patient_id?: number; patient_name?: string; ts: string; slot: string; kind: 'voice' | 'text' | 'wearable' | string
  data_origin: ClinicalOrigin; transcript_preview?: string | null; crisis: boolean
  analysis_available: boolean; analysis_provenance?: string | null; feature_count: number; scored_sources: string[]
}
export interface EntryPage { total: number; limit: number; offset: number; entries: ClinicalEntry[] }
export interface EntryFeature { name: string; value: number; source: string; branch: string; center?: number | null; scale?: number | null; z?: number | null; weight?: number | null; share?: number | null; used?: boolean; usage_known?: boolean; unit?: string; contribution?: number | null }
export interface EntryAnalysis {
  provenance: string; baseline_version?: number | null; baseline_status?: string | null
  tracks?: Record<string, { D?: Record<string, number>; Z?: Record<string, Record<string, number>>; shares?: Record<string, Record<string, number>>; stats?: Record<string, Record<string, {center: number; scale: number; n: number}>> }>
  before?: Partial<CurrentState> | null; after?: Partial<CurrentState> | null
  day_weights?: Record<string, unknown> | unknown[]; phrase?: Record<string, unknown> | null
  model_outputs?: Record<string, unknown>; explanation?: Explanation | null
  [key: string]: unknown
}
export interface EntryDetail {
  entry: ClinicalEntry; transcript: string | null; features: EntryFeature[]; analysis: EntryAnalysis | null
  limitations?: string[]; word_evidence?: { word: string; contribution?: number; direction?: string }[] | Record<string, unknown> | null
  raw_outputs?: { text: Record<string, unknown> | null; acoustic: Record<string, number>; health: Record<string, number> }
  data_source?: { source: string; measurement_date: string | null; received_at: string; synthetic: boolean }
}

async function get<T>(url: string, method = 'GET'): Promise<T> {
  const response = await fetch(`/api${url}`, { method, cache: 'no-store' })
  if (!response.ok) {
    const body = await response.json().catch(() => null)
    throw new Error(typeof body?.detail === 'string' ? body.detail : `The clinical workspace could not load (${response.status}). Please try again.`)
  }
  return response.json() as Promise<T>
}
export const clinical = {
  overview: () => get<ClinicalOverview>('/clinician/overview'),
  workspace: (pid: number, days: number) => get<ClinicalWorkspace>(`/clinician/patients/${pid}/workspace?days=${days}`),
  entries: (pid: number, limit = 25, offset = 0) => get<EntryPage>(`/clinician/patients/${pid}/entries?limit=${limit}&offset=${offset}`),
  entry: (id: number) => get<EntryDetail>(`/clinician/entries/${id}`),
  demo: () => get<{ patient_id: number; created: boolean; origin: string; disclosure: string }>('/clinician/demo', 'POST'),
}

/** One non-overlapping refresh every five seconds, while this tab is visible. */
export function useClinicalResource<T>(fetcher: () => Promise<T>, auto = true) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(true)
  const [refreshing, setRefreshing] = useState(false)
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null)
  const epoch = useRef(0)
  const inFlight = useRef(false)
  const refresh = useCallback(async () => {
    if (inFlight.current) return
    const generation = epoch.current
    inFlight.current = true; setRefreshing(true)
    try {
      const value = await fetcher()
      if (generation === epoch.current) { setData(value); setError(''); setUpdatedAt(new Date()) }
    } catch (e) { if (generation === epoch.current) setError(e instanceof Error ? e.message : 'Could not load this view.') }
    finally { if (generation === epoch.current) { inFlight.current = false; setLoading(false); setRefreshing(false) } }
  }, [fetcher])
  useEffect(() => {
    epoch.current += 1; inFlight.current = false; setData(null); setLoading(true)
    void refresh()
    const visibleRefresh = () => { if (document.visibilityState === 'visible') void refresh() }
    const timer = auto ? window.setInterval(visibleRefresh, 5000) : undefined
    window.addEventListener('focus', visibleRefresh); document.addEventListener('visibilitychange', visibleRefresh)
    window.addEventListener('steady:clinical', visibleRefresh)
    return () => { epoch.current += 1; inFlight.current = false; window.clearInterval(timer); window.removeEventListener('focus', visibleRefresh); document.removeEventListener('visibilitychange', visibleRefresh); window.removeEventListener('steady:clinical', visibleRefresh) }
  }, [refresh, auto])
  return { data, error, loading, refreshing, updatedAt, refresh }
}

export const isNumber = (value: unknown): value is number => typeof value === 'number' && Number.isFinite(value)
export const shortDay = (date: string) => new Date(`${date.slice(0, 10)}T12:00:00`).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })
export const number = (value: unknown, digits = 2) => isNumber(value) ? value.toLocaleString('en-GB', { minimumFractionDigits: digits, maximumFractionDigits: digits }) : '—'
export function timestamp(iso: string | null | undefined, full = false) {
  if (!iso) return 'No entry yet'
  const date = new Date(/Z$|[+-]\d\d:\d\d$/.test(iso) ? iso : `${iso}Z`)
  if (Number.isNaN(date.getTime())) return 'Time unavailable'
  return date.toLocaleString('en-GB', full ? { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit', second: '2-digit' } : { day: 'numeric', month: 'short', hour: '2-digit', minute: '2-digit' })
}
export function relativeTime(iso: string | null | undefined) {
  if (!iso) return 'No entries'
  const date = new Date(/Z$|[+-]\d\d:\d\d$/.test(iso) ? iso : `${iso}Z`)
  const delta = Math.max(0, Math.floor((Date.now() - date.getTime()) / 60000))
  if (!Number.isFinite(delta)) return 'Time unavailable'
  return delta < 1 ? 'Just now' : delta < 60 ? `${delta}m ago` : delta < 1440 ? `${Math.floor(delta / 60)}h ago` : `${Math.floor(delta / 1440)}d ago`
}
export function patientStatus(patient: ClinicalPatient) {
  const level = patient.current?.level ?? patient.level ?? 0
  if (level >= 3) return { key: 'crisis', label: 'Crisis flag', tone: 'alert' }
  if (level >= 2) return { key: 'review', label: 'Care-team review', tone: 'alert' }
  if (level >= 1) return { key: 'watch', label: 'Watch', tone: 'watch' }
  if (patient.baseline?.status !== 'active' || !isNumber(patient.current?.S)) return { key: 'building', label: 'Building baseline', tone: 'blue' }
  if (patient.current?.elevated) return { key: 'elevated', label: 'Elevated pattern', tone: 'blue' }
  return { key: 'usual', label: 'Within usual', tone: 'sage' }
}
export const BRANCHES = [{ key: 'acoustic', label: 'Voice', icon: 'graphic_eq', color: '#7c8f75' }, { key: 'text', label: 'Language', icon: 'text_fields', color: '#5d819c' }, { key: 'health', label: 'Wearable', icon: 'monitor_heart', color: '#b39464' }]
export const featureLabel = (name: string) => ({ text_logit: 'Language log-odds', first_person: 'First-person language', absolutist: 'Absolutist language', sentiment: 'Sentiment', speech_rate: 'Speech rate', f0_std: 'Pitch variation', pause_ratio: 'Pause ratio', energy: 'Vocal energy', jitter: 'Jitter', shimmer: 'Shimmer', arousal: 'Arousal', valence: 'Valence', steps: 'Daily steps', sleep_hours: 'Sleep duration', resting_hr: 'Resting heart rate', hrv_rmssd: 'HRV · RMSSD', active_minutes: 'Active minutes' }[name] || name.replace(/_/g, ' '))
