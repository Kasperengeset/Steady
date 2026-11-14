import { useCallback, useEffect, useRef, useState } from 'react'
import { api } from '../lib/api'
import { clinical, useClinicalResource } from '../lib/clinical'
import { Icon } from '../ui/kit'
import { Badge, DemoButton, Panel } from './ClinicalUI'

const EXAMPLES = {
  steady: 'I got up at my usual time and made breakfast. I went for a short walk before lunch, and later I spent some time drawing. I was tired in the afternoon but enjoyed talking with a friend. Nothing special happened today. It felt good to have a little routine again.',
  difficult: 'I woke up tired and found it hard to get out of bed. I had planned to go outside but stayed home instead. Nothing felt worth doing, even the things I normally enjoy. I kept putting off messages from friends and felt alone for most of the day. I hope tomorrow feels a little easier.',
}
export default function LiveAnalysis({ patientId, demoKey, onSaved }: { patientId: number; demoKey?: string | null; onSaved: (id: number, patientId: number) => void }) {
  const cohort = useClinicalResource(useCallback(() => clinical.overview(), []), false)
  const [target, setTarget] = useState<number | null>(demoKey ? patientId : null)
  const [mode, setMode] = useState<'text' | 'voice'>('text'), [text, setText] = useState(''), [file, setFile] = useState<File | null>(null)
  const [slot, setSlot] = useState<'morning' | 'evening'>(new Date().getHours() < 14 ? 'morning' : 'evening')
  const [busy, setBusy] = useState(false), [error, setError] = useState('')
  const draftId = useRef<string | null>(null), inFlight = useRef(false)
  const fileInput = useRef<HTMLInputElement | null>(null)
  const profiles = (cohort.data?.patients || []).filter((patient) => patient.demo_key)
  const selected = profiles.find((patient) => patient.id === target)
  const words = text.trim() ? text.trim().split(/\s+/).length : 0
  useEffect(() => { setTarget(demoKey ? patientId : null); draftId.current = null }, [patientId, demoKey])
  const save = async () => {
    if (inFlight.current || !target || !selected?.demo_key) return
    if (mode === 'voice' && (!file || file.size > 15 * 1024 * 1024)) { setError('Choose an audio file under 15 MB.'); return }
    inFlight.current = true; setBusy(true); setError('')
    try {
      draftId.current ??= globalThis.crypto?.randomUUID?.() || `clinical-${Date.now()}-${Math.random().toString(36).slice(2)}`
      const result = mode === 'text' ? await api.checkInText(target, slot, text.trim(), draftId.current) : await api.checkIn(target, slot, file!, undefined, draftId.current)
      draftId.current = null; setText(''); setFile(null); if (fileInput.current) fileInput.current.value = ''
      window.dispatchEvent(new Event('steady:clinical')); window.dispatchEvent(new Event('steady:engagement'))
      onSaved(result.checkin_id, target)
    } catch (e) { setError(e instanceof Error ? e.message : 'Analysis could not finish. Your draft is still here; try again.') }
    finally { inFlight.current = false; setBusy(false) }
  }
  return <Panel title="Put a real entry through the models" eyebrow="LIVE INFERENCE LAB" className="cl-live-lab" aside={<Badge tone="blue">Saves to demo profiles only</Badge>}>
    <p className="cl-lab-intro">Run the actual voice or language pipeline. The entry is saved to the selected demo care record, then opens with raw outputs, baseline-relative features and state before/after.</p>
    <div className="cl-lab-target"><label><span>Demo care record</span><select value={target ?? ''} disabled={busy} onChange={(event) => { setTarget(event.target.value ? Number(event.target.value) : null); draftId.current = null }}><option value="">Choose an explicit demo profile</option>{profiles.map((patient) => <option key={patient.id} value={patient.id}>{patient.name} · Demo #{patient.id}</option>)}</select></label><label><span>Check-in slot</span><select value={slot} disabled={busy} onChange={(event) => { setSlot(event.target.value as 'morning' | 'evening'); draftId.current = null }}><option value="morning">Morning</option><option value="evening">Evening</option></select></label></div>
    {!profiles.length && <div className="cl-lab-empty"><p>No demo profile yet. Create an isolated workspace with clearly labelled synthetic baseline history.</p><DemoButton /></div>}
    <div className="cl-tabs cl-tabs-small" role="tablist" aria-label="Live analysis input"><button role="tab" aria-selected={mode === 'text'} disabled={busy} onClick={() => { setMode('text'); draftId.current = null }}><Icon name="edit_note" />Written diary</button><button role="tab" aria-selected={mode === 'voice'} disabled={busy} onClick={() => { setMode('voice'); draftId.current = null }}><Icon name="graphic_eq" />Voice recording</button></div>
    {mode === 'text' ? <><label className="cl-lab-text"><span className="sr-only">Diary text to analyze</span><textarea value={text} onChange={(event) => { setText(event.target.value); draftId.current = null }} maxLength={10000} rows={5} disabled={busy} placeholder="Type a diary entry here. Use your own words, or load an example below. Nothing is sent until Save & analyze." /></label><div className="cl-lab-examples"><span>{words} words · English text model</span><button disabled={busy} onClick={() => { setText(EXAMPLES.steady); draftId.current = null }}>Load a steadier-day example</button><button disabled={busy} onClick={() => { setText(EXAMPLES.difficult); draftId.current = null }}>Load a harder-day example</button></div></> : <label className="cl-upload"><Icon name="upload_file" /><strong>{file ? file.name : 'Choose a diary recording'}</strong><span>Actual acoustic inference · audio file, under 15 MB</span><input ref={fileInput} type="file" accept="audio/*,.wav,.m4a,.mp3,.webm" disabled={busy} onChange={(event) => { setFile(event.target.files?.[0] || null); draftId.current = null }} /></label>}
    <div className="cl-lab-footer"><p><Icon name="info" />Synthetic baseline history is a demonstration reference. New entries use the real model pipeline.</p><button className="cl-button cl-button-primary" disabled={busy || !selected?.demo_key || mode === 'text' && words < 3 || mode === 'voice' && !file} onClick={save}><Icon name={busy ? 'hourglass_top' : 'play_arrow'} />{busy ? 'Extracting measurements…' : 'Save & analyze entry'}</button></div>
    {error && <p role="alert" className="cl-inline-error">{error}</p>}{cohort.error && <p role="alert" className="cl-inline-error">Could not load demo profiles. <button onClick={cohort.refresh}>Retry</button></p>}
  </Panel>
}
