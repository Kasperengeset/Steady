import { useCallback, useState } from 'react'
import { Link } from 'react-router-dom'
import { views } from '../lib/views'
import { timestamp, useClinicalResource } from '../lib/clinical'
import { Icon } from '../ui/kit'
import { Badge, EmptyState, ErrorState, LiveStatus, LoadingState, Panel } from './ClinicalUI'

const TABS = [{ key: 'open', label: 'Needs action' }, { key: 'all', label: 'All notifications' }, { key: 'handled', label: 'Handled' }] as const
const KINDS = {
  crisis: { label: 'Crisis flag', tone: 'alert', icon: 'priority_high' },
  clinic: { label: 'Care-team review', tone: 'alert', icon: 'forum' },
  friends: { label: 'Watch', tone: 'watch', icon: 'visibility' },
  baseline: { label: 'Baseline review', tone: 'blue', icon: 'adjust' },
}

export default function Notifications() {
  const [tab, setTab] = useState<'open' | 'all' | 'handled'>('open')
  const resource = useClinicalResource(useCallback(() => views.notifications(tab), [tab]))
  const [pending, setPending] = useState<string | null>(null), [actionError, setActionError] = useState('')
  const handle = async (id: string) => {
    if (pending) return
    setPending(id); setActionError('')
    try { await views.markHandled(id); await resource.refresh(); window.dispatchEvent(new Event('steady:clinical')) }
    catch (e) { setActionError(e instanceof Error ? e.message : 'The notification could not be marked as handled. Try again.') }
    finally { setPending(null) }
  }
  if (!resource.data) return resource.error ? <ErrorState message={resource.error} retry={resource.refresh} /> : <LoadingState label="Loading the care inbox" />
  return <div className="cl-page">
    <div className="cl-page-title"><div><p className="cl-eyebrow">FOLLOW UP, WITH THE FULL PICTURE</p><h1>Care inbox</h1><p>Review new signals, open the underlying observations and keep track of follow-up.</p></div><LiveStatus {...resource} /></div>
    {resource.error && <ErrorState message={resource.error} retry={resource.refresh} />}
    {actionError && <p role="alert" className="cl-inline-error">{actionError}</p>}
    <div className="cl-tabs" role="tablist" aria-label="Notification status">{TABS.map((item) => <button key={item.key} role="tab" aria-selected={tab === item.key} onClick={() => setTab(item.key)}>{item.label}</button>)}</div>
    <Panel title={TABS.find((item) => item.key === tab)?.label} eyebrow="NEWEST FIRST · CRISIS FLAGS PRIORITISED" aside={<Badge>{resource.data.length} notifications</Badge>}>
      {!resource.data.length ? <EmptyState icon="done_all" title="No notifications in this view">Review the cohort to check observation coverage and baseline readiness.</EmptyState> : <div className="cl-notification-list">{resource.data.map((item) => {
        const kind = KINDS[item.kind]
        return <article className={`cl-notification ${item.kind === 'crisis' && !item.handled ? 'is-crisis' : ''}`} key={item.id}>
          <span className={`cl-notification-icon cl-badge-${kind.tone}`}><Icon name={kind.icon} /></span>
          <div className="cl-notification-body"><div><h3>{item.patient}</h3><Badge tone={kind.tone}>{kind.label}</Badge>{item.handled && <Badge tone="sage">Handled</Badge>}<time>{timestamp(item.ts, true)}</time></div><p>{item.message}</p>{item.quote && <blockquote>{item.quote}</blockquote>}</div>
          <div className="cl-notification-actions"><Link to={`/clinician/patients/${item.patient_id}`} className="cl-button cl-button-primary">Open care record <Icon name="arrow_forward" /></Link>{!item.handled && item.kind !== 'baseline' && <button className="cl-button" disabled={pending !== null} onClick={() => handle(item.id)}>{pending === item.id ? 'Saving…' : 'Mark as handled'}</button>}</div>
        </article>
      })}</div>}
    </Panel>
  </div>
}
