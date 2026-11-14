import { useCallback, useState } from 'react'
import { clinical, number, BRANCHES, featureLabel, useClinicalResource, type ModelRecord } from '../lib/clinical'
import { Icon } from '../ui/kit'
import { Badge, EmptyState, ErrorState, LiveStatus, LoadingState, Panel } from './ClinicalUI'

function validationMetrics(model: ModelRecord): [string, number][] {
  const raw = model.validation?.metrics
  return raw && typeof raw === 'object' ? Object.entries(raw).filter((item): item is [string, number] => typeof item[1] === 'number' && Number.isFinite(item[1])) : []
}
const metricLabel = (key: string) => ({ roc_auc: 'ROC-AUC', auc: 'ROC-AUC', f1: 'F1', mcc: 'MCC', balanced_accuracy: 'Balanced accuracy', accuracy: 'Accuracy', mean_auc: 'ROC-AUC · mean' }[key] || key.replace(/_/g, ' '))
const metadata = (value: unknown) => typeof value === 'number' ? value.toLocaleString('en-GB') : typeof value === 'string' ? value : null

export function ModelCards({ models, compact = false }: { models: ModelRecord[]; compact?: boolean }) {
  return <div className={`cl-model-cards ${compact ? 'is-compact' : ''}`}>{models.map((model) => { const branch = BRANCHES.find((item) => item.key === model.key), metrics = validationMetrics(model); return <article key={model.key} className="cl-model-card"><div className="cl-model-card-top"><span className="cl-model-icon" style={{ color: branch?.color }}><Icon name={branch?.icon || 'hub'} /></span><Badge tone={model.status === 'missing' || model.status === 'unavailable' ? 'watch' : 'sage'}>{model.status.replace(/_/g, ' ')}</Badge></div><p className="cl-eyebrow">{branch?.label || model.key} branch</p><h3>{model.label}</h3><div className="cl-model-primary">{metrics.length ? <><strong>{number(metrics[0][1], 3)}</strong><span>{metricLabel(metrics[0][0])}<small>Held-out dataset result</small></span></> : <span className="cl-muted">Validation metric unavailable</span>}</div>{!compact && <p>{model.description || 'Raw measurements are compared with this person’s baseline before they contribute to drift.'}</p>}<div className="cl-model-foot"><span>{model.features?.length ?? '—'} runtime features</span><code>{model.artifact_sha256 ? model.artifact_sha256.slice(0, 10) : 'No artifact hash'}</code></div></article> })}</div>
}

export function ModelEvidence({ models }: { models: ModelRecord[] }) {
  const [selected, setSelected] = useState<string | null>(null)
  const current = models.find((model) => model.key === selected) || models[0]
  if (!current) return <EmptyState icon="hub" title="No model evidence is available">Model metadata will appear here when the backend exposes its loaded artifacts.</EmptyState>
  const metrics = validationMetrics(current)
  const evaluationSample = [metadata(current.validation?.sample_size), metadata(current.validation?.sample_unit)].filter(Boolean).join(' ') || 'Not recorded'
  return <div className="cl-stack"><ModelCards models={models} /><div className="cl-tabs" role="tablist" aria-label="Model branch">{models.map((model) => <button key={model.key} role="tab" aria-selected={current.key === model.key} onClick={() => setSelected(model.key)}>{BRANCHES.find((branch) => branch.key === model.key)?.label || model.key}<Icon name="arrow_outward" /></button>)}</div>
    <div className="cl-two-column"><Panel title={current.label} eyebrow="LOADED MODEL · HELD-OUT EVIDENCE" aside={<Badge tone="blue">{current.key}</Badge>}><div className="cl-validation-metrics">{metrics.map(([key, value]) => <div key={key}><strong>{number(value, 3)}</strong><span>{metricLabel(key)}</span></div>)}</div>{metrics.length === 0 && <p className="cl-muted">No validation metrics were supplied for this artifact.</p>}<dl className="cl-metadata"><div><dt>Evaluation sample</dt><dd>{evaluationSample}</dd></div><div><dt>Validation split</dt><dd>{metadata(current.validation?.split) || 'Not recorded'}</dd></div><div><dt>Report</dt><dd className="cl-mono">{metadata(current.validation?.report) || 'Not available'}</dd></div><div><dt>Artifact</dt><dd className="cl-mono">{current.artifact_path || 'Not recorded'}</dd></div><div><dt>SHA-256</dt><dd className="cl-hash">{current.artifact_sha256 || 'Not recorded'}</dd></div></dl><p className="cl-evidence-note"><Icon name="info" />These are population dataset results. They are not a confidence score for an individual patient or a validation of relapse detection.</p></Panel>
    <Panel title="What the engine actually uses" eyebrow="RUNTIME FEATURES & DIRECTIONS"><div className="cl-table-scroll"><table className="cl-table cl-table-dense"><thead><tr><th>Measurement</th><th>Weight</th><th>Drift direction</th></tr></thead><tbody>{(current.features || []).map((feature) => <tr key={feature.name}><td><strong>{featureLabel(feature.name)}</strong><code>{feature.name}</code></td><td className="cl-numeric">{number(feature.weight, 4)}</td><td>{feature.weight === 0 ? <Badge>Context only</Badge> : <span>{feature.direction < 0 ? 'Lower' : 'Higher'} than baseline</span>}</td></tr>)}</tbody></table></div>{!current.features?.length && <p className="cl-muted">No runtime feature manifest was supplied.</p>}</Panel></div>
    <Panel title="Interpretation and limits" eyebrow="KEEP THE DATA IN CONTEXT"><ul className="cl-limit-list">{(current.limitations || []).map((limitation) => <li key={limitation}><Icon name="info" /><p>{limitation}</p></li>)}</ul>{!current.limitations?.length && <p className="cl-muted">No model-specific limitations were supplied.</p>}</Panel>
  </div>
}

export default function ModelRegistry() {
  const resource = useClinicalResource(useCallback(() => clinical.overview(), []))
  if (!resource.data) return resource.error ? <ErrorState message={resource.error} retry={resource.refresh} /> : <LoadingState label="Loading the model registry" />
  return <div className="cl-page"><div className="cl-page-title"><div><p className="cl-eyebrow">MEASUREMENTS, WITH A TRACEABLE SOURCE</p><h1>Model evidence</h1><p>Loaded artifacts, held-out results and the exact measurements used in the app.</p></div><LiveStatus {...resource} /></div>{resource.error && <ErrorState message={resource.error} retry={resource.refresh} />}<ModelEvidence models={resource.data.models} /></div>
}
