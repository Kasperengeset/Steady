import { useId, useState } from 'react'
import { isNumber, number } from '../lib/clinical'

export interface ChartSeries { key: string; label: string; color: string; values: (number | null | undefined)[]; dashed?: boolean }
export default function ClinicalChart({ dates, series, selected, onSelect, label = 'Distance from personal baseline', thresholds = false, height = 250 }: { dates: string[]; series: ChartSeries[]; selected?: number | null; onSelect?: (index: number) => void; label?: string; thresholds?: boolean; height?: number }) {
  const id = useId()
  const [hidden, setHidden] = useState<string[]>([])
  const shown = series.filter((item) => !hidden.includes(item.key))
  const values = shown.flatMap((item) => item.values.filter(isNumber))
  const [hovered, setHovered] = useState<number | null>(null)
  if (!dates.length || !series.some((item) => item.values.some(isNumber))) return <div className="cl-chart-empty"><span className="icon" aria-hidden>show_chart</span><p>No scored observations in this period</p><small>Raw model measurements remain available in the entry inspector.</small></div>
  const W = 760, H = height, left = 45, right = 20, top = 24, bottom = 32
  const low = Math.min(0, ...values, ...(thresholds ? [-1] : [])), high = Math.max(0, ...values, ...(thresholds ? [2] : []))
  const span = high - low || 2, lo = low - span * .14, hi = high + span * .18
  const x = (index: number) => left + index / Math.max(1, dates.length - 1) * (W - left - right)
  const y = (value: number) => H - bottom - (value - lo) / (hi - lo) * (H - top - bottom)
  const tick = Array.from({ length: 5 }, (_, index) => lo + (hi - lo) * index / 4)
  const picked = hovered ?? selected
  const titleDate = picked !== null && picked !== undefined ? dates[picked] : null
  const formatDate = (date: string) => new Date(`${date.slice(0, 10)}T12:00:00`).toLocaleDateString('en-GB', { day: 'numeric', month: 'short' })
  const path = (points: ChartSeries['values']) => { let drawing = false; return points.map((value, i) => { if (!isNumber(value)) { drawing = false; return '' } const segment = `${drawing ? 'L' : 'M'}${x(i)},${y(value)}`; drawing = true; return segment }).join(' ') }
  return <div className="cl-chart">
    <div className="cl-chart-legend">{series.map((item) => <button key={item.key} type="button" aria-pressed={!hidden.includes(item.key)} onClick={() => setHidden((old) => old.includes(item.key) ? old.filter((key) => key !== item.key) : [...old, item.key])} className={hidden.includes(item.key) ? 'is-hidden' : ''}><span style={{ background: item.color }} />{item.label}</button>)}<small>{label}</small></div>
    <svg viewBox={`0 0 ${W} ${H}`} role="img" aria-labelledby={id} className="cl-chart-svg" onMouseLeave={() => setHovered(null)}>
      <title id={id}>{label}. {dates.length} days. Missing observations are shown as gaps.</title>
      {thresholds && <rect x={left} y={y(1)} width={W - left - right} height={y(-1) - y(1)} fill="#edf2e9" />}
      {tick.map((value) => <g key={value}><line x1={left} x2={W - right} y1={y(value)} y2={y(value)} stroke="#e6ebe6" strokeDasharray="3 5" /><text x={left - 10} y={y(value) + 4} textAnchor="end" fill="#63745f" fontSize="14">{number(value, 1)}</text></g>)}
      <line x1={left} x2={W - right} y1={y(0)} y2={y(0)} stroke="#becbc0" />
      {thresholds && [1, 2].map((value) => <g key={value}><line x1={left} x2={W - right} y1={y(value)} y2={y(value)} stroke={value === 2 ? '#ba8066' : '#ad996c'} strokeDasharray="5 5" opacity=".65" /><text x={W - right} y={y(value) - 5} textAnchor="end" fill="#9a7a5f" fontSize="12">{value === 2 ? 'Review reference · 2' : 'Watch reference · 1'}</text></g>)}
      {shown.map((item) => <path key={item.key} d={path(item.values)} stroke={item.color} fill="none" strokeWidth="2.6" strokeDasharray={item.dashed ? '6 5' : undefined} strokeLinecap="round" strokeLinejoin="round" />)}
      {picked !== null && picked !== undefined && <line x1={x(picked)} x2={x(picked)} y1={top} y2={H - bottom} stroke="#6a7b70" strokeDasharray="3 4" />}
      {shown.map((item) => item.values.map((value, index) => isNumber(value) ? <circle key={`${item.key}-${index}`} cx={x(index)} cy={y(value)} r={picked === index ? 4.5 : 2} fill={picked === index ? '#fff' : item.color} stroke={item.color} strokeWidth="2" /> : null))}
      {dates.map((date, index) => <rect key={`${date}-${index}`} x={x(index) - Math.max(8, (W - left - right) / dates.length / 2)} y={top} width={Math.max(16, (W - left - right) / dates.length)} height={H - top - bottom} fill="transparent" onMouseEnter={() => setHovered(index)} onClick={() => onSelect?.(index)} />)}
      {[0, Math.floor((dates.length - 1) / 2), dates.length - 1].filter((index, i, all) => all.indexOf(index) === i).map((index) => <text key={index} x={x(index)} y={H - 9} textAnchor={index === 0 ? 'start' : index === dates.length - 1 ? 'end' : 'middle'} fill="#63745f" fontSize="13">{formatDate(dates[index])}</text>)}
    </svg>
    <div className="cl-chart-readout"><span>{titleDate ? formatDate(titleDate) : 'Select a day to inspect'}</span>{shown.map((item) => <span key={item.key}><i style={{ background: item.color }} />{item.label} <strong>{picked === null || picked === undefined ? '—' : number(item.values[picked])}</strong></span>)}</div>
    {onSelect && <label className="cl-chart-keyboard"><span>Inspect day with arrow keys</span><input type="range" min={0} max={Math.max(0, dates.length - 1)} value={selected ?? dates.length - 1} onChange={(event) => onSelect(Number(event.target.value))} aria-label="Selected observation day" /></label>}
  </div>
}
