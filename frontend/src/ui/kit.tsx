// Small shared UI pieces for the Steady screens: icons, a calm SVG line chart, formatting.
import type { ReactNode } from 'react'
import type { SignalFormat } from '../lib/views'

export function Icon({ name, fill = false, className = '' }: { name: string; fill?: boolean; className?: string }) {
  return <span aria-hidden className={`icon ${fill ? 'icon-fill' : ''} ${className}`}>{name}</span>
}

export function Card({ children, className = '' }: { children: ReactNode; className?: string }) {
  return <section className={`rounded-[1.75rem] bg-surface-low p-5 ${className}`}>{children}</section>
}

export interface Series { values: (number | null)[]; color: string; width?: number; dashed?: boolean }

/** Line chart with optional soft band (e.g. "your usual") and dashed reference lines. */
export function LineChart({
  series, band, refs = [], height = 160, yMin, yMax, smooth = true, onPick, picked,
}: {
  series: Series[]
  band?: [number, number]
  refs?: { y: number; label: string }[]
  height?: number
  yMin?: number
  yMax?: number
  smooth?: boolean
  onPick?: (i: number) => void
  picked?: number | null
}) {
  const W = 600, H = height, P = 8
  const all = series.flatMap((s) => s.values.filter((v): v is number => v !== null))
  const lo = Math.min(yMin ?? Infinity, ...(band ?? []), ...refs.map((r) => r.y), ...all, 0)
  const hi = Math.max(yMax ?? -Infinity, ...(band ?? []), ...refs.map((r) => r.y), ...all, 0)
  const pad = (hi - lo) * 0.12 || 1
  const y0 = lo - pad, y1 = hi + pad
  const n = Math.max(...series.map((s) => s.values.length), 2)
  const x = (i: number) => P + (i / (n - 1)) * (W - 2 * P)
  const y = (v: number) => H - P - ((v - y0) / (y1 - y0)) * (H - 2 * P)

  const path = (vals: (number | null)[]) => {
    let d = '', prev: [number, number] | null = null
    vals.forEach((v, i) => {
      if (v === null) { prev = null; return }
      const p: [number, number] = [x(i), y(v)]
      if (!prev) d += `M${p[0]},${p[1]}`
      else if (smooth) { const mx = (prev[0] + p[0]) / 2; d += ` C${mx},${prev[1]} ${mx},${p[1]} ${p[0]},${p[1]}` }
      else d += ` L${p[0]},${p[1]}`
      prev = p
    })
    return d
  }

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height }} preserveAspectRatio="none" role="img">
      {band && <rect x={P} width={W - 2 * P} y={y(band[1])} height={y(band[0]) - y(band[1])} rx={14} fill="#dfeee3" opacity={0.75} />}
      {refs.map((r) => (
        <g key={r.label}>
          <line x1={P} x2={W - P} y1={y(r.y)} y2={y(r.y)} stroke="#b98a55" strokeDasharray="6 6" strokeWidth={1.2} opacity={0.8} />
          <text x={P + 6} y={y(r.y) - 6} textAnchor="start" fontSize={13} fill="#8a6a43">{r.label}</text>
        </g>
      ))}
      {series.map((s, k) => (
        <path key={k} d={path(s.values)} fill="none" stroke={s.color} strokeWidth={s.width ?? 3}
              strokeDasharray={s.dashed ? '7 7' : undefined} strokeLinecap="round" strokeLinejoin="round"
              vectorEffect="non-scaling-stroke" />
      ))}
      {series[0]?.values.map((v, i) => v === null ? null : (
        <circle key={i} cx={x(i)} cy={y(v)} r={picked === i ? 7 : onPick ? 4 : i === series[0].values.length - 1 ? 6 : 0}
                fill={picked === i ? '#43654f' : '#ffffff'} stroke="#43654f" strokeWidth={2.5} vectorEffect="non-scaling-stroke"
                className={onPick ? 'cursor-pointer' : ''} onClick={onPick ? () => onPick(i) : undefined} />
      ))}
    </svg>
  )
}

/** Tiny line for table rows and signal rows. */
export function Sparkline({ values, color = '#43654f', width = 96, height = 28 }: { values: (number | null)[]; color?: string; width?: number; height?: number }) {
  const v = values.filter((x): x is number => x !== null)
  if (v.length < 2) return <span className="inline-block" style={{ width, height }} />
  const lo = Math.min(...v), hi = Math.max(...v), span = hi - lo || 1
  const pts = values.map((x, i) => x === null ? null : `${(i / (values.length - 1)) * width},${height - 3 - ((x - lo) / span) * (height - 6)}`)
    .filter(Boolean).join(' ')
  return <svg width={width} height={height}><polyline points={pts} fill="none" stroke={color} strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" /></svg>
}

export function fmtSignal(v: number | undefined, f: SignalFormat): string {
  if (v === undefined || v === null || Number.isNaN(v)) return '-'
  switch (f) {
    case 'int': return Math.round(v).toLocaleString('en-GB')
    case 'pct': return `${Math.round(v * 100)}%`
    case 'clock': {
      const h = ((v % 24) + 24) % 24
      return `${String(Math.floor(h)).padStart(2, '0')}:${String(Math.round((h % 1) * 60) % 60).padStart(2, '0')}`
    }
    case 'h': return `${v.toFixed(1)} h`
    case 'bpm': return `${Math.round(v)} bpm`
    case 'ms': return `${Math.round(v)} ms`
    default: return Math.abs(v) >= 10 ? v.toFixed(0) : v.toFixed(2)
  }
}

export function timeAgo(iso: string): string {
  const t = new Date(iso.endsWith('Z') || iso.includes('+') ? iso : iso + 'Z').getTime()
  const m = Math.round((Date.now() - t) / 60000)
  if (m < 1) return 'just now'
  if (m < 60) return `${m} min ago`
  const h = Math.round(m / 60)
  if (h < 24) return `${h} h ago`
  const d = Math.round(h / 24)
  return d === 1 ? 'yesterday' : `${d} days ago`
}

export function shortDate(iso: string): string {
  return new Date(iso.slice(0, 10) + 'T12:00:00Z').toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'short' })
}

export const toneStyle: Record<string, string> = {
  Steady: 'bg-sage-soft text-primary',
  Lifted: 'bg-primary-fixed text-primary',
  Quieter: 'bg-ochre-soft text-[#7a5a32]',
}
