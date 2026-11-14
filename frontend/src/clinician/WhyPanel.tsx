// "Why is this changing?": which signals moved against the healthy baseline, grouped by source.
import { useEffect, useState } from 'react'
import { views, type Explain, type SignalRow, type SourceExplain } from '../lib/views'
import { Card, Icon, Sparkline, fmtSignal, shortDate } from '../ui/kit'

const SRC_ICON: Record<string, string> = { health: 'directions_walk', acoustic: 'graphic_eq', text: 'edit_note' }
const DIR = {
  worse: { label: 'worse', text: 'text-terracotta', bar: '#c98b72', chip: 'bg-terracotta-soft text-terracotta' },
  better: { label: 'better', text: 'text-primary', bar: '#7fa48a', chip: 'bg-sage-soft text-primary' },
  same: { label: 'same', text: 'text-ink-soft', bar: '#cfd3cd', chip: 'bg-surface-high text-ink-soft' },
} as const

export default function WhyPanel({ pid, days }: { pid: number; days: number }) {
  const [e, setE] = useState<Explain | null>(null)
  const [open, setOpen] = useState<Record<string, boolean>>({})
  useEffect(() => {
    setE(null)
    views.explain(pid, days).then((x) => {
      setE(x)
      const top = [...(x.sources ?? [])].sort((a, b) => b.share - a.share)[0]
      setOpen(top ? { [top.source]: true } : {})
    })
  }, [pid, days])

  if (!e) return <Card><p className="text-ink-soft">Working out what changed…</p></Card>
  if (!e.available) return <Card><h2 className="text-lg font-semibold">Why is this changing?</h2><p className="mt-2 text-ink-soft">{e.message}</p></Card>

  return (
    <Card className="bg-white p-6 shadow-sm">
      <h2 className="text-lg font-semibold">Why is this changing?</h2>
      <p className="mt-2 text-lg">{e.summary}</p>
      <p className="mt-1 text-sm text-ink-soft">{e.data_note} Compared with this patient's healthy baseline.</p>

      <p className="mt-6 text-xs font-medium uppercase tracking-wider text-ink-soft">What is driving it</p>
      <div className="mt-2 flex h-4 overflow-hidden rounded-full bg-surface-high">
        {e.sources.filter((s) => s.share > 0).map((s) => (
          <div key={s.source} title={`${s.label} ${Math.round(s.share * 100)}%`}
               style={{ width: `${s.share * 100}%`, background: DIR[s.direction ?? 'same'].bar }} className="border-r-2 border-white last:border-0" />
        ))}
      </div>
      <div className="mt-2 flex flex-wrap gap-x-6 gap-y-1 text-sm">
        {e.sources.map((s) => (
          <span key={s.source} className={DIR[s.direction ?? 'same'].text}>
            {s.label} {s.direction ? `${Math.round(s.share * 100)}%, ${DIR[s.direction].label}` : '(no data)'}
          </span>
        ))}
      </div>

      <div className="mt-6 flex flex-col gap-3">
        {[...e.sources].sort((a, b) => b.share - a.share).map((s) => (
          <SourceCard key={s.source} s={s} open={!!open[s.source]} toggle={() => setOpen({ ...open, [s.source]: !open[s.source] })}
                      words={s.source === 'text' ? e.words : undefined} />
        ))}
        <Habits h={e.habits} />
      </div>
    </Card>
  )
}

function SourceCard({ s, open, toggle, words }: { s: SourceExplain; open: boolean; toggle: () => void; words?: Explain['words'] }) {
  const dir = DIR[s.direction ?? 'same']
  return (
    <div className="rounded-3xl bg-surface-low">
      <button onClick={toggle} className="flex w-full items-center gap-3 p-4 text-left">
        <span className="grid h-10 w-10 place-items-center rounded-full bg-white text-primary"><Icon name={SRC_ICON[s.source]} /></span>
        <span className="flex-1">
          <span className="font-semibold">{s.label}</span>
          <span className="block text-sm text-ink-soft">{s.summary}</span>
        </span>
        {s.direction && <span className={`rounded-full px-3 py-1 text-xs font-semibold ${dir.chip}`}>{dir.label}</span>}
        <Icon name={open ? 'expand_less' : 'expand_more'} className="text-outline" />
      </button>
      {open && (
        <div className="px-4 pb-4">
          <table className="w-full text-sm">
            <thead className="text-xs text-ink-soft"><tr><th className="py-1 text-left font-medium">Signal</th><th className="text-right font-medium">Usual</th><th className="text-right font-medium">This period</th><th /><th /></tr></thead>
            <tbody>{s.signals.map((r) => <SignalLine key={r.feature} r={r} />)}</tbody>
          </table>
          {words && (words.stand_out.length > 0 || words.quote) && (
            <div className="mt-4 rounded-2xl bg-white p-4">
              {words.stand_out.length > 0 && (
                <>
                  <p className="text-xs font-medium uppercase tracking-wider text-ink-soft">Words that stood out</p>
                  <div className="mt-2 flex flex-wrap gap-2">
                    {words.stand_out.map((w) => (
                      <span key={w.word} className="rounded-full bg-ochre-soft px-3 py-1 text-sm text-[#7a5a32]">{w.word} <span className="text-xs opacity-70">{shortDate(w.date)}</span></span>
                    ))}
                  </div>
                </>
              )}
              {words.quote && (
                <blockquote className="mt-4 border-l-4 border-ochre pl-3 text-sm">
                  <Highlighted text={words.quote.text} words={words.quote.highlights} />
                  <span className="mt-1 block text-xs text-ink-soft">{shortDate(words.quote.date)}, the entry furthest from their usual language</span>
                </blockquote>
              )}
            </div>
          )}
          {s.source === 'acoustic' && <p className="mt-3 text-xs text-ink-soft">Recordings are deleted after analysis. These describe how they spoke, not what they said.</p>}
        </div>
      )}
    </div>
  )
}

function SignalLine({ r }: { r: SignalRow }) {
  if (r.status === 'no_data') return (
    <tr className="border-t border-surface-high text-outline"><td className="py-2" title={r.technical}>{r.name}</td><td colSpan={4} className="text-right">not enough data</td></tr>
  )
  const d = DIR[r.status]
  return (
    <tr className="border-t border-surface-high">
      <td className="py-2" title={r.technical}>{r.name}</td>
      <td className="text-right text-ink-soft">{fmtSignal(r.usual, r.format)}</td>
      <td className={`text-right font-semibold ${d.text}`}>{fmtSignal(r.now, r.format)}</td>
      <td className={`pl-3 text-xs font-semibold ${d.text}`}>{d.label}</td>
      <td className="pl-2 text-right"><Sparkline values={r.series ?? []} width={70} height={22} color={r.status === 'worse' ? '#b4674a' : '#5b7e67'} /></td>
    </tr>
  )
}

function Habits({ h }: { h: Explain['habits'] }) {
  const fewer = h.per_week_now < h.per_week_usual * 0.7
  const shorter = h.words_usual && h.words_now ? h.words_now < h.words_usual * 0.7 : false
  return (
    <div className="rounded-3xl bg-surface-low p-4">
      <div className="flex items-center gap-3">
        <span className="grid h-10 w-10 place-items-center rounded-full bg-white text-primary"><Icon name="event_available" /></span>
        <span className="flex-1"><span className="font-semibold">Check-in habits</span>
          <span className="block text-sm text-ink-soft">Shown for context, not part of the score.</span></span>
      </div>
      <div className="mt-3 grid grid-cols-3 gap-3 text-sm">
        <Stat label="Check-ins per week" usual={h.per_week_usual} now={h.per_week_now} warn={fewer} />
        <Stat label="Words per entry" usual={h.words_usual} now={h.words_now} warn={shorter} />
        <div className="rounded-2xl bg-white p-3"><p className="text-ink-soft">Days since last check-in</p>
          <p className={`mt-1 text-lg font-semibold ${h.days_since_last_checkin >= 3 ? 'text-terracotta' : ''}`}>{h.days_since_last_checkin}</p></div>
      </div>
    </div>
  )
}

function Stat({ label, usual, now, warn }: { label: string; usual: number | null; now: number | null; warn: boolean }) {
  return (
    <div className="rounded-2xl bg-white p-3">
      <p className="text-ink-soft">{label}</p>
      <p className={`mt-1 text-lg font-semibold ${warn ? 'text-terracotta' : ''}`}>{now ?? '-'} <span className="text-sm font-normal text-ink-soft">usual {usual ?? '-'}</span></p>
    </div>
  )
}

function Highlighted({ text, words }: { text: string; words: string[] }) {
  if (!words.length) return <>"{text}"</>
  const re = new RegExp(`\\b(${words.map((w) => w.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')).join('|')})\\b`, 'gi')
  const parts = text.split(re)
  return <>"{parts.map((p, i) => i % 2 ? <mark key={i} className="rounded bg-ochre-soft px-0.5 text-ink">{p}</mark> : p)}"</>
}
