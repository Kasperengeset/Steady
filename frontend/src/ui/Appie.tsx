import type { CompanionView } from '../lib/companion'
import { Icon } from './kit'

export function Appie({ mood = 'calm', tier = 1, name = 'Appie', className = '' }: { mood?: string; tier?: number; name?: string; className?: string }) {
  const pose = ['calm', 'happy', 'celebrate', 'sleep', 'wave', 'care', 'eat'].includes(mood) ? mood : 'calm'
  const file = tier > 1 && ['calm', 'happy'].includes(pose) ? `${pose}_t${Math.min(3, tier)}` : pose
  const detail = file.endsWith('_t2') ? ' wearing headphones' : file.endsWith('_t3') ? ' with a backpack and star' : ', your ring-tailed lemur companion'
  return <img src={`/appie/${file}.webp`} alt={`${name}${detail}`} className={`appie-art object-contain ${className}`} />
}

export function Food({ amount, className = '' }: { amount: number; className?: string }) {
  return <span className={`inline-flex items-center gap-1.5 ${className}`}><span aria-hidden>🥭</span><span>{amount}<span className="sr-only"> pet food</span></span></span>
}

export function LevelProgress({ data }: { data: CompanionView }) {
  const progress = data.xp_next ? Math.min(100, Math.max(0, (data.xp - data.xp_level_start) / (data.xp_next - data.xp_level_start) * 100)) : 100
  return <div>
    <div className="mb-2 flex items-center justify-between gap-2 text-xs font-medium text-ink-soft"><span>Level {data.level} · {data.title}</span><span>{data.xp_next ? `${data.xp} / ${data.xp_next} XP` : `${data.xp} XP · max level`}</span></div>
    <div role="progressbar" aria-label="Progress to next level" aria-valuenow={Math.round(progress)} aria-valuemin={0} aria-valuemax={100} className="h-2.5 overflow-hidden rounded-full bg-primary/10"><div className="h-full rounded-full bg-primary-container transition-[width] duration-700" style={{ width: `${progress}%` }} /></div>
  </div>
}

export function Week({ data }: { data: CompanionView }) {
  return <div className="grid grid-cols-7 gap-1.5" aria-label="Your check-ins this week">
    {data.week.map((d, index) => <div key={d.date} className="flex min-w-0 flex-col items-center gap-2" role="img" aria-label={`${d.date}: ${d.checked_in ? 'checked in' : 'no check-in'}`}>
      <span className="text-[11px] font-medium text-ink-soft">{new Date(`${d.date}T12:00:00`).toLocaleDateString('en-GB', { weekday: 'short' }).slice(0, 1)}</span>
      <span className={`grid aspect-square w-full max-w-9 place-items-center rounded-full ${d.checked_in ? 'bg-primary text-white' : index === 6 ? 'border border-dashed border-primary bg-white/50 text-primary' : 'bg-primary/7 text-outline'}`}><Icon name={d.checked_in ? 'check' : index === 6 ? 'more_horiz' : 'remove'} className="text-base" /></span>
      {index === 6 && <span className="-mt-1 h-1 w-1 rounded-full bg-primary" />}
    </div>)}
  </div>
}

export function LoadError({ error, retry }: { error: string; retry: () => void }) {
  return <div role="alert" className="my-6 rounded-3xl bg-sand p-5"><p className="font-semibold">A little connection hiccup</p><p className="mt-2 text-sm text-ink-soft">{error}</p><button onClick={retry} className="mt-4 rounded-full bg-primary px-5 py-2.5 text-sm font-semibold text-white">Try again</button></div>
}
