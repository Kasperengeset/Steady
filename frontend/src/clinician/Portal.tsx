// Clinician portal frame (desktop): sidebar + page. Same calm style as the patient app, denser, numbers allowed.
import { useEffect, useState } from 'react'
import { NavLink, Outlet } from 'react-router-dom'
import { views } from '../lib/views'
import { Icon } from '../ui/kit'

export const KIND: Record<string, { label: string; dot: string; chip: string }> = {
  crisis: { label: 'Crisis flag', dot: 'bg-terracotta', chip: 'bg-terracotta-soft text-terracotta' },
  clinic: { label: 'Needs a call', dot: 'bg-terracotta', chip: 'bg-terracotta-soft text-terracotta' },
  friends: { label: 'Watch', dot: 'bg-ochre', chip: 'bg-ochre-soft text-[#7a5a32]' },
  baseline: { label: 'Baseline', dot: 'bg-[#7d9db0]', chip: 'bg-[#e3edf2] text-[#3e5b6c]' },
}

export const STATUS: Record<string, { label: string; chip: string }> = {
  building: { label: 'Building baseline', chip: 'bg-[#e3edf2] text-[#3e5b6c]' },
  close: { label: 'Steady', chip: 'bg-sage-soft text-primary' },
  different: { label: 'Watch', chip: 'bg-ochre-soft text-[#7a5a32]' },
  care: { label: 'Needs a call', chip: 'bg-terracotta-soft text-terracotta' },
}

export default function Portal() {
  const [open, setOpen] = useState(0)
  useEffect(() => { views.notifications('open').then((n) => setOpen(n.length)).catch(() => undefined) }, [])
  const link = ({ isActive }: { isActive: boolean }) =>
    `flex items-center gap-3 rounded-full px-4 py-3 ${isActive ? 'bg-sage-soft font-semibold text-primary' : 'text-ink-soft hover:bg-surface-mid'}`

  return (
    <div className="flex min-h-full bg-surface">
      <aside className="sticky top-0 flex h-screen w-64 shrink-0 flex-col gap-2 border-r border-surface-high bg-surface-low p-5">
        <div className="mb-6 flex items-center gap-3 px-2">
          <span className="grid h-10 w-10 place-items-center rounded-full bg-sage-soft text-primary"><Icon name="spa" /></span>
          <div><p className="font-semibold">Steady</p><p className="text-xs text-ink-soft">Clinic portal</p></div>
        </div>
        <NavLink to="/clinician" end className={link}>
          <Icon name="notifications" /> Notifications
          {open > 0 && <span className="ml-auto rounded-full bg-terracotta px-2 text-xs font-semibold text-white">{open}</span>}
        </NavLink>
        <NavLink to="/clinician/patients" className={link}><Icon name="groups" /> Patients</NavLink>
        <p className="mt-auto px-2 text-xs text-ink-soft">Each patient is compared with their own baseline. This is not a diagnosis.</p>
      </aside>
      <main className="min-w-0 flex-1 px-10 py-8">
        <Outlet />
      </main>
    </div>
  )
}
