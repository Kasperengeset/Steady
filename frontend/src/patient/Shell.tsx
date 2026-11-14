// Mobile frame for the patient app: header + one screen + bottom navigation (Stitch "Steady" design).
import { NavLink, Outlet } from 'react-router-dom'
import { Icon } from '../ui/kit'

/** Demo: the patient is chosen with ?p=<id> once (default 1, Alex) and remembered for the session. */
export const PID: number = (() => {
  const q = new URLSearchParams(window.location.search).get('p')
  if (q) sessionStorage.setItem('steady.pid', q)
  return Number(sessionStorage.getItem('steady.pid') ?? 1) || 1
})()

const tabs = [
  { to: '/', label: 'Today', icon: 'home', end: true },
  { to: '/habits', label: 'Habits', icon: 'task_alt' },
  { to: '/checkin', label: 'Check in', icon: 'mic' },
  { to: '/appie', label: 'Appie', icon: 'pets' },
  { to: '/circle', label: 'Circle', icon: 'group' },
]

export default function Shell() {
  return (
    <div className="patient-shell mx-auto flex min-h-full max-w-[480px] flex-col bg-surface">
      <header className="sticky top-0 z-10 flex items-center gap-3 bg-surface/90 px-5 pb-3 pt-5 backdrop-blur">
        <span className="grid h-10 w-10 place-items-center rounded-full bg-sage-soft text-primary"><Icon name="spa" /></span>
        <span className="flex-1 text-xl font-semibold tracking-tight">Steady<span className="ml-2 text-[11px] font-medium uppercase tracking-[.15em] text-outline">a little, daily</span></span>
        <NavLink to="/insights" aria-label="Your insights" className="grid h-10 w-10 place-items-center rounded-full bg-surface-high/60 text-primary"><Icon name="insights" /></NavLink>
      </header>
      <main className="flex-1 px-5 pb-28">
        <Outlet />
      </main>
      <nav aria-label="Main navigation" className="patient-nav fixed inset-x-0 bottom-0 z-10 mx-auto flex max-w-[480px] justify-around border-t border-surface-high bg-surface-low/95 px-2 pb-4 pt-2 backdrop-blur">
        {tabs.map((t) => (
          <NavLink key={t.to} to={t.to} end={t.end}
                   className={({ isActive }) => `flex min-w-0 flex-1 flex-col items-center gap-1 text-[11px] ${isActive ? 'font-semibold text-primary' : 'text-outline'}`}>
            {({ isActive }) => (
              <>
                <span className={`grid h-8 w-14 place-items-center rounded-full ${isActive ? 'bg-sage-soft' : ''}`}>
                  <Icon name={t.icon} fill={isActive} />
                </span>
                {t.label}
              </>
            )}
          </NavLink>
        ))}
      </nav>
    </div>
  )
}
