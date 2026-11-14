import { useCallback } from 'react'
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom'
import { views } from '../lib/views'
import { useClinicalResource } from '../lib/clinical'
import { Icon } from '../ui/kit'
import { DemoButton } from './ClinicalUI'
import './clinical.css'

export default function Portal() {
  const location = useLocation()
  const { data: alerts } = useClinicalResource(useCallback(() => views.notifications('open'), []))
  const section = location.pathname.includes('/patients/') ? 'Patient workspace' : location.pathname.includes('/models') ? 'Model evidence' : location.pathname.includes('/notifications') ? 'Care inbox' : 'Cohort overview'
  const link = ({ isActive }: { isActive: boolean }) => `cl-nav-link ${isActive ? 'is-active' : ''}`
  return <div className="clinical-app">
    <a href="#clinical-content" className="cl-skip-link">Skip to workspace</a>
    <aside className="cl-sidebar">
      <Link to="/clinician" className="cl-brand"><span><Icon name="spa" /></span><div>Steady<small>CLINICAL WORKSPACE</small></div></Link>
      <div className="cl-sidebar-label">CARE, WITH CONTEXT</div>
      <nav aria-label="Clinical navigation"><NavLink to="/clinician" end className={link}><Icon name="space_dashboard" />Overview</NavLink><NavLink to="/clinician/patients" className={link}><Icon name="group" />Patients</NavLink><NavLink to="/clinician/notifications" className={link}><Icon name="notifications_none" />Care inbox{Boolean(alerts?.length) && <b>{alerts!.length}</b>}</NavLink><NavLink to="/clinician/models" className={link}><Icon name="hub" />Model evidence<span className="cl-nav-new">3</span></NavLink></nav>
      <div className="cl-sidebar-note"><Icon name="account_tree" /><strong>Three signals.<br />One personal baseline.</strong><p>Voice, language and wearable measurements in one traceable care record.</p><Link to="/clinician/models">Explore the evidence <Icon name="arrow_forward" /></Link></div>
      <div className="cl-sidebar-bottom"><span className="cl-team-avatar">CT</span><div><strong>Care team workspace</strong><small>Clinical observations · local time</small></div></div>
    </aside>
    <div className="cl-main-shell"><header className="cl-topbar"><div className="cl-breadcrumb"><span>Care workspace</span><Icon name="chevron_right" /><strong>{section}</strong></div><div className="cl-topbar-actions"><span className="cl-topbar-date">{new Date().toLocaleDateString('en-GB', { weekday: 'short', day: 'numeric', month: 'long' })}</span><DemoButton compact /><Link to="/" className="cl-icon-button" title="Open patient app" aria-label="Open patient app"><Icon name="phone_iphone" /></Link></div></header><main id="clinical-content" className="cl-content"><Outlet /></main><footer className="cl-footer"><span>Steady · Beyond the clinic</span><span>Measurements inform a conversation. They do not establish a diagnosis.</span></footer></div>
  </div>
}
