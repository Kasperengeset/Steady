import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import PatientView from './views/PatientView'
import ClinicianView from './views/ClinicianView'
import FriendView from './views/FriendView'
import { CirclePage, FriendCirclePage, HabitsPage } from './features/engagement'

export default function App() {
  return (
    <BrowserRouter>
      <main>
        <Routes>
          <Route path="/" element={<PatientView />} />
          <Route path="/clinician" element={<ClinicianView />} />
          <Route path="/friends" element={<FriendView />} />
          <Route path="/habits" element={<HabitsPage />} />
          <Route path="/circle" element={<CirclePage />} />
          <Route path="/circle/friend" element={<FriendCirclePage />} />
        </Routes>
      </main>
      <nav className="tabs">
        <NavLink to="/" end>Patient</NavLink>
        <NavLink to="/clinician">Clinician</NavLink>
        <NavLink to="/friends">Friends</NavLink>
        <NavLink to="/habits">Habits</NavLink>
        <NavLink to="/circle">Circle</NavLink>
      </nav>
    </BrowserRouter>
  )
}
