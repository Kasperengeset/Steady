import { BrowserRouter, NavLink, Route, Routes } from 'react-router-dom'
import PatientView from './views/PatientView'
import ClinicianView from './views/ClinicianView'
import FriendView from './views/FriendView'

export default function App() {
  return (
    <BrowserRouter>
      <main>
        <Routes>
          <Route path="/" element={<PatientView />} />
          <Route path="/clinician" element={<ClinicianView />} />
          <Route path="/friends" element={<FriendView />} />
        </Routes>
      </main>
      <nav className="tabs">
        <NavLink to="/" end>Pasient</NavLink>
        <NavLink to="/clinician">Behandler</NavLink>
        <NavLink to="/friends">Venner</NavLink>
      </nav>
    </BrowserRouter>
  )
}
