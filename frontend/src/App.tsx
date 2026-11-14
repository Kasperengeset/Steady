import { BrowserRouter, Route, Routes } from 'react-router-dom'
import Shell from './patient/Shell'
import Today from './patient/Today'
import CheckIn from './patient/CheckIn'
import FollowUp from './patient/FollowUp'
import { ActionDetail, Insights } from './patient/Insights'
import Friends from './patient/Friends'
import Companion from './patient/Companion'
import Habits from './patient/Habits'
import Circle, { FriendCircle } from './patient/Circle'
import Portal from './clinician/Portal'
import Notifications from './clinician/Notifications'
import Patients from './clinician/Patients'
import PatientDetail from './clinician/PatientDetail'
import FriendNudge from './friend/FriendNudge'

// Patient app (mobile): /  ·  Clinician portal (desktop): /clinician  ·  A friend's view: /friend/1
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Shell />}>
          <Route index element={<Today />} />
          <Route path="checkin" element={<CheckIn />} />
          <Route path="followup" element={<FollowUp />} />
          <Route path="insights" element={<Insights />} />
          <Route path="insights/:key" element={<ActionDetail />} />
          <Route path="friends" element={<Friends />} />
          <Route path="appie" element={<Companion />} />
          <Route path="habits" element={<Habits />} />
          <Route path="circle" element={<Circle />} />
        </Route>
        <Route path="clinician" element={<Portal />}>
          <Route index element={<Notifications />} />
          <Route path="patients" element={<Patients />} />
          <Route path="patients/:id" element={<PatientDetail />} />
        </Route>
        <Route path="friend/:pid" element={<FriendNudge />} />
        <Route path="join" element={<FriendCircle />} />
      </Routes>
    </BrowserRouter>
  )
}
