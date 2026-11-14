import { useCallback, useEffect, useState } from 'react'
import { checkinApi, type PromptSettingsIn, type PromptStatus } from '../../lib/api_checkin'

const POLL_MS = 30_000

/**
 * Today's daily prompt (#22). Polls the server every 30 s and when the tab comes back into view, and
 * counts the on-time window down locally every second. Without Web Push the app shows the prompt when
 * opened, so the poll is what makes the banner appear while the app is open.
 */
export function usePrompt(patientId: number) {
  const [status, setStatus] = useState<PromptStatus | null>(null)
  const [fetchedAt, setFetchedAt] = useState(0)
  const [now, setNow] = useState(() => Date.now())
  const [error, setError] = useState<string | null>(null)

  const apply = useCallback((p: PromptStatus) => {
    setStatus(p)
    setFetchedAt(Date.now())
    setError(null)
  }, [])

  const refresh = useCallback(async () => {
    try {
      apply(await checkinApi.prompt(patientId))
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    }
  }, [patientId, apply])

  useEffect(() => {
    void refresh()
    const poll = window.setInterval(() => void refresh(), POLL_MS)
    const tick = window.setInterval(() => setNow(Date.now()), 1000)
    const onVisible = () => document.visibilityState === 'visible' && void refresh()
    document.addEventListener('visibilitychange', onVisible)
    window.addEventListener('focus', onVisible)
    return () => {
      clearInterval(poll)
      clearInterval(tick)
      document.removeEventListener('visibilitychange', onVisible)
      window.removeEventListener('focus', onVisible)
    }
  }, [refresh])

  // A scheduled prompt opens on its own: refresh right when it fires instead of waiting for the poll
  useEffect(() => {
    if (status?.state !== 'scheduled' || !status.at) return
    const ms = new Date(status.at).getTime() - Date.now()
    if (ms > POLL_MS) return
    const t = window.setTimeout(() => void refresh(), Math.max(ms, 0) + 500)
    return () => clearTimeout(t)
  }, [status, refresh])

  const sendNow = useCallback(async () => apply(await checkinApi.promptNow(patientId)), [patientId, apply])
  const updateSettings = useCallback(
    async (body: PromptSettingsIn) => apply(await checkinApi.promptSettings(patientId, body)),
    [patientId, apply],
  )

  const secondsLeft =
    status?.open && status.on_time_left_s !== null
      ? Math.max(0, status.on_time_left_s - Math.floor(Math.max(0, now - fetchedAt) / 1000))
      : null

  return { status, secondsLeft, error, refresh, sendNow, updateSettings }
}
