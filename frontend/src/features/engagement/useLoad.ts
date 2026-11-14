import { useCallback, useEffect, useRef, useState } from 'react'

export const DEMO_PATIENT_ID = 1 // same demo patient as PatientView until there is a login

/** Load data once, reload on demand, and (optionally) again when the tab gets focus. */
export function useLoad<T>(load: () => Promise<T>, opts: { refreshOnFocus?: boolean } = {}) {
  const [data, setData] = useState<T | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loaded, setLoaded] = useState(false)
  const [version, setVersion] = useState(0)
  const loadRef = useRef(load)
  loadRef.current = load

  useEffect(() => {
    let live = true
    loadRef.current().then(
      (d) => live && (setData(d), setError(null), setLoaded(true)),
      (e: unknown) => live && setError(e instanceof Error ? e.message : String(e)),
    )
    return () => {
      live = false
    }
  }, [version])

  const reload = useCallback(() => setVersion((v) => v + 1), [])

  useEffect(() => {
    if (!opts.refreshOnFocus) return
    window.addEventListener('focus', reload)
    return () => window.removeEventListener('focus', reload)
  }, [opts.refreshOnFocus, reload])

  return { data, error, loaded, reload, setData }
}

/** Browser storage can be blocked (private mode): never let that break a page. */
export const store = {
  get(key: string): string | null {
    try {
      return localStorage.getItem(key)
    } catch {
      return null
    }
  },
  set(key: string, value: string | null) {
    try {
      if (value === null) localStorage.removeItem(key)
      else localStorage.setItem(key, value)
    } catch {
      /* ignore */
    }
  },
}
