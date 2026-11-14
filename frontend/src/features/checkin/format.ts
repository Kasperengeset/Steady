/** "1 h 23 min" / "12 min" / "45 s" – for the on-time window. */
export function formatLeft(s: number): string {
  if (s >= 3600) return `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min`
  if (s >= 60) return `${Math.ceil(s / 60)} min`
  return `${s} s`
}
