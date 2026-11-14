/* Canonical prompt delivery: lock-screen text contains no health or diary data. */
self.addEventListener('push', (event) => {
  let payload = {}
  try { payload = event.data?.json() || {} } catch { /* fixed safe defaults */ }
  const title = typeof payload.title === 'string' ? payload.title : 'Time for your check-in'
  let url = new URL('/checkin', self.location.origin)
  try {
    url = new URL(payload.tag === 'daily-prompt' && payload.url === '/' ? '/checkin' :
      typeof payload.url === 'string' ? payload.url : '/checkin', self.location.origin)
  } catch { /* malformed data keeps the local check-in destination */ }
  event.waitUntil(self.registration.showNotification(title, {
    body: typeof payload.body === 'string' ? payload.body : 'A little moment for you and Appie.',
    icon: '/icon.svg', badge: '/icon.svg',
    tag: typeof payload.tag === 'string' ? payload.tag : 'daily-prompt',
    data: { url: url.origin === self.location.origin ? url.href : self.location.origin + '/checkin' },
  }))
})
self.addEventListener('notificationclick', (event) => {
  event.notification.close()
  let url
  try { url = new URL(event.notification.data?.url || '/checkin', self.location.origin) }
  catch { url = new URL('/checkin', self.location.origin) }
  if (url.origin !== self.location.origin) return
  event.waitUntil(self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then(async (clients) => {
    const existing = clients.find((client) => new URL(client.url).origin === self.location.origin)
    if (existing) { await existing.navigate(url.href); return existing.focus() }
    return self.clients.openWindow(url.href)
  }))
})
