/* Web Push for the daily prompt (#22, P2). Imported into the generated service worker
   (vite.config.ts → workbox.importScripts). The payload is fixed text from backend/app/push.py –
   never health data, because it shows on a lock screen. */
self.addEventListener('push', (event) => {
  let data = { title: 'Time for your check-in', body: 'Read the phrase, then tell me about your day.', url: '/', tag: 'daily-prompt' }
  try {
    if (event.data) data = { ...data, ...event.data.json() }
  } catch (e) {
    /* keep the default text */
  }
  event.waitUntil(
    self.registration.showNotification(data.title, {
      body: data.body,
      tag: data.tag,
      icon: '/icon.svg',
      badge: '/icon.svg',
      data: { url: data.url || '/' },
    }),
  )
})

self.addEventListener('notificationclick', (event) => {
  event.notification.close()
  const url = (event.notification.data && event.notification.data.url) || '/'
  event.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then((wins) => {
      for (const w of wins) {
        if ('focus' in w) {
          w.navigate(url)
          return w.focus()
        }
      }
      return self.clients.openWindow(url)
    }),
  )
})
