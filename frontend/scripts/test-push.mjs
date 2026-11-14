import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import vm from 'node:vm'

const source = readFileSync(new URL('../public/push-sw.js', import.meta.url), 'utf8')
function worker() {
  const handlers = {}, notices = [], navigations = []
  const self = {
    location: { origin: 'https://steady.test' },
    addEventListener: (name, callback) => { handlers[name] = callback },
    registration: { showNotification: async (title, options) => { notices.push({ title, options }) } },
    clients: { matchAll: async () => [], openWindow: async (url) => { navigations.push(url) } },
  }
  vm.runInNewContext(source, { self, URL })
  return { handlers, notices, navigations, self }
}

test('a daily prompt shows one tagged notification and opens the check-in flow', async () => {
  const w = worker()
  let done
  w.handlers.push({ data: { json: () => ({ title: 'Time for your check-in', body: 'Read the phrase.', url: '/', tag: 'daily-prompt' }) }, waitUntil: (promise) => { done = promise } })
  await done
  assert.equal(w.notices[0].options.tag, 'daily-prompt')
  assert.equal(w.notices[0].options.data.url, 'https://steady.test/checkin')
  let closed = false
  w.handlers.notificationclick({ notification: { data: w.notices[0].options.data, close: () => { closed = true } }, waitUntil: (promise) => { done = promise } })
  await done
  assert.ok(closed)
  assert.deepEqual(w.navigations, ['https://steady.test/checkin'])
})

test('malformed or cross-origin push destinations stay within Steady', async () => {
  for (const url of ['http://[bad', 'https://other.test/private', 'javascript:alert(1)']) {
    const w = worker()
    let done
    w.handlers.push({ data: { json: () => ({ url }) }, waitUntil: (promise) => { done = promise } })
    await done
    assert.equal(w.notices[0].options.data.url, 'https://steady.test/checkin')
  }
})

test('an invalid payload still gives a useful notification without exposing data', async () => {
  const w = worker()
  let done
  w.handlers.push({ data: { json: () => { throw new Error('invalid JSON') } }, waitUntil: (promise) => { done = promise } })
  await done
  assert.equal(w.notices[0].title, 'Time for your check-in')
  assert.equal(w.notices[0].options.data.url, 'https://steady.test/checkin')
})
