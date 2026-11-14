// Focused lifecycle tests with a simulated browser microphone; no hardware permission is requested.
// Run: node --test scripts/test-recording.mjs
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { test } from 'node:test'
import vm from 'node:vm'
import ts from 'typescript'

const compile = (path) => ts.transpileModule(readFileSync(new URL(path, import.meta.url), 'utf8'), {
  compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.CommonJS },
}).outputText
const recorderSource = compile('../src/hooks/useRecorder.ts')
const wavSource = compile('../src/lib/wav.ts')
const settle = async () => { for (let i = 0; i < 8; i++) await Promise.resolve() }
const deferred = () => { let resolve; const promise = new Promise((done) => { resolve = done }); return { promise, resolve } }

function harness({ pendingPermission = false, failContext = false, permissionError = false, sampleRate = 16000 } = {}) {
  const state = [], cleanupEffects = [], tracks = [], contexts = [], taps = [], intervals = new Map(), timeouts = new Map()
  const permission = deferred()
  let timerId = 0, nextState = 0
  const track = new EventTarget()
  track.stopped = false
  track.stop = () => { track.stopped = true }
  tracks.push(track)
  const stream = { getTracks: () => tracks, getAudioTracks: () => tracks }
  const react = {
    useState: (initial) => { const index = nextState++; state[index] = initial; return [initial, (value) => { state[index] = value }] },
    useRef: (initial) => ({ current: initial }),
    useCallback: (callback) => callback,
    useEffect: (effect) => { const cleanup = effect(); if (cleanup) cleanupEffects.push(cleanup) },
  }
  class Context {
    constructor() {
      if (failContext) throw new Error('No audio device')
      this.sampleRate = sampleRate
      this.state = 'suspended'
      this.audioWorklet = { addModule: async () => {} }
      this.destination = {}
      contexts.push(this)
    }
    async resume() { this.state = 'running' }
    async close() { this.state = 'closed' }
    createMediaStreamSource() { return { connect() {}, disconnect() {} } }
  }
  class Tap {
    constructor() { this.port = { onmessage: null, close: () => { this.closed = true } }; taps.push(this) }
    connect() {}
    disconnect() { this.disconnected = true }
  }
  const window = {
    AudioContext: Context,
    setInterval: (callback) => { intervals.set(++timerId, callback); return timerId },
    clearInterval: (id) => intervals.delete(id),
    setTimeout: (callback) => { timeouts.set(++timerId, callback); return timerId },
    clearTimeout: (id) => timeouts.delete(id),
  }
  const exports = {}
  vm.runInNewContext(recorderSource, {
    exports, window, Blob, DOMException, Float32Array,
    AudioContext: Context, AudioWorkletNode: Tap,
    URL: { createObjectURL: () => 'blob:test-worklet', revokeObjectURL() {} },
    navigator: { mediaDevices: { getUserMedia: () => permissionError ? Promise.reject(new DOMException('Blocked', 'NotAllowedError')) : pendingPermission ? permission.promise : Promise.resolve(stream) } },
    performance: { now: () => 100 },
    require: (name) => name === 'react' ? react : {
      encodeWav16k: async (samples, rate) => { assert.equal(rate, sampleRate); return new Blob([samples], { type: 'audio/wav' }) },
    },
  })
  return { hook: exports.useRecorder(), state, stream, permission, tracks, contexts, taps, intervals, timeouts, unmount: () => cleanupEffects.forEach((cleanup) => cleanup()) }
}

test('cancelling while microphone permission is pending stops the eventual stream', async () => {
  const h = harness({ pendingPermission: true })
  const recording = h.hook.start()
  const cancelled = assert.rejects(recording, { name: 'AbortError' })
  h.hook.cancel()
  h.permission.resolve(h.stream)
  await cancelled
  await settle()
  assert.equal(h.tracks[0].stopped, true)
  assert.equal(h.contexts.length, 0)
  assert.equal(h.state[0], false)
  assert.equal(h.state[1], false)
})

test('leaving during the permission dialog releases the stream when permission resolves', async () => {
  const h = harness({ pendingPermission: true })
  const recording = h.hook.start()
  const cancelled = assert.rejects(recording, { name: 'AbortError' })
  h.unmount()
  h.permission.resolve(h.stream)
  await cancelled
  await settle()
  assert.equal(h.tracks[0].stopped, true)
  assert.equal(h.contexts.length, 0)
})

test('double-clicking Record cannot request a second microphone session', async () => {
  const h = harness({ pendingPermission: true })
  const first = h.hook.start()
  const cancelled = assert.rejects(first, { name: 'AbortError' })
  await assert.rejects(h.hook.start(), /already starting or running/)
  h.hook.cancel()
  h.permission.resolve(h.stream)
  await cancelled
  await settle()
})

test('permission denial gives an actionable message without starting audio resources', async () => {
  const h = harness({ permissionError: true })
  await assert.rejects(h.hook.start(), /Microphone access is blocked/)
  assert.equal(h.contexts.length, 0)
  assert.ok(h.state.some((value) => typeof value === 'string' && /write a check-in/.test(value)))
})

test('AudioContext construction failure still releases the acquired microphone', async () => {
  const h = harness({ failContext: true })
  await assert.rejects(h.hook.start(), /could not start/)
  assert.equal(h.tracks[0].stopped, true)
  assert.equal(h.state[0], false)
  assert.equal(h.state[1], false)
})

test('a usable memo becomes a WAV and releases tracks, nodes, context and timers', async () => {
  const h = harness()
  const recording = h.hook.start()
  await settle()
  assert.equal(h.state[0], true)
  h.taps[0].port.onmessage({ data: new Float32Array(16000 * 4).fill(0.2) })
  h.hook.stop()
  const wav = await recording
  assert.equal(wav.type, 'audio/wav')
  assert.equal(h.tracks[0].stopped, true)
  assert.equal(h.contexts[0].state, 'closed')
  assert.equal(h.taps[0].closed, true)
  assert.equal(h.taps[0].disconnected, true)
  assert.equal(h.timeouts.size, 0)
  assert.equal(h.intervals.size, 0)
  assert.equal(h.state[0], false)
  assert.equal(h.state[2], false)
})

test('short and silent captures are rejected before encoding', async () => {
  for (const [samples, problem] of [[new Float32Array(16000).fill(0.2), /little short/], [new Float32Array(16000 * 4), /silent/]]) {
    const h = harness()
    const recording = h.hook.start()
    const rejected = assert.rejects(recording, problem)
    await settle()
    h.taps[0].port.onmessage({ data: samples })
    h.hook.stop()
    await rejected
    assert.equal(h.tracks[0].stopped, true)
    assert.equal(h.contexts[0].state, 'closed')
  }
})

test('the recording limit finalizes a usable memo automatically', async () => {
  const h = harness()
  const recording = h.hook.start()
  await settle()
  h.taps[0].port.onmessage({ data: new Float32Array(16000 * 4).fill(0.2) })
  Array.from(h.timeouts.values())[0]()
  assert.equal((await recording).type, 'audio/wav')
  assert.equal(h.tracks[0].stopped, true)
})

test('the real WAV encoder writes mono 16 kHz PCM headers and clamps samples', async () => {
  const exports = {}
  vm.runInNewContext(wavSource, { exports, Blob, ArrayBuffer, DataView })
  const blob = await exports.encodeWav16k(new Float32Array([-2, -1, 0, 1, 2]), 16000)
  const bytes = await blob.arrayBuffer(), header = new DataView(bytes)
  assert.equal(blob.type, 'audio/wav')
  assert.equal(bytes.byteLength, 54)
  assert.equal(new TextDecoder().decode(bytes.slice(0, 4)), 'RIFF')
  assert.equal(new TextDecoder().decode(bytes.slice(8, 12)), 'WAVE')
  assert.equal(header.getUint16(22, true), 1)
  assert.equal(header.getUint32(24, true), 16000)
  assert.equal(header.getUint16(34, true), 16)
  assert.deepEqual([0, 1, 2, 3, 4].map((index) => header.getInt16(44 + index * 2, true)), [-32768, -32768, 0, 32767, 32767])
})
