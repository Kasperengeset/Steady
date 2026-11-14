// Render the real inspector with controlled API data; no browser or patient data is used.
import assert from 'node:assert/strict'
import { readFileSync } from 'node:fs'
import { createRequire } from 'node:module'
import { test } from 'node:test'
import vm from 'node:vm'
import React from 'react'
import { renderToStaticMarkup } from 'react-dom/server'
import ts from 'typescript'

const require = createRequire(import.meta.url)
function compile(file, dependencies = {}) {
  const code = ts.transpileModule(readFileSync(new URL(file, import.meta.url), 'utf8'), {
    compilerOptions: { target: ts.ScriptTarget.ES2020, module: ts.ModuleKind.CommonJS, jsx: ts.JsxEmit.ReactJSX },
  }).outputText
  const module = { exports: {} }
  vm.runInNewContext(code, { module, exports: module.exports, require: (name) => dependencies[name] || require(name), Date, console, Blob, URL }, { filename: file })
  return module.exports
}
const helpers = compile('../src/lib/clinical.ts')
const primitive = ({ children, title, ...props }) => React.createElement('div', { 'aria-label': props['aria-label'] }, title, children)
function render(detail, expectedPatientId = 1) {
  const { default: Inspector } = compile('../src/clinician/EntryInspector.tsx', {
    '../lib/clinical': { ...helpers, useClinicalResource: () => ({ data: detail, error: '', refresh() {} }) },
    '../ui/kit': { Icon: ({ name }) => React.createElement('span', null, name) },
    './ClinicalUI': { Badge: primitive, EmptyState: primitive, ErrorState: primitive, OriginBadge: primitive, Panel: primitive, Score: primitive },
  })
  return renderToStaticMarkup(React.createElement(Inspector, { id: detail.entry.id, expectedPatientId }))
}
function detail() {
  return {
    entry: { id: 101, patient_id: 1, ts: '2026-11-14T12:00:00Z', slot: 'evening', kind: 'text', data_origin: 'live', crisis: false },
    transcript: 'PRIVATE_DIARY_SENTINEL', features: [{ name: 'text_logit', source: 'text', branch: 'text', value: 1.2, used: true, z: 2, center: 0, scale: .6, share: 1 }],
    analysis: { provenance: 'snapshot', baseline_origin: 'live', baseline_version: 1, before: null, after: { S: 1, driver: 'anchored' }, tracks: {} },
    raw_outputs: { text: { text_logit: 1.2, p_depressive: .77 }, acoustic: {}, health: {} }, word_evidence: { available: false }, limitations: [],
  }
}

test('a mismatched patient context never renders the transcript, features or download', () => {
  const data = detail(); data.entry.patient_id = 2
  const html = render(data)
  assert.match(html, /different care record/)
  assert.doesNotMatch(html, /PRIVATE_DIARY_SENTINEL|text_logit|Download entry analysis|Depressive-language pattern/)
})

test('the matching patient can inspect actual outputs and download the stored evidence', () => {
  const html = render(detail())
  assert.match(html, /PRIVATE_DIARY_SENTINEL/)
  assert.match(html, /Download entry analysis JSON/)
  assert.match(html, /77.0%/)
  assert.match(html, /not the probability of depression or relapse/)
  assert.match(html, /text_logit/)
})

test('voice without transcription clearly omits language rather than displaying an invented score', () => {
  const data = detail(); data.entry.kind = 'voice'; data.transcript = null
  data.raw_outputs = { text: null, acoustic: { arousal: .42, valence: .55 }, health: {} }
  const html = render(data)
  assert.match(html, /Transcription was unavailable/)
  assert.match(html, /no model observation for this entry/)
  assert.doesNotMatch(html, /PRIVATE_DIARY_SENTINEL|Depressive-language pattern|77\.0%/)
})
