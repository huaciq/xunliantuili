const assert = require('node:assert/strict')
const { webcrypto } = require('node:crypto')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')
const vm = require('node:vm')
const ts = require('typescript')

function harness(sourceFile, responses = [], options = {}) {
  const timers = new Map()
  const cleanup = []
  const calls = []
  let now = 1_000_000
  let timerId = 0
  const timer = (kind, callback) => {
    timers.set(++timerId, { kind, callback })
    return timerId
  }
  const api = { get: async (url, options) => {
    calls.push({ url, options })
    const response = responses.shift()
    if (response instanceof Error || response?.isAxiosError) throw response
    return response
  } }
  const exported = { exports: {} }
  const source = fs.readFileSync(path.join(__dirname, '../src/composables', sourceFile), 'utf8')
  const javascript = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022, esModuleInterop: true },
  }).outputText
  const context = {
    module: exported, exports: exported.exports,
    require(name) {
      if (name === 'vue') return { ref: (value) => ({ value }), onBeforeUnmount: (callback) => cleanup.push(callback) }
      if (name === 'axios') return { isAxiosError: (error) => Boolean(error?.isAxiosError) }
      if (name === '../api') return { __esModule: true, default: api }
      throw new Error(`Unexpected dependency ${name}`)
    },
    Date: class extends Date { static now() { return now } },
    crypto: options.insecureHttp
      ? { getRandomValues: webcrypto.getRandomValues.bind(webcrypto) }
      : webcrypto,
    AbortController,
    window: {
      setTimeout: (callback) => timer('timeout', callback),
      setInterval: (callback) => timer('interval', callback),
      clearTimeout: (id) => timers.delete(id), clearInterval: (id) => timers.delete(id),
    },
  }
  vm.runInNewContext(javascript, context, { filename: sourceFile })
  return {
    exports: exported.exports, timers, calls,
    advance(ms) {
      now += ms
      for (const entry of [...timers.values()]) if (entry.kind === 'interval') entry.callback()
    },
    async poll() {
      const entry = [...timers.entries()].find(([, value]) => value.kind === 'timeout')
      assert.ok(entry, 'A progress refresh should be scheduled')
      timers.delete(entry[0])
      entry[1].callback()
      await new Promise(setImmediate)
    },
    dispose() { cleanup.forEach((callback) => callback()) },
  }
}

function progress(status, percent, message = '正在量化校准') {
  return { data: { status, percent, stage: 'quantize', message, updated_at: '2026-11-20T00:00:00Z' } }
}

test('displays server stages without increasing percentage with elapsed time', async () => {
  const browser = harness('useNbConversionProgress.ts', [progress('running', 35)])
  const tracker = browser.exports.useNbConversionProgress()
  const id = tracker.start('project-1')
  assert.match(id, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/)
  await browser.poll()
  assert.equal(browser.calls[0].url, `/projects/project-1/conversions/onnx-to-nb/progress/${id}`)
  assert.equal(tracker.progress.value.percent, 35)
  browser.advance(25_000)
  assert.equal(tracker.elapsedSeconds.value, 25)
  assert.equal(tracker.progress.value.percent, 35)
  browser.dispose()
  assert.equal(browser.timers.size, 0)
})

test('missing initial progress and network failures do not terminate conversion tracking', async () => {
  const browser = harness('useNbConversionProgress.ts', [
    { isAxiosError: true, response: { status: 404 } }, new Error('offline'), progress('running', 75),
  ])
  const tracker = browser.exports.useNbConversionProgress()
  tracker.start('project-2')
  await browser.poll()
  assert.equal(tracker.warning.value, '')
  await browser.poll()
  assert.ok(tracker.warning.value)
  assert.equal(tracker.progress.value.status, 'pending')
  await browser.poll()
  assert.equal(tracker.warning.value, '')
  assert.equal(tracker.progress.value.percent, 75)
  tracker.fail('模型转换失败')
  assert.equal(tracker.progress.value.status, 'failed')
  assert.equal(tracker.progress.value.percent, 75)
  assert.equal(browser.timers.size, 0)
})

test('100 percent waits for the completed file response, then polling and clock stop', async () => {
  const browser = harness('useNbConversionProgress.ts', [progress('succeeded', 100, '已完成')])
  const tracker = browser.exports.useNbConversionProgress()
  tracker.start('project-3')
  await browser.poll()
  assert.equal(tracker.progress.value.percent, 99)
  assert.equal(tracker.progress.value.status, 'running')
  assert.equal([...browser.timers.values()].filter((entry) => entry.kind === 'timeout').length, 0)
  tracker.succeed()
  assert.equal(tracker.progress.value.status, 'succeeded')
  assert.equal(tracker.progress.value.percent, 100)
  assert.equal(browser.timers.size, 0)
})

test('unmount aborts an in-flight refresh and late responses cannot change state', async () => {
  let resolve
  const delayed = new Promise((callback) => { resolve = callback })
  const browser = harness('useNbConversionProgress.ts', [delayed])
  const tracker = browser.exports.useNbConversionProgress()
  tracker.start('project-4')
  await browser.poll()
  browser.dispose()
  assert.equal(browser.calls[0].options.signal.aborted, true)
  assert.equal(browser.timers.size, 0)
  resolve(progress('running', 90))
  await new Promise(setImmediate)
  assert.equal(tracker.progress.value.percent, 0)
  tracker.succeed()
  assert.equal(tracker.progress.value.percent, 0)
})

test('HTTP pages without randomUUID still send a valid version 4 conversion ID', async () => {
  const browser = harness('useNbConversionProgress.ts', [progress('running', 20)], { insecureHttp: true })
  const tracker = browser.exports.useNbConversionProgress()
  const id = tracker.start('project-http')
  assert.match(id, /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/)
  await browser.poll()
  assert.equal(browser.calls[0].url, `/projects/project-http/conversions/onnx-to-nb/progress/${id}`)
  browser.dispose()
  assert.equal(browser.timers.size, 0)
})
