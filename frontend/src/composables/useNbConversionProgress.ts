import { isAxiosError } from 'axios'
import { onBeforeUnmount, ref } from 'vue'

import api from '../api'

export interface NbConversionProgress {
  status: 'pending' | 'running' | 'succeeded' | 'failed'
  percent: number
  stage: string
  message: string
  updated_at: string
}

function conversionId(): string {
  if (typeof crypto.randomUUID === 'function') return crypto.randomUUID()
  const bytes = crypto.getRandomValues(new Uint8Array(16))
  bytes[6] = (bytes[6]! & 0x0f) | 0x40
  bytes[8] = (bytes[8]! & 0x3f) | 0x80
  const hex = Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('')
  return `${hex.slice(0, 8)}-${hex.slice(8, 12)}-${hex.slice(12, 16)}-${hex.slice(16, 20)}-${hex.slice(20)}`
}

export function useNbConversionProgress() {
  const progress = ref<NbConversionProgress | null>(null)
  const warning = ref('')
  const elapsedSeconds = ref(0)
  let activeId: string | null = null
  let startedAt = 0
  let pollTimer: number | null = null
  let clockTimer: number | null = null
  let controller: AbortController | null = null
  let disposed = false

  function updateElapsed() {
    if (startedAt) elapsedSeconds.value = Math.max(0, Math.floor((Date.now() - startedAt) / 1000))
  }

  function stop() {
    activeId = null
    if (pollTimer !== null) window.clearTimeout(pollTimer)
    if (clockTimer !== null) window.clearInterval(clockTimer)
    pollTimer = null
    clockTimer = null
    controller?.abort()
    controller = null
  }

  async function poll(projectId: string, id: string) {
    if (disposed || activeId !== id) return
    const request = new AbortController()
    controller = request
    let terminal = false
    try {
      const { data } = await api.get<NbConversionProgress>(
        `/projects/${projectId}/conversions/onnx-to-nb/progress/${id}`,
        { signal: request.signal, timeout: 5000 },
      )
      if (disposed || activeId !== id) return
      if (!Number.isFinite(data.percent) || typeof data.message !== 'string') {
        throw new Error('Invalid conversion progress')
      }
      const percent = Math.max(progress.value?.percent ?? 0, Math.max(0, Math.min(99, data.percent)))
      if (data.status === 'succeeded') {
        progress.value = { ...data, status: 'running', percent: 99, message: '转换已完成，正在接收 NB 文件' }
        terminal = true
      } else {
        progress.value = { ...data, percent }
        terminal = data.status === 'failed'
      }
      warning.value = ''
    } catch (error) {
      if (disposed || activeId !== id || request.signal.aborted) return
      const awaitingRegistration = isAxiosError(error) && error.response?.status === 404
      warning.value = awaitingRegistration && elapsedSeconds.value < 10
        ? ''
        : '进度暂时无法刷新，转换仍在进行，正在重试。'
    } finally {
      if (controller === request) controller = null
      if (!disposed && activeId === id && !terminal) {
        pollTimer = window.setTimeout(() => { void poll(projectId, id) }, 1000)
      }
    }
  }

  function start(projectId: string): string {
    stop()
    const id = conversionId()
    activeId = id
    startedAt = Date.now()
    elapsedSeconds.value = 0
    warning.value = ''
    progress.value = {
      status: 'pending', percent: 0, stage: 'submitting', message: '正在提交转换任务',
      updated_at: new Date().toISOString(),
    }
    clockTimer = window.setInterval(updateElapsed, 1000)
    pollTimer = window.setTimeout(() => { void poll(projectId, id) }, 500)
    return id
  }

  function succeed() {
    if (disposed) return
    updateElapsed()
    stop()
    warning.value = ''
    progress.value = {
      status: 'succeeded', percent: 100, stage: 'complete', message: '转换完成，可下载 NB',
      updated_at: new Date().toISOString(),
    }
  }

  function fail(message: string) {
    if (disposed) return
    updateElapsed()
    stop()
    warning.value = ''
    progress.value = {
      status: 'failed', percent: progress.value?.percent ?? 0, stage: 'failed', message,
      updated_at: new Date().toISOString(),
    }
  }

  function reset() {
    stop()
    startedAt = 0
    elapsedSeconds.value = 0
    warning.value = ''
    progress.value = null
  }

  onBeforeUnmount(() => {
    disposed = true
    stop()
  })

  return { progress, warning, elapsedSeconds, start, succeed, fail, reset }
}
