<script setup lang="ts">
import { ArrowLeft, BarChart3, CircleStop, Download, FileBox, PackagePlus, RefreshCw, Terminal } from 'lucide-vue-next'
import { NButton, NDescriptions, NDescriptionsItem, NForm, NFormItem, NInput, NModal, NSelect, NSpin, NTag, useDialog, useMessage } from 'naive-ui'
import { computed, onBeforeUnmount, onMounted, reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import api, { errorMessage, websocketUrl } from '../api'
import type { MetricPoint, RunArtifact, TrainingRun, TrainingRunStatus } from '../types'

const route = useRoute()
const router = useRouter()
const message = useMessage()
const dialog = useDialog()
const run = ref<TrainingRun | null>(null)
const logs = ref<string[]>([])
const metrics = ref<MetricPoint[]>([])
const artifacts = ref<RunArtifact[]>([])
const loading = ref(false)
const registering = ref(false)
const showRegister = ref(false)
const registerForm = reactive({ name: '', description: '', stage: 'candidate' })
let socket: WebSocket | null = null
let reconnectTimer: number | null = null

const terminalStatuses = new Set<TrainingRunStatus>(['succeeded', 'failed', 'stopped'])
const canStop = computed(() => run.value && ['queued', 'preparing', 'starting', 'running'].includes(run.value.status))
const canRegister = computed(() => run.value?.status === 'succeeded' && artifacts.value.some((item) => item.artifact_type === 'checkpoint'))
const metricGroups = computed(() => {
  const groups = new Map<string, MetricPoint[]>()
  for (const point of metrics.value) groups.set(point.key, [...(groups.get(point.key) ?? []), point])
  return [...groups.entries()].map(([key, points]) => ({ key, points, latest: points.at(-1)?.value ?? 0 }))
})

async function load() {
  loading.value = true
  try {
    run.value = (await api.get<TrainingRun>(`/runs/${route.params.runId}`)).data
    const [metricResponse, artifactResponse] = await Promise.all([
      api.get<MetricPoint[]>(`/runs/${route.params.runId}/metrics`),
      api.get<RunArtifact[]>(`/runs/${route.params.runId}/artifacts`),
    ])
    metrics.value = metricResponse.data
    artifacts.value = artifactResponse.data
  }
  catch (error) { message.error(errorMessage(error)) }
  finally { loading.value = false }
}

async function registerModel() {
  registering.value = true
  try {
    await api.post(`/runs/${route.params.runId}/register-model`, registerForm)
    showRegister.value = false
    message.success('模型版本已注册')
    if (run.value) router.push(`/projects/${run.value.project_id}/models`)
  } catch (error) { message.error(errorMessage(error)); return false }
  finally { registering.value = false }
}

async function downloadArtifact(artifact: RunArtifact) {
  try {
    const response = await api.get(`/artifacts/${artifact.id}/download`, { responseType: 'blob' })
    const url = URL.createObjectURL(response.data)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = artifact.name
    anchor.click()
    URL.revokeObjectURL(url)
  } catch (error) { message.error(errorMessage(error)) }
}

function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`
}

function connectLogs() {
  if (!run.value || socket) return
  const token = localStorage.getItem('access_token')
  if (!token) return
  socket = new WebSocket(`${websocketUrl(`/runs/${run.value.id}/logs/stream`)}?token=${encodeURIComponent(token)}`)
  socket.onmessage = (event) => {
    const payload = JSON.parse(event.data) as { status: TrainingRunStatus; lines: string[] }
    logs.value = payload.lines
    if (run.value) run.value.status = payload.status
    if (terminalStatuses.has(payload.status)) {
      socket?.close()
      socket = null
      load()
    }
  }
  socket.onclose = () => {
    socket = null
    if (run.value && !terminalStatuses.has(run.value.status)) {
      reconnectTimer = window.setTimeout(connectLogs, 1500)
    }
  }
}

async function stopRun() {
  dialog.warning({
    title: '停止训练任务',
    content: '任务将收到停止请求，开发执行器会释放已分配的 GPU。',
    positiveText: '停止任务',
    negativeText: '取消',
    async onPositiveClick() {
      try {
        run.value = (await api.post<TrainingRun>(`/runs/${route.params.runId}/stop`)).data
      } catch (error) { message.error(errorMessage(error)) }
    },
  })
}

function statusType(status: TrainingRunStatus): 'default' | 'info' | 'warning' | 'success' | 'error' {
  if (status === 'succeeded') return 'success'
  if (status === 'failed') return 'error'
  if (status === 'running') return 'info'
  if (['queued', 'preparing', 'stopping', 'cancel_requested'].includes(status)) return 'warning'
  return 'default'
}

onMounted(async () => { await load(); connectLogs() })
onBeforeUnmount(() => {
  socket?.close()
  if (reconnectTimer) window.clearTimeout(reconnectTimer)
})
</script>

<template>
  <section class="page">
    <NButton v-if="run" quaternary class="back-button" @click="router.push(`/projects/${run.project_id}/runs`)"><template #icon><ArrowLeft /></template>返回任务列表</NButton>
    <NSpin :show="loading">
      <template v-if="run">
        <header class="page-header detail-header">
          <div><h1>{{ run.name }}</h1><p>{{ run.template_name }} · {{ run.runtime_image_name }}</p></div>
          <div class="page-actions"><NTag :type="statusType(run.status)" size="large">{{ run.status }}</NTag><NButton quaternary circle title="刷新" @click="load"><RefreshCw :size="18" /></NButton><NButton v-if="canRegister" type="primary" @click="showRegister = true"><template #icon><PackagePlus /></template>注册模型</NButton><NButton v-if="canStop" type="error" secondary @click="stopRun"><template #icon><CircleStop /></template>停止</NButton></div>
        </header>
        <NDescriptions bordered :column="2" label-placement="left" class="run-descriptions">
          <NDescriptionsItem label="数据版本">{{ run.dataset_label }}</NDescriptionsItem>
          <NDescriptionsItem label="代码版本">{{ run.code_label }}</NDescriptionsItem>
          <NDescriptionsItem label="GPU 请求">{{ run.requested_gpu_count }} × {{ run.requested_gpu_model }}</NDescriptionsItem>
          <NDescriptionsItem label="已分配 GPU">{{ run.allocated_gpus.join(', ') || '等待分配' }}</NDescriptionsItem>
          <NDescriptionsItem label="退出码">{{ run.exit_code ?? '-' }}</NDescriptionsItem>
          <NDescriptionsItem label="执行 ID">{{ run.execution_id ?? '-' }}</NDescriptionsItem>
          <NDescriptionsItem label="命令" :span="2"><code>{{ run.command.join(' ') }}</code></NDescriptionsItem>
        </NDescriptions>
        <section v-if="metricGroups.length" class="content-section result-section">
          <header class="section-header"><div><h2>训练指标</h2><p>开发执行器采集的逐轮实验结果。</p></div><BarChart3 :size="20" /></header>
          <div class="metric-grid">
            <div v-for="group in metricGroups" :key="group.key" class="metric-item">
              <div><span>{{ group.key }}</span><strong>{{ group.latest.toFixed(4) }}</strong></div>
              <div class="metric-bars"><i v-for="point in group.points" :key="point.id" :style="{ height: `${Math.max(8, Math.min(100, point.value * 100))}%` }" :title="`step ${point.step}: ${point.value}`" /></div>
            </div>
          </div>
        </section>
        <section v-if="artifacts.length" class="content-section result-section">
          <header class="section-header"><div><h2>训练产物</h2><p>checkpoint、指标报告和模型导出文件。</p></div><FileBox :size="20" /></header>
          <div class="artifact-list">
            <div v-for="artifact in artifacts" :key="artifact.id"><div><strong>{{ artifact.name }}</strong><span>{{ artifact.artifact_type }} · {{ formatBytes(artifact.size_bytes) }}</span></div><code>{{ artifact.sha256.slice(0, 12) }}</code><NButton quaternary circle title="下载" @click="downloadArtifact(artifact)"><Download :size="17" /></NButton></div>
          </div>
        </section>
        <section class="terminal-panel">
          <header><div><Terminal :size="18" /><strong>实时日志</strong></div><span>{{ logs.length }} 行</span></header>
          <pre>{{ logs.join('\n') || '等待任务日志...' }}</pre>
        </section>
        <section v-if="run.events.length" class="content-section">
          <header class="section-header"><div><h2>状态记录</h2><p>任务生命周期事件。</p></div></header>
          <div class="event-list"><div v-for="event in run.events" :key="event.id"><time>{{ new Date(event.created_at).toLocaleString('zh-CN', { hour12: false }) }}</time><span>{{ event.message }}</span></div></div>
        </section>
      </template>
    </NSpin>
    <NModal v-model:show="showRegister" preset="dialog" title="注册模型版本" positive-text="注册" negative-text="取消" :loading="registering" @positive-click="registerModel">
      <NForm :model="registerForm" label-placement="top">
        <NFormItem label="模型名称"><NInput v-model:value="registerForm.name" placeholder="例如 PCB 缺陷检测器" /></NFormItem>
        <NFormItem label="说明"><NInput v-model:value="registerForm.description" type="textarea" /></NFormItem>
        <NFormItem label="阶段"><NSelect v-model:value="registerForm.stage" :options="[{ label: '候选', value: 'candidate' }, { label: '生产', value: 'production' }]" /></NFormItem>
      </NForm>
    </NModal>
  </section>
</template>
