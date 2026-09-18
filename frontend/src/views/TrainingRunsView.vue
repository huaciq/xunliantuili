<script setup lang="ts">
import { ArrowLeft, Cpu, Plus, RefreshCw } from 'lucide-vue-next'
import {
  NButton,
  NEmpty,
  NForm,
  NFormItem,
  NInput,
  NInputNumber,
  NModal,
  NSelect,
  NSpin,
  NTag,
  useMessage,
} from 'naive-ui'
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import api, { errorMessage } from '../api'
import type {
  CodePackage,
  Dataset,
  GpuDevice,
  GpuModelPolicy,
  Project,
  RuntimeImage,
  TrainingRun,
  TrainingRunStatus,
  TrainingTemplate,
} from '../types'

const route = useRoute()
const router = useRouter()
const message = useMessage()
const projectId = computed(() => String(route.params.projectId))
const project = ref<Project | null>(null)
const runs = ref<TrainingRun[]>([])
const datasets = ref<Dataset[]>([])
const codePackages = ref<CodePackage[]>([])
const templates = ref<TrainingTemplate[]>([])
const runtimes = ref<RuntimeImage[]>([])
const gpus = ref<GpuDevice[]>([])
const loading = ref(false)
const saving = ref(false)
const showCreate = ref(false)

const form = reactive({
  name: '',
  template_id: '',
  dataset_version_id: '',
  code_version_id: '',
  runtime_image_id: '',
  requested_gpu_count: 1,
  requested_gpu_model: 'any' as GpuModelPolicy,
  priority: 0,
  epochs: 100,
  batch: 16,
  imgsz: 640,
  model: 'yolo11n.pt',
  arguments: '',
})

const selectedTemplate = computed(() => templates.value.find((item) => item.id === form.template_id))
const datasetOptions = computed(() =>
  datasets.value.flatMap((dataset) =>
    dataset.versions
      .filter((version) => version.status === 'ready')
      .map((version) => ({ label: `${dataset.name} v${version.version} · ${version.format}`, value: version.id })),
  ),
)
const codeOptions = computed(() =>
  codePackages.value.flatMap((item) =>
    item.versions
      .filter((version) => version.status === 'ready')
      .map((version) => ({ label: `${item.name} v${version.version}`, value: version.id })),
  ),
)

async function load() {
  loading.value = true
  try {
    const [projectResponse, runResponse, datasetResponse, codeResponse, templateResponse, runtimeResponse, gpuResponse] = await Promise.all([
      api.get<Project>(`/projects/${projectId.value}`),
      api.get<TrainingRun[]>(`/projects/${projectId.value}/runs`),
      api.get<Dataset[]>(`/projects/${projectId.value}/datasets`),
      api.get<CodePackage[]>(`/projects/${projectId.value}/code-packages`),
      api.get<TrainingTemplate[]>('/training-templates'),
      api.get<RuntimeImage[]>('/runtime-images'),
      api.get<GpuDevice[]>('/resources/gpus'),
    ])
    project.value = projectResponse.data
    runs.value = runResponse.data
    datasets.value = datasetResponse.data
    codePackages.value = codeResponse.data
    templates.value = templateResponse.data
    runtimes.value = runtimeResponse.data
    gpus.value = gpuResponse.data
    if (!form.template_id && templates.value.length) form.template_id = templates.value[0].id
  } catch (error) {
    message.error(errorMessage(error))
  } finally {
    loading.value = false
  }
}

watch(selectedTemplate, (template) => {
  if (template) form.runtime_image_id = template.default_runtime_image_id
})

async function createAndSubmit() {
  if (!form.name.trim() || !form.template_id || !form.dataset_version_id || !form.code_version_id) {
    message.warning('请完整选择训练模板、数据版本和代码版本')
    return false
  }
  saving.value = true
  const parameters = selectedTemplate.value?.key === 'yolo_detection'
    ? { epochs: form.epochs, batch: form.batch, imgsz: form.imgsz, model: form.model }
    : { arguments: form.arguments }
  try {
    const run = (
      await api.post<TrainingRun>(`/projects/${projectId.value}/runs`, {
        name: form.name,
        template_id: form.template_id,
        dataset_version_id: form.dataset_version_id,
        code_version_id: form.code_version_id,
        runtime_image_id: form.runtime_image_id,
        requested_gpu_count: form.requested_gpu_count,
        requested_gpu_model: form.requested_gpu_model,
        priority: form.priority,
        parameters,
      })
    ).data
    await api.post(`/runs/${run.id}/submit`)
    showCreate.value = false
    await router.push(`/runs/${run.id}`)
  } catch (error) {
    message.error(errorMessage(error))
    return false
  } finally {
    saving.value = false
  }
}

function statusType(status: TrainingRunStatus): 'default' | 'info' | 'warning' | 'success' | 'error' {
  if (status === 'succeeded') return 'success'
  if (status === 'failed') return 'error'
  if (status === 'running') return 'info'
  if (status === 'queued' || status === 'preparing' || status === 'stopping') return 'warning'
  return 'default'
}

function formatTime(value: string): string {
  return new Date(value).toLocaleString('zh-CN', { hour12: false })
}

onMounted(load)
</script>

<template>
  <section class="page">
    <NButton quaternary class="back-button" @click="router.push(`/projects/${projectId}`)">
      <template #icon><ArrowLeft /></template>返回项目
    </NButton>
    <header class="page-header">
      <div><h1>训练任务</h1><p>{{ project?.name }} · {{ runs.length }} 个任务</p></div>
      <div class="page-actions">
        <NButton quaternary circle title="刷新" @click="load"><RefreshCw :size="18" /></NButton>
        <NButton type="primary" @click="showCreate = true"><template #icon><Plus /></template>新建训练</NButton>
      </div>
    </header>

    <div class="gpu-strip">
      <div v-for="gpu in gpus" :key="gpu.id" :class="{ busy: gpu.allocated_run_id }">
        <Cpu :size="18" /><strong>GPU {{ gpu.index }}</strong><span>{{ gpu.model.replace('rtx_', 'RTX ') }} · {{ gpu.memory_gb }} GB</span><NTag size="small" :type="gpu.allocated_run_id ? 'warning' : 'success'">{{ gpu.allocated_run_id ? '占用' : '空闲' }}</NTag>
      </div>
    </div>

    <NSpin :show="loading">
      <div v-if="runs.length" class="data-table">
        <div class="table-row run-row table-head"><span>任务</span><span>模板</span><span>资源</span><span>状态</span><span>创建时间</span></div>
        <button v-for="run in runs" :key="run.id" class="table-row run-row run-link" type="button" @click="router.push(`/runs/${run.id}`)">
          <strong>{{ run.name }}</strong><span>{{ run.template_name }}</span><span>{{ run.requested_gpu_count }} × {{ run.requested_gpu_model }}</span><NTag size="small" :type="statusType(run.status)">{{ run.status }}</NTag><span>{{ formatTime(run.created_at) }}</span>
        </button>
      </div>
      <NEmpty v-else-if="!loading" description="暂无训练任务" />
    </NSpin>

    <NModal v-model:show="showCreate" preset="dialog" title="新建训练任务" positive-text="创建并提交" negative-text="取消" :loading="saving" class="run-dialog" @positive-click="createAndSubmit">
      <NForm :model="form" label-placement="top">
        <div class="form-grid">
          <NFormItem label="任务名称"><NInput v-model:value="form.name" placeholder="例如：YOLO 缺陷检测基线" /></NFormItem>
          <NFormItem label="训练模板"><NSelect v-model:value="form.template_id" :options="templates.map((item) => ({ label: item.name, value: item.id }))" /></NFormItem>
          <NFormItem label="数据集版本"><NSelect v-model:value="form.dataset_version_id" :options="datasetOptions" /></NFormItem>
          <NFormItem label="代码版本"><NSelect v-model:value="form.code_version_id" :options="codeOptions" /></NFormItem>
          <NFormItem label="运行镜像"><NSelect v-model:value="form.runtime_image_id" :options="runtimes.map((item) => ({ label: `${item.name} · ${item.framework}`, value: item.id }))" /></NFormItem>
          <NFormItem label="GPU 型号"><NSelect v-model:value="form.requested_gpu_model" :options="[{ label: '自动选择同型号卡', value: 'any' }, { label: 'RTX 3090', value: 'rtx_3090' }, { label: 'RTX 4090', value: 'rtx_4090' }]" /></NFormItem>
          <NFormItem label="GPU 数量"><NSelect v-model:value="form.requested_gpu_count" :options="[{ label: 'CPU', value: 0 }, { label: '1 张', value: 1 }, { label: '2 张', value: 2 }]" /></NFormItem>
          <NFormItem label="优先级"><NInputNumber v-model:value="form.priority" :min="-10" :max="10" /></NFormItem>
        </div>
        <div v-if="selectedTemplate?.key === 'yolo_detection'" class="form-grid parameters-grid">
          <NFormItem label="基础模型"><NInput v-model:value="form.model" /></NFormItem>
          <NFormItem label="训练轮数"><NInputNumber v-model:value="form.epochs" :min="1" /></NFormItem>
          <NFormItem label="批量大小"><NInputNumber v-model:value="form.batch" :min="1" /></NFormItem>
          <NFormItem label="图像尺寸"><NInputNumber v-model:value="form.imgsz" :min="32" :step="32" /></NFormItem>
        </div>
        <NFormItem v-else label="附加命令参数"><NInput v-model:value="form.arguments" placeholder="--epochs 100 --batch-size 16" /></NFormItem>
      </NForm>
    </NModal>
  </section>
</template>
