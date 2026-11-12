<script setup lang="ts">
import { isAxiosError } from 'axios'
import { Download, FileOutput, Upload } from 'lucide-vue-next'
import { NAlert, NButton, NFormItem, NSelect } from 'naive-ui'
import { computed, ref, shallowRef, watch } from 'vue'

import api, { errorMessage } from '../api'
import { useAuthStore } from '../stores/auth'
import type { Dataset, Project } from '../types'

const props = defineProps<{ project: Project; datasets: Dataset[] }>()
const auth = useAuthStore()
const canConvert = computed(() => auth.isAdmin || props.project.current_user_role !== 'viewer')
const datasetOptions = computed(() => props.datasets.flatMap((dataset) =>
  dataset.versions.filter((version) => version.status === 'ready').map((version) => ({
    label: `${dataset.name} v${version.version}`,
    value: version.id,
  })),
))
const datasetVersionId = ref<string | null>(null)
const fileInput = ref<HTMLInputElement | null>(null)
const selectedFile = shallowRef<File | null>(null)
const result = shallowRef<{ file: Blob; name: string } | null>(null)
const converting = ref(false)
const failure = ref('')

watch(datasetOptions, (options) => {
  if (options.some((option) => option.value === datasetVersionId.value)) return
  datasetVersionId.value = options.length === 1 ? options[0]!.value : null
}, { immediate: true })

watch(datasetVersionId, () => {
  result.value = null
  failure.value = ''
})

function selectFile(event: Event) {
  const input = event.target as HTMLInputElement
  const file = input.files?.[0]
  input.value = ''
  if (!file) return
  result.value = null
  selectedFile.value = null
  failure.value = ''
  if (!/\.onnx$/i.test(file.name) || !file.size) {
    failure.value = '请选择非空的 .onnx 文件。'
    return
  }
  selectedFile.value = file
}

async function conversionError(error: unknown) {
  if (isAxiosError(error) && error.response?.data instanceof Blob) {
    const body = await error.response.data.text()
    try {
      const payload = JSON.parse(body)
      if (typeof payload.detail === 'string') return payload.detail
    } catch {
      // A reverse proxy may return an HTML error page.
    }
    return error.response.status === 413 ? 'ONNX 文件超过上传大小限制。' : error.message
  }
  return errorMessage(error)
}

async function convert() {
  const file = selectedFile.value
  if (!file || !datasetVersionId.value || converting.value) return
  converting.value = true
  failure.value = ''
  result.value = null
  const data = new FormData()
  data.append('file', file)
  data.append('dataset_version_id', datasetVersionId.value)
  try {
    const response = await api.post<Blob>(
      `/projects/${props.project.id}/conversions/onnx-to-nb`, data,
      { responseType: 'blob', timeout: 0 },
    )
    const stem = Array.from(file.name.slice(0, -5)).slice(0, 200).join('')
    result.value = { file: response.data, name: `${stem}.nb` }
  } catch (error) {
    failure.value = await conversionError(error)
  } finally {
    converting.value = false
  }
}

function download() {
  if (!result.value) return
  const url = URL.createObjectURL(result.value.file)
  const link = document.createElement('a')
  link.href = url
  link.download = result.value.name
  link.click()
  window.setTimeout(() => URL.revokeObjectURL(url), 1000)
}
</script>

<template>
  <section class="content-section">
    <header class="section-header">
      <div><h2>模型转换</h2><p>ONNX → NB</p></div>
    </header>
    <NAlert v-if="!canConvert" type="info">当前账号没有模型转换权限。</NAlert>
    <NAlert v-else-if="project.nb_conversion === 'unconfigured'" type="warning">
      服务器尚未配置 NB 转换器。
    </NAlert>
    <NAlert v-else-if="!datasetOptions.length" type="info">
      请先在项目数据集中上传模型对应的点位数据集，供 INT8 校准使用。
    </NAlert>
    <template v-else>
      <NFormItem label="校准数据集版本" class="calibration-field">
        <NSelect
          v-model:value="datasetVersionId" :options="datasetOptions" :disabled="converting"
          placeholder="选择模型对应的数据集版本"
        />
      </NFormItem>
      <div class="conversion-actions">
        <input ref="fileInput" type="file" accept=".onnx" hidden @change="selectFile" />
        <NButton :disabled="converting" @click="fileInput?.click()">
          <template #icon><Upload /></template>选择 ONNX 文件
        </NButton>
        <NButton
          type="primary" :disabled="!selectedFile || !datasetVersionId"
          :loading="converting" @click="convert"
        >
          <template #icon><FileOutput /></template>{{ converting ? '转换中' : '开始转换' }}
        </NButton>
        <span v-if="selectedFile" class="conversion-filename">{{ selectedFile.name }}</span>
      </div>
      <NAlert v-if="failure" type="error" class="conversion-feedback">{{ failure }}</NAlert>
      <div v-if="result" class="conversion-actions conversion-feedback" role="status">
        <NButton type="success" secondary @click="download">
          <template #icon><Download /></template>下载 NB
        </NButton>
        <span class="conversion-filename">{{ result.name }}</span>
      </div>
    </template>
  </section>
</template>

<style scoped>
.calibration-field { max-width: 440px; }
.conversion-actions { display: flex; align-items: center; flex-wrap: wrap; gap: 12px; }
.conversion-filename { color: var(--muted); overflow-wrap: anywhere; }
.conversion-feedback { margin-top: 12px; }
</style>
