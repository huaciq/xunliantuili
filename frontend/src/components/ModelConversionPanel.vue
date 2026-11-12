<script setup lang="ts">
import { isAxiosError } from 'axios'
import { Download, FileOutput, Upload } from 'lucide-vue-next'
import { NAlert, NButton } from 'naive-ui'
import { computed, ref, shallowRef } from 'vue'

import api, { errorMessage } from '../api'
import { useAuthStore } from '../stores/auth'
import type { Project } from '../types'

const props = defineProps<{ project: Project }>()
const auth = useAuthStore()
const canConvert = computed(() => auth.isAdmin || props.project.current_user_role !== 'viewer')
const fileInput = ref<HTMLInputElement | null>(null)
const selectedFile = shallowRef<File | null>(null)
const result = shallowRef<{ file: Blob; name: string } | null>(null)
const converting = ref(false)
const failure = ref('')

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
  if (!file || converting.value) return
  converting.value = true
  failure.value = ''
  result.value = null
  const data = new FormData()
  data.append('file', file)
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
    <template v-else>
      <div class="conversion-actions">
        <input ref="fileInput" type="file" accept=".onnx" hidden @change="selectFile" />
        <NButton :disabled="converting" @click="fileInput?.click()">
          <template #icon><Upload /></template>选择 ONNX 文件
        </NButton>
        <NButton type="primary" :disabled="!selectedFile" :loading="converting" @click="convert">
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
.conversion-actions { display: flex; align-items: center; flex-wrap: wrap; gap: 12px; }
.conversion-filename { color: var(--muted); overflow-wrap: anywhere; }
.conversion-feedback { margin-top: 12px; }
</style>
