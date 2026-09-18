<script setup lang="ts">
import { ArrowLeft, Box, Download, FileOutput, RefreshCw } from 'lucide-vue-next'
import { NButton, NEmpty, NSpin, NTag, useMessage } from 'naive-ui'
import { onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import api, { errorMessage } from '../api'
import type { ExportJob, RegisteredModel } from '../types'

const route = useRoute()
const router = useRouter()
const message = useMessage()
const models = ref<RegisteredModel[]>([])
const loading = ref(false)
const exportingId = ref<string | null>(null)

async function load() {
  loading.value = true
  try {
    models.value = (await api.get<RegisteredModel[]>(`/projects/${route.params.projectId}/models`)).data
  } catch (error) { message.error(errorMessage(error)) }
  finally { loading.value = false }
}

async function createExport(versionId: string) {
  exportingId.value = versionId
  try {
    await api.post(`/model-versions/${versionId}/exports`, { format: 'onnx' })
    message.success('ONNX 导出完成，一致性检查已通过')
    await load()
  } catch (error) { message.error(errorMessage(error)) }
  finally { exportingId.value = null }
}

async function download(job: ExportJob) {
  if (!job.artifact_id) return
  try {
    const response = await api.get(`/artifacts/${job.artifact_id}/download`, { responseType: 'blob' })
    const url = URL.createObjectURL(response.data)
    const anchor = document.createElement('a')
    anchor.href = url
    anchor.download = 'model.onnx'
    anchor.click()
    URL.revokeObjectURL(url)
  } catch (error) { message.error(errorMessage(error)) }
}

function metric(version: RegisteredModel['versions'][number], key: string) {
  const value = version.metrics[key]
  return value === undefined ? '-' : value.toFixed(4)
}

onMounted(load)
</script>

<template>
  <section class="page">
    <NButton quaternary class="back-button" @click="router.push(`/projects/${route.params.projectId}`)">
      <template #icon><ArrowLeft /></template>返回项目
    </NButton>
    <header class="page-header detail-header">
      <div><h1>模型注册表</h1><p>管理训练 checkpoint、模型版本和可交付导出文件。</p></div>
      <NButton quaternary circle title="刷新" @click="load"><RefreshCw :size="18" /></NButton>
    </header>
    <NSpin :show="loading">
      <NEmpty v-if="!models.length && !loading" description="尚未注册模型，请从成功的训练任务注册" />
      <div v-else class="model-list">
        <section v-for="model in models" :key="model.id" class="model-panel">
          <header>
            <div class="model-title"><span><Box :size="19" /></span><div><h2>{{ model.name }}</h2><p>{{ model.description || '暂无说明' }}</p></div></div>
            <NTag size="small">{{ model.versions.length }} 个版本</NTag>
          </header>
          <div class="model-version-table">
            <div class="model-version-row model-version-head"><span>版本</span><span>来源任务</span><span>阶段</span><span>mAP50</span><span>Precision</span><span>导出</span><span>操作</span></div>
            <div v-for="version in [...model.versions].reverse()" :key="version.id" class="model-version-row">
              <strong>v{{ version.version }}</strong>
              <button class="text-link" type="button" @click="router.push(`/runs/${version.source_run_id}`)">{{ version.source_run_name }}</button>
              <NTag size="small" :type="version.stage === 'production' ? 'success' : 'default'">{{ version.stage }}</NTag>
              <code>{{ metric(version, 'val/map50') }}</code>
              <code>{{ metric(version, 'val/precision') }}</code>
              <span v-if="version.exports.length" class="export-state"><FileOutput :size="15" />{{ version.exports.at(-1)?.validation_status }}</span>
              <span v-else class="muted">未导出</span>
              <div class="row-actions">
                <NButton v-if="!version.exports.length" size="small" :loading="exportingId === version.id" @click="createExport(version.id)">导出 ONNX</NButton>
                <NButton v-else size="small" secondary @click="download(version.exports.at(-1)!)"><template #icon><Download /></template>下载</NButton>
              </div>
            </div>
          </div>
        </section>
      </div>
    </NSpin>
  </section>
</template>
