<script setup lang="ts">
import { ArrowLeft, FileArchive, Plus, RefreshCw, Upload } from 'lucide-vue-next'
import {
  NButton,
  NEmpty,
  NForm,
  NFormItem,
  NInput,
  NModal,
  NSpin,
  NTag,
  useMessage,
} from 'naive-ui'
import { computed, onMounted, reactive, ref, watch } from 'vue'
import { useRoute, useRouter } from 'vue-router'

import api, { errorMessage } from '../api'
import type { CodePackage, Dataset, Project, ResourceVersionStatus } from '../types'

const props = defineProps<{ kind: 'datasets' | 'code' }>()
const route = useRoute()
const router = useRouter()
const message = useMessage()
const project = ref<Project | null>(null)
const datasets = ref<Dataset[]>([])
const codePackages = ref<CodePackage[]>([])
const loading = ref(false)
const saving = ref(false)
const uploading = ref(false)
const showCreate = ref(false)
const showUpload = ref(false)
const selectedResource = ref<Dataset | CodePackage | null>(null)
const selectedFile = ref<File | null>(null)
const fileInput = ref<HTMLInputElement | null>(null)
const createForm = reactive({ name: '', description: '' })
const codeForm = reactive({ default_workdir: '.', default_entrypoint: '' })

const projectId = computed(() => String(route.params.projectId))
const isDataset = computed(() => props.kind === 'datasets')
const title = computed(() => (isDataset.value ? '数据集' : '代码包'))
const resources = computed(() => (isDataset.value ? datasets.value : codePackages.value))

async function load() {
  loading.value = true
  try {
    project.value = (await api.get<Project>(`/projects/${projectId.value}`)).data
    if (isDataset.value) {
      datasets.value = (await api.get<Dataset[]>(`/projects/${projectId.value}/datasets`)).data
    } else {
      codePackages.value = (
        await api.get<CodePackage[]>(`/projects/${projectId.value}/code-packages`)
      ).data
    }
  } catch (error) {
    message.error(errorMessage(error))
  } finally {
    loading.value = false
  }
}

async function createResource() {
  if (createForm.name.trim().length < 2) {
    message.warning(`请输入${title.value}名称`)
    return false
  }
  saving.value = true
  const endpoint = isDataset.value
    ? `/projects/${projectId.value}/datasets`
    : `/projects/${projectId.value}/code-packages`
  try {
    await api.post(endpoint, createForm)
    Object.assign(createForm, { name: '', description: '' })
    showCreate.value = false
    await load()
  } catch (error) {
    message.error(errorMessage(error))
    return false
  } finally {
    saving.value = false
  }
}

function openUpload(resource: Dataset | CodePackage) {
  selectedResource.value = resource
  selectedFile.value = null
  Object.assign(codeForm, { default_workdir: '.', default_entrypoint: '' })
  showUpload.value = true
}

function chooseFile() {
  fileInput.value?.click()
}

function onFileChange(event: Event) {
  selectedFile.value = (event.target as HTMLInputElement).files?.[0] ?? null
}

async function uploadVersion() {
  if (!selectedResource.value || !selectedFile.value) {
    message.warning('请选择 ZIP、TAR、TAR.GZ 或 TGZ 文件')
    return false
  }
  uploading.value = true
  const data = new FormData()
  data.append('file', selectedFile.value)
  if (!isDataset.value) {
    data.append('default_workdir', codeForm.default_workdir)
    data.append('default_entrypoint', codeForm.default_entrypoint)
  }
  const endpoint = isDataset.value
    ? `/datasets/${selectedResource.value.id}/versions/upload`
    : `/code-packages/${selectedResource.value.id}/versions/upload`
  try {
    await api.post(endpoint, data, { timeout: 0 })
    showUpload.value = false
    message.success('版本上传并校验完成')
    await load()
  } catch (error) {
    message.error(errorMessage(error))
    return false
  } finally {
    uploading.value = false
  }
}

function formatBytes(value: number): string {
  if (!value) return '0 B'
  const units = ['B', 'KB', 'MB', 'GB', 'TB']
  const index = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1)
  return `${(value / 1024 ** index).toFixed(index ? 1 : 0)} ${units[index]}`
}

function statusType(value: ResourceVersionStatus): 'success' | 'warning' | 'error' {
  if (value === 'ready') return 'success'
  if (value === 'processing') return 'warning'
  return 'error'
}

watch(() => props.kind, load)
onMounted(load)
</script>

<template>
  <section class="page">
    <NButton quaternary class="back-button" @click="router.push(`/projects/${projectId}`)">
      <template #icon><ArrowLeft /></template>返回项目
    </NButton>
    <header class="page-header">
      <div><h1>{{ title }}</h1><p>{{ project?.name }} · {{ resources.length }} 项资源</p></div>
      <div class="page-actions">
        <NButton quaternary circle title="刷新" @click="load"><RefreshCw :size="18" /></NButton>
        <NButton type="primary" @click="showCreate = true"><template #icon><Plus /></template>新建{{ title }}</NButton>
      </div>
    </header>

    <NSpin :show="loading">
      <div v-if="resources.length" class="resource-list">
        <article v-for="resource in resources" :key="resource.id" class="resource-panel">
          <header class="resource-header">
            <div class="resource-name">
              <div class="project-icon"><FileArchive :size="20" /></div>
              <div><h2>{{ resource.name }}</h2><p>{{ resource.description || '暂无说明' }}</p></div>
            </div>
            <NButton size="small" @click="openUpload(resource)"><template #icon><Upload /></template>上传新版本</NButton>
          </header>
          <div v-if="resource.versions.length" class="version-table">
            <div class="version-row version-head"><span>版本</span><span>源文件</span><span>状态 / 格式</span><span>文件</span><span>解压大小</span><span>SHA-256</span></div>
            <div v-for="version in resource.versions" :key="version.id" class="version-row">
              <strong>v{{ version.version }}</strong>
              <span class="truncate" :title="version.source_filename">{{ version.source_filename }}</span>
              <span class="version-state"><NTag size="small" :type="statusType(version.status)">{{ version.status }}</NTag><NTag v-if="isDataset" size="small" :bordered="false">{{ 'format' in version ? version.format : '' }}</NTag></span>
              <span>{{ version.file_count }}</span>
              <span>{{ formatBytes(version.extracted_size) }}</span>
              <code :title="version.sha256">{{ version.sha256.slice(0, 10) }}</code>
              <p v-if="version.error_message" class="version-error">{{ version.error_message }}</p>
            </div>
          </div>
          <div v-else class="empty-inline">尚未上传版本</div>
        </article>
      </div>
      <NEmpty v-else-if="!loading" :description="`暂无${title}`" />
    </NSpin>

    <NModal v-model:show="showCreate" preset="dialog" :title="`新建${title}`" positive-text="创建" negative-text="取消" :loading="saving" @positive-click="createResource">
      <NForm :model="createForm" label-placement="top">
        <NFormItem label="名称"><NInput v-model:value="createForm.name" maxlength="120" /></NFormItem>
        <NFormItem label="说明"><NInput v-model:value="createForm.description" type="textarea" maxlength="2000" :rows="3" /></NFormItem>
      </NForm>
    </NModal>

    <NModal v-model:show="showUpload" preset="dialog" :title="`上传 ${selectedResource?.name} 的新版本`" positive-text="上传并处理" negative-text="取消" :loading="uploading" @positive-click="uploadVersion">
      <div class="upload-picker" role="button" tabindex="0" @click="chooseFile" @keydown.enter="chooseFile">
        <Upload :size="24" /><strong>{{ selectedFile?.name ?? '选择压缩包' }}</strong><span>ZIP、TAR、TAR.GZ、TGZ，最大 5 GB</span>
        <input ref="fileInput" type="file" accept=".zip,.tar,.tar.gz,.tgz" hidden @change="onFileChange" />
      </div>
      <NForm v-if="!isDataset" :model="codeForm" label-placement="top" class="upload-code-form">
        <NFormItem label="默认工作目录"><NInput v-model:value="codeForm.default_workdir" placeholder="." /></NFormItem>
        <NFormItem label="默认入口命令"><NInput v-model:value="codeForm.default_entrypoint" placeholder="python train.py" /></NFormItem>
      </NForm>
    </NModal>
  </section>
</template>

