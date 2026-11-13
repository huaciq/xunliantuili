<script setup lang="ts">
import { ArrowLeft, Cpu, Database, FileArchive, Package, Plus, Settings, Users } from 'lucide-vue-next'
import { NButton, NForm, NFormItem, NInput, NModal, NSelect, NSpin, NTag, useMessage } from 'naive-ui'
import { computed, onMounted, reactive, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import api, { errorMessage } from '../api'
import ModelConversionPanel from '../components/ModelConversionPanel.vue'
import { projectTypeLabels, projectTypeOptions } from '../projectTypes'
import { useAuthStore } from '../stores/auth'
import type { CodePackage, Dataset, Project, ProjectRole, ProjectType, RegisteredModel, TrainingRun } from '../types'

const route = useRoute()
const router = useRouter()
const message = useMessage()
const auth = useAuthStore()
const project = ref<Project | null>(null)
const loading = ref(false)
const saving = ref(false)
const showMember = ref(false)
const showSettings = ref(false)
const savingSettings = ref(false)
const selectedProjectType = ref<ProjectType>('general')
const datasets = ref<Dataset[]>([])
const codePackageCount = ref(0)
const runCount = ref(0)
const modelCount = ref(0)
const memberForm = reactive<{ email: string; role: ProjectRole }>({ email: '', role: 'researcher' })
const canManage = computed(() => auth.isAdmin || project.value?.current_user_role === 'project_owner')
const roleOptions = [
  { label: '项目负责人', value: 'project_owner' },
  { label: '研究员', value: 'researcher' },
  { label: '只读成员', value: 'viewer' },
]

async function load() {
  loading.value = true
  try {
    const [projectResponse, datasetsResponse, codeResponse, runsResponse, modelsResponse] = await Promise.all([
      api.get<Project>(`/projects/${route.params.id}`),
      api.get<Dataset[]>(`/projects/${route.params.id}/datasets`),
      api.get<CodePackage[]>(`/projects/${route.params.id}/code-packages`),
      api.get<TrainingRun[]>(`/projects/${route.params.id}/runs`),
      api.get<RegisteredModel[]>(`/projects/${route.params.id}/models`),
    ])
    project.value = projectResponse.data
    datasets.value = datasetsResponse.data
    codePackageCount.value = codeResponse.data.length
    runCount.value = runsResponse.data.length
    modelCount.value = modelsResponse.data.length
  }
  catch (error) { message.error(errorMessage(error)) }
  finally { loading.value = false }
}

function openSettings() {
  if (!project.value) return
  selectedProjectType.value = project.value.project_type
  showSettings.value = true
}

async function saveSettings() {
  if (!project.value || savingSettings.value) return false
  savingSettings.value = true
  try {
    const { data } = await api.patch<Project>(`/projects/${project.value.id}`, {
      project_type: selectedProjectType.value,
    })
    project.value = data
    showSettings.value = false
    message.success('项目类型已更新')
  } catch (error) {
    message.error(errorMessage(error))
    return false
  } finally {
    savingSettings.value = false
  }
}

async function addMember() {
  saving.value = true
  try {
    await api.post(`/projects/${route.params.id}/members`, memberForm)
    showMember.value = false
    memberForm.email = ''
    await load()
  } catch (error) { message.error(errorMessage(error)); return false }
  finally { saving.value = false }
}

onMounted(load)
</script>

<template>
  <section class="page">
    <NButton quaternary class="back-button" @click="router.push('/projects')"><template #icon><ArrowLeft /></template>返回项目</NButton>
    <NSpin :show="loading">
      <template v-if="project">
        <header class="page-header detail-header">
          <div><h1>{{ project.name }}</h1><p>{{ project.description || '暂无项目说明' }}</p></div>
          <div class="page-actions">
            <NTag :bordered="false">{{ projectTypeLabels[project.project_type] }}</NTag>
            <NTag :bordered="false">{{ project.current_user_role ?? '系统管理员' }}</NTag>
            <NButton v-if="canManage" @click="openSettings">
              <template #icon><Settings /></template>项目设置
            </NButton>
          </div>
        </header>
        <div class="summary-strip">
          <button type="button" @click="router.push(`/projects/${project.id}/datasets`)"><Database :size="20" /><strong>{{ datasets.length }}</strong><span>数据集</span></button>
          <button type="button" @click="router.push(`/projects/${project.id}/code`)"><FileArchive :size="20" /><strong>{{ codePackageCount }}</strong><span>代码包</span></button>
          <button type="button" @click="router.push(`/projects/${project.id}/runs`)"><Cpu :size="20" /><strong>{{ runCount }}</strong><span>训练任务</span></button>
          <button type="button" @click="router.push(`/projects/${project.id}/models`)"><Package :size="20" /><strong>{{ modelCount }}</strong><span>注册模型</span></button>
          <div><Users :size="20" /><strong>{{ project.members.length }}</strong><span>成员</span></div>
        </div>
        <ModelConversionPanel
          v-if="project.nb_conversion" :key="project.id" :project="project" :datasets="datasets"
        />
        <section class="content-section">
          <header class="section-header"><div><h2>项目成员</h2><p>成员权限应用于本项目中的全部资源。</p></div><NButton v-if="canManage" @click="showMember = true"><template #icon><Plus /></template>添加成员</NButton></header>
          <div class="data-table">
            <div class="table-row table-head"><span>成员</span><span>邮箱</span><span>角色</span></div>
            <div v-for="member in project.members" :key="member.id" class="table-row"><strong>{{ member.name }}</strong><span>{{ member.email }}</span><NTag size="small">{{ member.role }}</NTag></div>
          </div>
        </section>
      </template>
    </NSpin>
    <NModal v-model:show="showSettings" preset="dialog" title="项目设置" positive-text="保存" negative-text="取消" :loading="savingSettings" @positive-click="saveSettings">
      <NForm label-placement="top">
        <NFormItem label="项目类型"><NSelect v-model:value="selectedProjectType" :options="projectTypeOptions" /></NFormItem>
      </NForm>
    </NModal>
    <NModal v-model:show="showMember" preset="dialog" title="添加项目成员" positive-text="保存" negative-text="取消" :loading="saving" @positive-click="addMember">
      <NForm :model="memberForm" label-placement="top">
        <NFormItem label="用户邮箱"><NInput v-model:value="memberForm.email" /></NFormItem>
        <NFormItem label="项目角色"><NSelect v-model:value="memberForm.role" :options="roleOptions" /></NFormItem>
      </NForm>
    </NModal>
  </section>
</template>
