<script setup lang="ts">
import { ArrowRight, FolderKanban, Plus, RefreshCw } from 'lucide-vue-next'
import { NButton, NEmpty, NForm, NFormItem, NInput, NModal, NSpin, useMessage } from 'naive-ui'
import { onMounted, reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import api, { errorMessage } from '../api'
import type { Project } from '../types'

const router = useRouter()
const message = useMessage()
const projects = ref<Project[]>([])
const loading = ref(false)
const saving = ref(false)
const showCreate = ref(false)
const form = reactive({ name: '', description: '' })

async function load() {
  loading.value = true
  try { projects.value = (await api.get<Project[]>('/projects')).data }
  catch (error) { message.error(errorMessage(error)) }
  finally { loading.value = false }
}

async function createProject() {
  if (form.name.trim().length < 2) { message.warning('请输入项目名称'); return false }
  saving.value = true
  try {
    const project = (await api.post<Project>('/projects', form)).data
    showCreate.value = false
    Object.assign(form, { name: '', description: '' })
    await router.push(`/projects/${project.id}`)
  } catch (error) { message.error(errorMessage(error)); return false }
  finally { saving.value = false }
}

onMounted(load)
</script>

<template>
  <section class="page">
    <header class="page-header">
      <div><h1>项目</h1><p>{{ projects.length }} 个可访问项目</p></div>
      <div class="page-actions">
        <NButton quaternary circle title="刷新" @click="load"><RefreshCw :size="18" /></NButton>
        <NButton type="primary" @click="showCreate = true"><template #icon><Plus /></template>新建项目</NButton>
      </div>
    </header>
    <NSpin :show="loading">
      <div v-if="projects.length" class="project-grid">
        <button v-for="project in projects" :key="project.id" class="project-card" type="button" @click="router.push(`/projects/${project.id}`)">
          <div class="project-icon"><FolderKanban :size="21" /></div>
          <div class="project-body">
            <div class="project-title"><h2>{{ project.name }}</h2><ArrowRight :size="18" /></div>
            <p>{{ project.description || '暂无项目说明' }}</p>
            <div class="project-meta"><span>{{ project.members.length }} 位成员</span><span>{{ project.current_user_role ?? '管理员' }}</span></div>
          </div>
        </button>
      </div>
      <NEmpty v-else-if="!loading" description="暂无项目" />
    </NSpin>
    <NModal v-model:show="showCreate" preset="dialog" title="新建项目" positive-text="创建" negative-text="取消" :loading="saving" @positive-click="createProject">
      <NForm :model="form" label-placement="top">
        <NFormItem label="项目名称"><NInput v-model:value="form.name" maxlength="120" /></NFormItem>
        <NFormItem label="项目说明"><NInput v-model:value="form.description" type="textarea" maxlength="2000" :rows="3" /></NFormItem>
      </NForm>
    </NModal>
  </section>
</template>

