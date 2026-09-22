<script setup lang="ts">
import { Pencil, Plus, RefreshCw } from 'lucide-vue-next'
import { NButton, NForm, NFormItem, NInput, NModal, NSwitch, NTag, useMessage } from 'naive-ui'
import { onMounted, reactive, ref } from 'vue'

import api, { errorMessage } from '../api'
import type { RuntimeImage } from '../types'

const message = useMessage()
const images = ref<RuntimeImage[]>([])
const saving = ref(false)
const showEditor = ref(false)
const editingId = ref<string | null>(null)
const form = reactive({ name: '', image: '', digest: 'development', framework: 'pytorch', is_active: true })

async function load() {
  try {
    images.value = (await api.get<RuntimeImage[]>('/admin/runtime-images')).data
  } catch (error) {
    message.error(errorMessage(error))
  }
}

function openCreate() {
  editingId.value = null
  Object.assign(form, { name: '', image: '', digest: 'development', framework: 'pytorch', is_active: true })
  showEditor.value = true
}

function openEdit(item: RuntimeImage) {
  editingId.value = item.id
  Object.assign(form, {
    name: item.name,
    image: item.image,
    digest: item.digest,
    framework: item.framework,
    is_active: item.is_active,
  })
  showEditor.value = true
}

async function save() {
  if (!form.name.trim() || !form.image.trim() || !form.framework.trim()) {
    message.warning('请完整填写镜像名称、镜像引用和框架标识')
    return false
  }
  saving.value = true
  try {
    if (editingId.value) await api.patch(`/admin/runtime-images/${editingId.value}`, form)
    else await api.post('/admin/runtime-images', form)
    showEditor.value = false
    await load()
  } catch (error) {
    message.error(errorMessage(error))
    return false
  } finally {
    saving.value = false
  }
}

async function toggle(item: RuntimeImage, active: boolean) {
  try {
    await api.patch(`/admin/runtime-images/${item.id}`, { is_active: active })
    await load()
  } catch (error) {
    message.error(errorMessage(error))
  }
}

onMounted(load)
</script>

<template>
  <section class="page">
    <header class="page-header">
      <div><h1>运行镜像</h1><p>登记服务器上已经构建或拉取完成的训练镜像</p></div>
      <div class="page-actions">
        <NButton quaternary circle title="刷新" @click="load"><RefreshCw :size="18" /></NButton>
        <NButton type="primary" @click="openCreate"><template #icon><Plus /></template>登记镜像</NButton>
      </div>
    </header>
    <div class="data-table">
      <div class="table-row runtime-table table-head"><span>名称</span><span>镜像引用</span><span>框架</span><span>状态</span><span>操作</span></div>
      <div v-for="item in images" :key="item.id" class="table-row runtime-table">
        <strong>{{ item.name }}</strong>
        <code class="truncate">{{ item.image }}</code>
        <NTag size="small">{{ item.framework }}</NTag>
        <NSwitch :value="item.is_active" @update:value="(value) => toggle(item, value)" />
        <NButton quaternary circle title="编辑" @click="openEdit(item)"><Pencil :size="16" /></NButton>
      </div>
    </div>
    <NModal v-model:show="showEditor" preset="dialog" :title="editingId ? '编辑运行镜像' : '登记运行镜像'" positive-text="保存" negative-text="取消" :loading="saving" @positive-click="save">
      <NForm :model="form" label-placement="top">
        <NFormItem label="名称"><NInput v-model:value="form.name" placeholder="例如：INP-Former CUDA 11.8" /></NFormItem>
        <NFormItem label="Docker 镜像引用"><NInput v-model:value="form.image" placeholder="train-platform/inpformer:torch2.0-cu118-v1" /></NFormItem>
        <NFormItem label="镜像摘要或版本标识"><NInput v-model:value="form.digest" placeholder="sha256:... 或 development" /></NFormItem>
        <NFormItem label="框架标识"><NInput v-model:value="form.framework" placeholder="INP-Former 请填写 inpformer" /></NFormItem>
        <NFormItem label="允许创建任务"><NSwitch v-model:value="form.is_active" /></NFormItem>
      </NForm>
    </NModal>
  </section>
</template>
