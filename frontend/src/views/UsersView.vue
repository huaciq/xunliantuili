<script setup lang="ts">
import { Plus, RefreshCw } from 'lucide-vue-next'
import { NButton, NForm, NFormItem, NInput, NModal, NSelect, NTag, useMessage } from 'naive-ui'
import { onMounted, reactive, ref } from 'vue'
import api, { errorMessage } from '../api'
import type { SystemRole, User } from '../types'

const message = useMessage()
const users = ref<User[]>([])
const showCreate = ref(false)
const saving = ref(false)
const form = reactive<{ email: string; name: string; password: string; system_role: SystemRole }>({ email: '', name: '', password: '', system_role: 'user' })

async function load() {
  try { users.value = (await api.get<User[]>('/users')).data }
  catch (error) { message.error(errorMessage(error)) }
}
async function createUser() {
  saving.value = true
  try {
    await api.post('/users', form)
    showCreate.value = false
    Object.assign(form, { email: '', name: '', password: '', system_role: 'user' })
    await load()
  } catch (error) { message.error(errorMessage(error)); return false }
  finally { saving.value = false }
}
onMounted(load)
</script>

<template>
  <section class="page">
    <header class="page-header"><div><h1>用户</h1><p>{{ users.length }} 个平台账号</p></div><div class="page-actions"><NButton quaternary circle title="刷新" @click="load"><RefreshCw :size="18" /></NButton><NButton type="primary" @click="showCreate = true"><template #icon><Plus /></template>创建用户</NButton></div></header>
    <div class="data-table">
      <div class="table-row user-table table-head"><span>姓名</span><span>邮箱</span><span>系统角色</span><span>状态</span></div>
      <div v-for="user in users" :key="user.id" class="table-row user-table"><strong>{{ user.name }}</strong><span>{{ user.email }}</span><NTag size="small">{{ user.system_role }}</NTag><span class="status-online">正常</span></div>
    </div>
    <NModal v-model:show="showCreate" preset="dialog" title="创建用户" positive-text="创建" negative-text="取消" :loading="saving" @positive-click="createUser">
      <NForm :model="form" label-placement="top"><NFormItem label="姓名"><NInput v-model:value="form.name" /></NFormItem><NFormItem label="邮箱"><NInput v-model:value="form.email" /></NFormItem><NFormItem label="初始密码"><NInput v-model:value="form.password" type="password" show-password-on="click" /></NFormItem><NFormItem label="系统角色"><NSelect v-model:value="form.system_role" :options="[{ label: '普通用户', value: 'user' }, { label: '系统管理员', value: 'system_admin' }]" /></NFormItem></NForm>
    </NModal>
  </section>
</template>

