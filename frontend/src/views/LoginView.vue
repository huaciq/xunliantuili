<script setup lang="ts">
import { Boxes, LogIn } from 'lucide-vue-next'
import { NButton, NForm, NFormItem, NInput, useMessage } from 'naive-ui'
import { reactive, ref } from 'vue'
import { useRouter } from 'vue-router'
import { errorMessage } from '../api'
import { useAuthStore } from '../stores/auth'

const auth = useAuthStore()
const router = useRouter()
const message = useMessage()
const submitting = ref(false)
const form = reactive({ email: 'admin@example.com', password: 'ChangeMe123!' })

async function submit() {
  submitting.value = true
  try {
    await auth.login(form.email, form.password)
    await router.replace('/projects')
  } catch (error) {
    message.error(errorMessage(error))
  } finally {
    submitting.value = false
  }
}
</script>

<template>
  <main class="login-page">
    <section class="login-panel">
      <div class="login-heading">
        <div class="login-mark"><Boxes :size="25" /></div>
        <div><h1>视觉训练平台</h1><p>实验室训练资源控制台</p></div>
      </div>
      <NForm :model="form" size="large" @submit.prevent="submit">
        <NFormItem label="邮箱"><NInput v-model:value="form.email" autocomplete="username" /></NFormItem>
        <NFormItem label="密码"><NInput v-model:value="form.password" type="password" show-password-on="click" autocomplete="current-password" /></NFormItem>
        <NButton type="primary" block attr-type="submit" :loading="submitting"><template #icon><LogIn /></template>登录</NButton>
      </NForm>
    </section>
  </main>
</template>

