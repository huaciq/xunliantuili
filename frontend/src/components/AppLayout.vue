<script setup lang="ts">
import { Boxes, Container, Cpu, Database, FileArchive, FolderKanban, LogOut, Menu, Package, Users } from 'lucide-vue-next'
import { NButton, NDrawer, NDrawerContent } from 'naive-ui'
import { computed, onMounted, ref } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { useAuthStore } from '../stores/auth'

const auth = useAuthStore()
const route = useRoute()
const router = useRouter()
const mobileNav = ref(false)
const selectedProjectId = computed(() => {
  const value = route.params.projectId ?? route.params.id
  return typeof value === 'string' ? value : null
})
const navigation = computed(() => [
  { label: '项目', path: '/projects', icon: FolderKanban, visible: true },
  {
    label: '数据集',
    path: selectedProjectId.value ? `/projects/${selectedProjectId.value}/datasets` : '/projects',
    icon: Database,
    visible: true,
    disabled: !selectedProjectId.value,
  },
  {
    label: '代码包',
    path: selectedProjectId.value ? `/projects/${selectedProjectId.value}/code` : '/projects',
    icon: FileArchive,
    visible: true,
    disabled: !selectedProjectId.value,
  },
  {
    label: '训练任务',
    path: selectedProjectId.value ? `/projects/${selectedProjectId.value}/runs` : '/projects',
    icon: Cpu,
    visible: true,
    disabled: !selectedProjectId.value,
  },
  {
    label: '模型',
    path: selectedProjectId.value ? `/projects/${selectedProjectId.value}/models` : '/projects',
    icon: Package,
    visible: true,
    disabled: !selectedProjectId.value,
  },
  { label: '用户', path: '/users', icon: Users, visible: auth.isAdmin },
  { label: '运行镜像', path: '/runtime-images', icon: Container, visible: auth.isAdmin },
])

function go(path: string, disabled?: boolean) {
  if (disabled) return
  mobileNav.value = false
  router.push(path)
}
function logout() {
  auth.logout()
  router.replace('/login')
}
onMounted(() => auth.loadCurrentUser())
</script>

<template>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand"><Boxes :size="22" />视觉训练平台</div>
      <nav class="nav-list">
        <button v-for="item in navigation.filter((entry) => entry.visible)" :key="item.path" class="nav-item" :class="{ active: route.path.startsWith(item.path), disabled: item.disabled }" type="button" @click="go(item.path, item.disabled)">
          <component :is="item.icon" :size="18" /><span>{{ item.label }}</span><span v-if="item.disabled" class="soon">待开发</span>
        </button>
      </nav>
      <div class="sidebar-user">
        <div class="user-copy"><strong>{{ auth.user?.name ?? '加载中' }}</strong><span>{{ auth.user?.email }}</span></div>
        <NButton quaternary circle title="退出登录" @click="logout"><LogOut :size="18" /></NButton>
      </div>
    </aside>
    <header class="mobile-header">
      <NButton quaternary circle title="打开导航" @click="mobileNav = true"><Menu :size="20" /></NButton>
      <div class="brand"><Boxes :size="20" />视觉训练平台</div>
    </header>
    <main class="main-content"><RouterView /></main>
    <NDrawer v-model:show="mobileNav" placement="left" :width="280">
      <NDrawerContent title="导航" closable>
        <nav class="nav-list mobile">
          <button v-for="item in navigation.filter((entry) => entry.visible)" :key="item.path" class="nav-item" :class="{ active: route.path.startsWith(item.path), disabled: item.disabled }" type="button" @click="go(item.path, item.disabled)">
            <component :is="item.icon" :size="18" />{{ item.label }}
          </button>
        </nav>
      </NDrawerContent>
    </NDrawer>
  </div>
</template>
