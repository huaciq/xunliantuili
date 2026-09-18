import { defineStore } from 'pinia'
import { computed, ref } from 'vue'
import api from '../api'
import type { User } from '../types'

export const useAuthStore = defineStore('auth', () => {
  const user = ref<User | null>(null)
  const loading = ref(false)
  const authenticated = computed(() => Boolean(localStorage.getItem('access_token')))
  const isAdmin = computed(() => user.value?.system_role === 'system_admin')

  async function login(email: string, password: string) {
    const response = await api.post<{ access_token: string }>('/auth/login', { email, password })
    localStorage.setItem('access_token', response.data.access_token)
    await loadCurrentUser()
  }

  async function loadCurrentUser() {
    if (!authenticated.value) return
    loading.value = true
    try {
      user.value = (await api.get<User>('/auth/me')).data
    } finally {
      loading.value = false
    }
  }

  function logout() {
    localStorage.removeItem('access_token')
    user.value = null
  }

  return { user, loading, authenticated, isAdmin, login, loadCurrentUser, logout }
})

