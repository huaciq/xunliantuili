<script setup lang="ts">
import { NProgress } from 'naive-ui'
import { computed } from 'vue'

import type { NbConversionProgress } from '../composables/useNbConversionProgress'

const props = defineProps<{
  progress: NbConversionProgress | null
  elapsedSeconds: number
  warning?: string
}>()

const processing = computed(() => props.progress?.status === 'pending' || props.progress?.status === 'running')
const elapsed = computed(() => {
  const minutes = Math.floor(props.elapsedSeconds / 60)
  const seconds = props.elapsedSeconds % 60
  return minutes ? `${minutes} 分 ${seconds} 秒` : `${seconds} 秒`
})
const status = computed(() => props.progress?.status === 'failed' ? 'error'
  : props.progress?.status === 'succeeded' ? 'success' : 'default')
</script>

<template>
  <div v-if="progress" class="nb-progress" role="group" aria-label="NB 转换阶段进度">
    <div class="nb-progress-heading"><strong>NB 转换阶段进度</strong><span>已用时 {{ elapsed }}</span></div>
    <NProgress type="line" :percentage="progress.percent" :processing="processing" :status="status" />
    <p role="status" aria-live="polite">{{ progress.message }}</p>
    <span v-if="processing" class="nb-progress-hint">进度随实际转换阶段更新，量化等阶段可能需要较长时间。</span>
    <span v-if="warning" class="nb-progress-warning">{{ warning }}</span>
  </div>
</template>

<style scoped>
.nb-progress { margin: 12px 0; padding: 14px 16px; background: var(--surface, #fff); border: 1px solid var(--line, #e4e7ec); border-radius: 8px; }
.nb-progress-heading { display: flex; align-items: center; justify-content: space-between; flex-wrap: wrap; gap: 8px; margin-bottom: 8px; }
.nb-progress-heading span, .nb-progress-hint { color: var(--muted); font-size: 12px; }
.nb-progress p { margin: 8px 0 0; overflow-wrap: anywhere; }
.nb-progress-hint, .nb-progress-warning { display: block; margin-top: 4px; }
.nb-progress-warning { color: #a16207; font-size: 12px; }
</style>
