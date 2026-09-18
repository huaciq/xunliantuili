<script setup lang="ts">
import { NConfigProvider, NDialogProvider, NMessageProvider, dateZhCN, zhCN } from 'naive-ui'
</script>

<template>
  <NConfigProvider :locale="zhCN" :date-locale="dateZhCN">
    <NDialogProvider><NMessageProvider><RouterView /></NMessageProvider></NDialogProvider>
  </NConfigProvider>
</template>

