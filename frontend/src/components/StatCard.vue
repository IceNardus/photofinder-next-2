<script setup lang="ts">
// 统计卡 — 单个 metric 显示。
defineProps<{ label: string; value: string | number; hint?: string }>();
</script>

<template>
  <div class="stat">
    <div class="stat-label">{{ label }}</div>
    <div class="stat-value">{{ value }}</div>
    <div v-if="hint" class="stat-hint">{{ hint }}</div>
  </div>
</template>
