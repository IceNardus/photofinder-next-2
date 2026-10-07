<script setup lang="ts">
// 进度条 — 0..1
defineProps<{ value: number; hint?: string }>();
</script>

<template>
  <div class="progress-wrap">
    <div class="progress">
      <div class="progress-bar" :style="{ width: `${Math.min(1, Math.max(0, value)) * 100}%` }" />
    </div>
    <div v-if="hint" class="progress-hint muted text-sm">{{ hint }}</div>
  </div>
</template>

<style scoped>
.progress-wrap {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}
.progress-hint {
  font-variant-numeric: tabular-nums;
}
</style>
