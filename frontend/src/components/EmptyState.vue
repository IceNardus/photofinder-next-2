<script setup lang="ts">
// 空态
defineProps<{ title: string; hint?: string }>();
</script>

<template>
  <div class="empty">
    <div class="empty-title">{{ title }}</div>
    <div v-if="hint" class="muted text-sm">{{ hint }}</div>
  </div>
</template>
