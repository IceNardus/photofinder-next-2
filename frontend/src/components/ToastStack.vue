<script setup lang="ts">
// 全局 toast 渲染（App.vue 末尾挂载一次）。
import { useToastStore } from '@/stores/toast';
const toast = useToastStore();
</script>

<template>
  <div v-if="toast.items.length" class="toast-stack">
    <div
      v-for="t in toast.items"
      :key="t.id"
      :class="['toast', t.kind]"
    >
      <div class="toast-msg">{{ t.message }}</div>
      <button class="toast-close" @click="toast.dismiss(t.id)">×</button>
    </div>
  </div>
</template>
