<script setup lang="ts">
// 任务列表 — 显示 kind / status / progress / created_at / 操作。
import { computed } from 'vue';
import { useTaskStore } from '@/stores/task';

const task = useTaskStore();
const rows = computed(() => task.tasks);
</script>

<template>
  <div v-if="rows.length === 0" class="empty">
    <div class="empty-title">暂无任务</div>
  </div>
  <div v-else class="table-wrap">
    <table class="table">
      <thead>
        <tr>
          <th style="width: 60px;">ID</th>
          <th>Kind</th>
          <th style="width: 90px;">状态</th>
          <th style="width: 180px;">进度</th>
          <th style="width: 70px;">重试</th>
          <th style="width: 160px;">创建</th>
          <th style="width: 80px;"></th>
        </tr>
      </thead>
      <tbody>
        <tr v-for="t in rows" :key="t.id">
          <td class="font-mono muted">#{{ t.id }}</td>
          <td class="font-mono text-sm">{{ t.kind }}</td>
          <td>
            <span
              :class="[
                'tag',
                t.status === 'completed' ? 'success' :
                t.status === 'running' ? 'info' :
                t.status === 'failed' ? 'danger' :
                t.status === 'cancelled' ? 'warning' : 'accent',
              ]"
            >{{ t.status }}</span>
          </td>
          <td>
            <div v-if="t.status === 'running'" class="progress">
              <div
                class="progress-bar"
                :style="{ width: `${(task.getProgress(t.id)?.progress ?? 0) * 100}%` }"
              />
            </div>
            <span v-else-if="t.error" class="text-sm muted">{{ t.error.slice(0, 40) }}</span>
            <span v-else class="text-sm muted">—</span>
          </td>
          <td class="font-mono muted">{{ t.retry_count }}/{{ t.max_retries }}</td>
          <td class="text-sm muted font-mono">{{ t.created_at.slice(11, 19) }}</td>
          <td>
            <button
              v-if="t.status === 'pending'"
              class="btn sm ghost"
              @click="task.cancel(t.id)"
            >取消</button>
          </td>
        </tr>
      </tbody>
    </table>
  </div>
</template>

<style scoped>
.table-wrap {
  overflow: auto;
  max-height: 480px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
}
</style>
