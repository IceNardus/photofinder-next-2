<script setup lang="ts">
// ScanLog — 滚动日志面板（最多 100 条，按时间倒序；新条目自动滚到底部）。
//
// 设计：
// - 颜色编码：info=neutral, success=green, warn=yellow, error=red
// - 自动滚动：仅当用户已经在底部时跟随；用户向上滚动查看历史时暂停跟随
// - 时间戳紧凑格式 HH:MM:SS

import { computed, nextTick, onUnmounted, ref, watch } from 'vue';
import { useScanStore } from '@/stores/scan';

const scan = useScanStore();

const container = ref<HTMLElement | null>(null);
const stickToBottom = ref(true);

function timeLabel(ts: number): string {
  const d = new Date(ts);
  const pad = (n: number) => String(n).padStart(2, '0');
  return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
}

const rows = computed(() => scan.logEntries);

function onScroll() {
  const el = container.value;
  if (!el) return;
  const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
  stickToBottom.value = distance < 16;
}

async function scrollToBottom() {
  await nextTick();
  const el = container.value;
  if (!el) return;
  el.scrollTop = el.scrollHeight;
}

watch(rows, async () => {
  if (stickToBottom.value) {
    void scrollToBottom();
  }
});

onUnmounted(() => {
  // store cleans up its own timer; nothing to do here
});
</script>

<template>
  <div class="scan-log">
    <div class="scan-log-head">
      <div class="scan-log-title">扫描日志</div>
      <div class="row gap-2">
        <span class="muted text-sm font-mono">{{ rows.length }} 条</span>
        <button
          v-if="rows.length"
          class="btn sm ghost"
          @click="scan.clearLog()"
        >清空</button>
      </div>
    </div>

    <div
      v-if="rows.length === 0"
      class="scan-log-empty muted text-sm"
    >暂无日志</div>

    <div
      v-else
      ref="container"
      class="scan-log-body"
      @scroll="onScroll"
    >
      <div
        v-for="(entry, idx) in rows"
        :key="`${entry.ts}-${idx}`"
        :class="['scan-log-row', `level-${entry.level}`]"
      >
        <span class="scan-log-ts font-mono">{{ timeLabel(entry.ts) }}</span>
        <span :class="['scan-log-tag', `tag-${entry.level}`]">{{ levelLabel(entry.level) }}</span>
        <span class="scan-log-msg">{{ entry.message }}</span>
      </div>
    </div>
  </div>
</template>

<script lang="ts">
// helper outside <script setup> so template can call it as a method
function levelLabel(level: 'info' | 'success' | 'warn' | 'error'): string {
  switch (level) {
    case 'info':    return 'INFO';
    case 'success': return 'OK';
    case 'warn':    return 'WARN';
    case 'error':   return 'ERR';
  }
}
export { levelLabel };
</script>

<style scoped>
.scan-log {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}
.scan-log-head {
  display: flex;
  align-items: center;
  justify-content: space-between;
}
.scan-log-title {
  font-size: var(--text-md);
  font-weight: 600;
}
.scan-log-body {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  max-height: 320px;
  overflow-y: auto;
  padding: var(--space-2);
  background: var(--color-bg-2);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  font-size: var(--text-sm);
  line-height: var(--leading-normal);
}
.scan-log-empty {
  padding: var(--space-3);
  background: var(--color-bg-2);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  text-align: center;
}
.scan-log-row {
  display: grid;
  grid-template-columns: 60px 36px 1fr;
  gap: var(--space-2);
  align-items: baseline;
  padding: 2px var(--space-2);
  border-radius: var(--radius-sm);
}
.scan-log-row:hover { background: var(--color-bg-3); }
.scan-log-ts {
  color: var(--color-fg-2);
  font-size: var(--text-xs);
}
.scan-log-tag {
  font-family: var(--font-mono);
  font-size: var(--text-xs);
  font-weight: 600;
  text-align: center;
  border-radius: var(--radius-sm);
  padding: 0 4px;
}
.tag-info    { background: rgba(90, 200, 250, 0.15); color: var(--color-info); }
.tag-success { background: rgba(52, 199, 89, 0.18); color: var(--color-success); }
.tag-warn    { background: rgba(255, 159, 10, 0.18); color: var(--color-warning); }
.tag-error   { background: rgba(255, 69, 58, 0.18); color: var(--color-danger); }
.scan-log-msg {
  word-break: break-word;
  color: var(--color-fg-1);
}
.level-error   .scan-log-msg { color: var(--color-danger); }
.level-warn    .scan-log-msg { color: var(--color-warning); }
.level-success .scan-log-msg { color: var(--color-success); }
</style>