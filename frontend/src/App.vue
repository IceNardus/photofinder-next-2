<script setup lang="ts">
// App.vue — sidebar + main content + toast stack。
//
// 全局：
// - app.refresh() 在挂载时拉一次（info/stats/models）
// - 每 5s 轮询 stats（不刷 info/models — 这两个不变）
// - scan/task store 的 listener 由各 view 自管

import { onMounted, onUnmounted, watch } from 'vue';
import { RouterLink, RouterView, useRoute } from 'vue-router';
import { useAppStore } from '@/stores/app';
import { useSettingsStore } from '@/stores/settings';
import { navItems } from '@/router';
import ToastStack from '@/components/ToastStack.vue';

const app = useAppStore();
const settings = useSettingsStore();
const route = useRoute();

let statsTimer: ReturnType<typeof setInterval> | null = null;

function startStatsPolling(intervalMs: number) {
  stopStatsPolling();
  // 默认 5s 轮询（可被用户在 Settings 调整）。
  // 主动 scan 时由 ScanView 触发额外刷新；这里是后台节奏。
  statsTimer = setInterval(() => {
    void app.refreshStats();
  }, Math.max(500, intervalMs));
}

function stopStatsPolling() {
  if (statsTimer) {
    clearInterval(statsTimer);
    statsTimer = null;
  }
}

onMounted(() => {
  void app.refresh();
  startStatsPolling(settings.settings.sidebarPollMs);
});

// 用户改了 polling 间隔 → 立即重启定时器
watch(
  () => settings.settings.sidebarPollMs,
  (ms) => startStatsPolling(ms),
);

onUnmounted(() => {
  stopStatsPolling();
});
</script>

<template>
  <div class="layout">
    <!-- ============ Sidebar ============ -->
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-logo">PF</div>
        <div class="brand-text">
          <div class="brand-name">PhotoFinder</div>
          <div class="brand-sub muted text-sm">Next 2</div>
        </div>
      </div>

      <nav class="nav">
        <RouterLink
          v-for="item in navItems"
          :key="item.name"
          :to="item.path"
          :class="['nav-item', { active: route.path === item.path }]"
        >
          <span class="nav-icon" :data-icon="item.icon" />
          <span class="nav-label">{{ item.title }}</span>
        </RouterLink>
      </nav>

      <!-- Sidebar 底部实时统计（每 5s 刷新） -->
      <div v-if="app.stats" class="sidebar-stats">
        <div class="sidebar-stats-title muted text-sm">总览</div>
        <div class="sidebar-stat">
          <span class="sidebar-stat-label">图片</span>
          <span class="sidebar-stat-value font-mono">{{ app.stats.image_count }}</span>
        </div>
        <div class="sidebar-stat">
          <span class="sidebar-stat-label">人脸</span>
          <span class="sidebar-stat-value font-mono">{{ app.stats.face_count }}</span>
        </div>
        <div class="sidebar-stat">
          <span class="sidebar-stat-label">对象</span>
          <span class="sidebar-stat-value font-mono">{{ app.stats.object_count }}</span>
        </div>
        <div class="sidebar-stat">
          <span class="sidebar-stat-label">人物</span>
          <span class="sidebar-stat-value font-mono">{{ app.stats.person_count }}</span>
        </div>
        <div v-if="app.stats.pending_task_count > 0" class="sidebar-stat sidebar-stat-busy">
          <span class="sidebar-stat-label">待处理任务</span>
          <span class="sidebar-stat-value font-mono">
            {{ app.stats.pending_task_count }}
          </span>
        </div>
      </div>

      <div class="sidebar-foot">
        <div v-if="app.info" class="text-sm muted font-mono">v{{ app.info.version }} · {{ app.info.platform }}</div>
      </div>
    </aside>

    <!-- ============ Main ============ -->
    <main class="main">
      <RouterView />
    </main>

    <ToastStack />
  </div>
</template>

<style scoped>
.layout {
  display: grid;
  grid-template-columns: var(--sidebar-width) 1fr;
  height: 100vh;
  overflow: hidden;
}
/* 移动端（窄屏）折叠 sidebar 为底部 tab bar */
@media (max-width: 640px) {
  .layout {
    grid-template-columns: 1fr;
    grid-template-rows: 1fr auto;
  }
  .sidebar {
    flex-direction: row;
    border-right: none;
    border-top: 1px solid var(--color-border);
    padding: 0;
    order: 2;
    height: auto;
  }
  .brand {
    display: none;
  }
  .nav {
    flex-direction: row;
    padding: 0;
    width: 100%;
    justify-content: space-around;
    gap: 0;
  }
  .nav-item {
    flex-direction: column;
    gap: var(--space-1);
    padding: var(--space-2) var(--space-3);
    font-size: var(--text-xs);
    border-radius: 0;
  }
  .nav-item.active {
    background: transparent;
    color: var(--color-accent);
    border-top: 2px solid var(--color-accent);
  }
  .sidebar-stats { display: none; }
  .sidebar-foot { display: none; }
  .main { order: 1; }
}
.sidebar {
  display: flex;
  flex-direction: column;
  background: var(--color-bg-1);
  border-right: 1px solid var(--color-border);
  padding: var(--space-4) 0;
}
.brand {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  padding: 0 var(--space-4) var(--space-5);
  border-bottom: 1px solid var(--color-border);
}
.brand-logo {
  width: 32px;
  height: 32px;
  display: flex;
  align-items: center;
  justify-content: center;
  background: var(--color-accent);
  color: white;
  border-radius: var(--radius-md);
  font-weight: 600;
  font-size: var(--text-sm);
}
.brand-name { font-weight: 600; font-size: var(--text-md); }
.brand-sub { font-size: var(--text-xs); }

.nav {
  display: flex;
  flex-direction: column;
  padding: var(--space-3) var(--space-2);
  gap: var(--space-1);
  flex: 1;
  overflow-y: auto;
}
.nav-item {
  display: flex;
  align-items: center;
  gap: var(--space-3);
  padding: var(--space-2) var(--space-3);
  border-radius: var(--radius-md);
  color: var(--color-fg-1);
  font-size: var(--text-md);
  font-weight: 500;
  text-decoration: none;
  transition: background 120ms, color 120ms;
}
.nav-item:hover { background: var(--color-bg-2); color: var(--color-fg-0); text-decoration: none; }
.nav-item.active {
  background: var(--color-accent-soft);
  color: var(--color-accent);
}
.nav-icon {
  width: 16px;
  height: 16px;
  display: inline-block;
  background: currentColor;
  -webkit-mask-size: contain;
  mask-size: contain;
  -webkit-mask-repeat: no-repeat;
  mask-repeat: no-repeat;
  opacity: 0.7;
}
/* 用 data-icon 区分（占位，用 emoji-free 的伪元素显示） */
.nav-icon[data-icon="scan"]::before { content: '◉'; opacity: 1; background: none; -webkit-mask: none; mask: none; }
.nav-icon[data-icon="persons"]::before { content: '☉'; opacity: 1; background: none; -webkit-mask: none; mask: none; }
.nav-icon[data-icon="face"]::before { content: '◐'; opacity: 1; background: none; -webkit-mask: none; mask: none; }
.nav-icon[data-icon="object"]::before { content: '◇'; opacity: 1; background: none; -webkit-mask: none; mask: none; }
.nav-icon[data-icon="settings"]::before { content: '⚙'; opacity: 1; background: none; -webkit-mask: none; mask: none; }

.sidebar-stats {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  padding: var(--space-3) var(--space-4);
  border-top: 1px solid var(--color-border);
}
.sidebar-stats-title {
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.05em;
  margin-bottom: var(--space-1);
}
.sidebar-stat {
  display: flex;
  align-items: baseline;
  justify-content: space-between;
  font-size: var(--text-sm);
}
.sidebar-stat-label { color: var(--color-fg-2); }
.sidebar-stat-value { color: var(--color-fg-0); font-variant-numeric: tabular-nums; }
.sidebar-stat-busy .sidebar-stat-value { color: var(--color-warning); }

.sidebar-foot {
  padding: var(--space-3) var(--space-4) 0;
  border-top: 1px solid var(--color-border);
}

.main {
  overflow-y: auto;
  background: var(--color-bg-0);
}
</style>