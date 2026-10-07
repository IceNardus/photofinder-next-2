<script setup lang="ts">
// SettingsView — 应用信息、模型状态、任务队列、调参面板、危险操作（清库）。
//
// 启动时一次 refresh()，监听 task 事件后自动刷新 task 列表。
// 调参面板（Phase 8）通过 settings store 持久化到 localStorage。

import { onMounted } from 'vue';
import { useAppStore } from '@/stores/app';
import { useTaskStore } from '@/stores/task';
import { useSettingsStore } from '@/stores/settings';
import { useToastStore } from '@/stores/toast';
import StatCard from '@/components/StatCard.vue';
import TaskTable from '@/components/TaskTable.vue';
import EmptyState from '@/components/EmptyState.vue';

const app = useAppStore();
const task = useTaskStore();
const settings = useSettingsStore();
const toast = useToastStore();

onMounted(async () => {
  await app.refresh();
  task.startListeners();
  await task.refresh();
});

async function enqueuePending() {
  try {
    const n = await app.enqueuePending();
    toast.success(`已入队 ${n} 张待索引图片`);
    await task.refresh();
  } catch (e) {
    toast.danger('入队失败: ' + String(e));
  }
}

async function clearDb() {
  if (!window.confirm('确定要清空数据库吗？所有图片、人脸、人物记录将被删除。建议重启应用。')) {
    return;
  }
  try {
    const r = await app.clearDb();
    if (r.success) {
      toast.success('数据库已清空。建议重启应用以重建 HNSW 索引。');
    } else {
      toast.warning('部分清理失败：' + r.errors.join('; '));
    }
    await task.refresh();
  } catch (e) {
    toast.danger('清空失败: ' + String(e));
  }
}

function setFilter(f: string | null) {
  task.filter = (f as 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | null);
  void task.refresh();
}

function fmtSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
  return `${(bytes / 1024 / 1024 / 1024).toFixed(2)} GB`;
}
</script>

<template>
  <div class="view">
    <header class="view-header">
      <h1>设置</h1>
      <p class="muted text-sm">应用信息、模型状态、任务队列、危险操作。</p>
    </header>

    <!-- ============ App info ============ -->
    <section class="card">
      <div class="card-header">
        <div>
          <div class="card-title">应用</div>
        </div>
        <button class="btn sm ghost" @click="app.refresh()">刷新</button>
      </div>
      <div v-if="app.info" class="kv">
        <div class="kv-row"><span class="muted">名称</span><span>{{ app.info.name }}</span></div>
        <div class="kv-row"><span class="muted">版本</span><span class="font-mono">{{ app.info.version }}</span></div>
        <div class="kv-row"><span class="muted">平台</span><span class="font-mono">{{ app.info.platform }}</span></div>
        <div class="kv-row"><span class="muted">数据目录</span><span class="font-mono text-sm">{{ app.info.data_dir }}</span></div>
        <div class="kv-row"><span class="muted">数据库</span><span class="font-mono text-sm">{{ app.info.db_path }}</span></div>
        <div class="kv-row"><span class="muted">模型目录</span><span class="font-mono text-sm">{{ app.info.models_dir ?? '—' }}</span></div>
      </div>
    </section>

    <!-- ============ Stats ============ -->
    <section v-if="app.stats" class="card">
      <div class="card-header">
        <div>
          <div class="card-title">统计</div>
        </div>
        <button class="btn sm primary" @click="enqueuePending">入队待索引图片</button>
      </div>
      <div class="stat-grid">
        <StatCard label="图片" :value="app.stats.image_count" />
        <StatCard label="人脸" :value="app.stats.face_count" />
        <StatCard label="对象" :value="app.stats.object_count" />
        <StatCard label="人物" :value="app.stats.person_count" />
        <StatCard label="待办任务" :value="app.stats.pending_task_count" />
        <StatCard label="已完成" :value="app.stats.completed_task_count" />
        <StatCard label="失败" :value="app.stats.failed_task_count" />
        <StatCard label="人脸向量数" :value="app.stats.face_index_size" />
      </div>
    </section>

    <!-- ============ Models ============ -->
    <section class="card">
      <div class="card-header">
        <div>
          <div class="card-title">模型</div>
          <div class="card-subtitle">{{ app.models.length }} 个模型</div>
        </div>
      </div>
      <EmptyState v-if="app.models.length === 0" title="未注册模型" hint="检查 models/manifest.toml" />
      <div v-else class="table-wrap">
        <table class="table">
          <thead>
            <tr>
              <th>ID</th>
              <th>状态</th>
              <th>路径</th>
              <th style="text-align: right;">大小</th>
              <th style="text-align: right;">校验</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="m in app.models" :key="m.id">
              <td class="font-mono">{{ m.id }}</td>
              <td>
                <span :class="['tag', m.loaded ? 'success' : 'warning']">
                  {{ m.loaded ? '已加载' : '未加载' }}
                </span>
              </td>
              <td class="font-mono text-sm muted">{{ m.path ?? '—' }}</td>
              <td class="font-mono text-sm" style="text-align: right;">{{ fmtSize(m.size_bytes) }}</td>
              <td style="text-align: right;">
                <span :class="['tag', m.verified ? 'success' : 'warning']">
                  {{ m.verified ? 'OK' : '未校验' }}
                </span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- ============ Tasks ============ -->
    <section class="card">
      <div class="card-header">
        <div>
          <div class="card-title">任务队列</div>
          <div class="card-subtitle">{{ task.tasks.length }} 个任务</div>
        </div>
        <div class="row gap-2">
          <button class="btn sm ghost" @click="setFilter(null)">全部</button>
          <button class="btn sm ghost" @click="setFilter('pending')">待办</button>
          <button class="btn sm ghost" @click="setFilter('running')">运行</button>
          <button class="btn sm ghost" @click="setFilter('failed')">失败</button>
          <button class="btn sm ghost" @click="setFilter('completed')">完成</button>
          <button class="btn sm primary" @click="task.refresh()">刷新</button>
        </div>
      </div>
      <TaskTable />
    </section>

    <!-- ============ Tuning ============ -->
    <section class="card">
      <div class="card-header">
        <div>
          <div class="card-title">调参</div>
          <div class="card-subtitle">UI 行为参数；持久化到本地</div>
        </div>
        <button class="btn sm ghost" @click="settings.reset()">恢复默认</button>
      </div>

      <div class="tuning-grid">
        <div class="tuning-row">
          <label>Top K（默认检索数量）</label>
          <input
            v-model.number="settings.settings.topK"
            type="number"
            class="input"
            min="1"
            max="200"
          />
        </div>
        <div class="tuning-row">
          <label>每页结果数</label>
          <input
            v-model.number="settings.settings.resultsPerPage"
            type="number"
            class="input"
            min="1"
            max="100"
          />
        </div>
        <div class="tuning-row">
          <label>高相似度阈值（≥ = green）</label>
          <input
            v-model.number="settings.settings.similarityHigh"
            type="number"
            class="input"
            min="0"
            max="1"
            step="0.05"
          />
        </div>
        <div class="tuning-row">
          <label>中相似度阈值（≥ = yellow）</label>
          <input
            v-model.number="settings.settings.similarityMid"
            type="number"
            class="input"
            min="0"
            max="1"
            step="0.05"
          />
        </div>
        <div class="tuning-row">
          <label>Sidebar 轮询间隔 (ms)</label>
          <input
            v-model.number="settings.settings.sidebarPollMs"
            type="number"
            class="input"
            min="500"
            step="500"
          />
        </div>
        <div class="tuning-row">
          <label>扫描状态轮询间隔 (ms)</label>
          <input
            v-model.number="settings.settings.scanPollMs"
            type="number"
            class="input"
            min="500"
            step="500"
          />
        </div>
        <div class="tuning-row">
          <label>日志缓冲上限</label>
          <input
            v-model.number="settings.settings.logBufferSize"
            type="number"
            class="input"
            min="10"
            max="1000"
          />
        </div>
      </div>
    </section>

    <!-- ============ Danger zone ============ -->
    <section class="card danger">
      <div class="card-header">
        <div>
          <div class="card-title" style="color: var(--color-danger);">危险操作</div>
          <div class="card-subtitle">不可逆</div>
        </div>
      </div>
      <div class="row gap-3 center">
        <div class="flex-1">
          <div>清空数据库</div>
          <div class="muted text-sm">删除所有图片、人脸、对象、人物、任务记录；删除 HNSW 索引文件。建议先停止所有扫描。</div>
        </div>
        <button class="btn danger" @click="clearDb">清空</button>
      </div>
    </section>
  </div>
</template>

<style scoped>
.view {
  display: flex;
  flex-direction: column;
  gap: var(--space-5);
  max-width: var(--content-max);
  margin: 0 auto;
  padding: var(--space-5);
}
.view-header h1 {
  font-size: var(--text-2xl);
  font-weight: 600;
  margin-bottom: var(--space-1);
}
.kv { display: flex; flex-direction: column; gap: var(--space-2); }
.kv-row {
  display: grid;
  grid-template-columns: 120px 1fr;
  gap: var(--space-3);
  font-size: var(--text-sm);
}
.kv-row .muted { font-size: var(--text-sm); }
.table-wrap {
  overflow: auto;
  max-height: 320px;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
}
.card.danger {
  border-color: rgba(255, 69, 58, 0.3);
}
.tuning-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap: var(--space-3);
}
.tuning-row {
  display: grid;
  grid-template-columns: 1fr 100px;
  gap: var(--space-3);
  align-items: center;
  font-size: var(--text-sm);
}
.tuning-row label {
  color: var(--color-fg-2);
}
</style>
