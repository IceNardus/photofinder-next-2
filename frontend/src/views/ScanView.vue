<script setup lang="ts">
// ScanView — 多文件夹队列 + 后台扫描 + 实时进度 + 滚动日志。
//
// UX：
// 1) 「浏览…」选目录 → 加入队列（重复路径会自动跳过）
// 2) 队列可手动删除单个 / 一键清空
// 3) 「扫描全部」按入队顺序串行执行 `scan_folder`
// 4) 实时进度：scan_progress 事件 + 1s 轮询 scan/processing status
// 5) 扫描日志：ScanLog 组件按时间倒序显示
// 6) 扫描完成后显示图片列表（人脸/物品计数）
// 7) 清空数据库（危险操作，需确认）

import { computed, onMounted, onScopeDispose, ref, watch } from 'vue';
import { open } from '@tauri-apps/plugin-dialog';
import { useScanStore } from '@/stores/scan';
import { useAppStore } from '@/stores/app';
import { useToastStore } from '@/stores/toast';
import ProgressBar from '@/components/ProgressBar.vue';
import ScanLog from '@/components/ScanLog.vue';
import StatCard from '@/components/StatCard.vue';
import { listLibrary } from '@/api';
import type { LibraryImage } from '@/api/types';

const scan = useScanStore();
const app = useAppStore();
const toast = useToastStore();

const newPath = ref('');
const busy = computed(() => scan.isScanning);

// 图片列表（最近50张）
const images = ref<LibraryImage[]>([]);
const loadingImages = ref(false);

// 扫描期间轮询图片列表
let imagePollTimer: ReturnType<typeof setInterval> | null = null;

function startImagePolling() {
  if (imagePollTimer) return;
  imagePollTimer = setInterval(() => {
    void loadImages();
  }, 3000);
}

function stopImagePolling() {
  if (imagePollTimer) {
    clearInterval(imagePollTimer);
    imagePollTimer = null;
  }
}

async function loadImages() {
  loadingImages.value = true;
  try {
    images.value = await listLibrary(50, 0);
  } catch (e) {
    console.error('loadImages failed', e);
  } finally {
    loadingImages.value = false;
  }
}

// 扫描状态变化时自动启停轮询
watch(busy, (isBusy) => {
  if (isBusy) {
    startImagePolling();
  } else {
    stopImagePolling();
  }
});

function imageName(path: string) {
  return path.split('/').pop() ?? path;
}

async function pickFolder() {
  try {
    const result = await open({
      directory: true,
      multiple: false,
      title: '选择要扫描的照片目录',
    });
    if (typeof result === 'string') {
      newPath.value = result;
    }
  } catch (e) {
    toast.danger('打开文件夹选择失败: ' + String(e));
  }
}

function addFromInput() {
  const p = newPath.value.trim();
  if (!p) {
    toast.warning('请输入或选择目录');
    return;
  }
  scan.addFolder(p);
  newPath.value = '';
}

function addFromPicker() {
  // pickFolder 仅设置 newPath；真正入队交给 addFromInput，
  // 这样用户可以编辑后再确认，避免 dialog 自动入队的隐式行为。
  pickFolder();
}

function removeOne(path: string) {
  scan.removeFolder(path);
}

function clearQueue() {
  if (busy.value) {
    toast.warning('扫描进行中，无法清空队列');
    return;
  }
  scan.folders.splice(0);
}

async function startScanAll() {
  if (scan.folders.length === 0) {
    toast.warning('请先添加至少一个目录');
    return;
  }
  try {
    await scan.scanAll();
    await app.refreshStats();
    await loadImages(); // 刷新图片列表（显示 face/object counts）
    toast.success('所有目录扫描完成');
  } catch (e) {
    toast.danger('扫描失败: ' + String(e));
  }
}

async function stopScan() {
  try {
    await scan.stop();
    toast.info('已请求停止扫描');
  } catch (e) {
    toast.danger('停止失败: ' + String(e));
  }
}

// 监听正在扫描的文件变化，实时写入扫描日志（避免重复日志）
let prevCurrentFile = '';
const stopWatchCurrentFile = watch(
  () => scan.currentFile,
  (file) => {
    if (file && file !== prevCurrentFile && scan.isScanning) {
      prevCurrentFile = file;
      // 提取文件名用于日志
      const fileName = file.split('/').pop() ?? file;
      scan.appendLog('info', `扫描: ${fileName}`);
    }
    if (!file) {
      prevCurrentFile = '';
    }
  },
);

onScopeDispose(() => {
  stopWatchCurrentFile();
  stopImagePolling();
});

onMounted(() => { app.refresh(); loadImages(); });
</script>

<template>
  <div class="view">
    <header class="view-header">
      <h1>扫描</h1>
      <p class="muted text-sm">添加目录 → 入队 → 串行扫描。后台自动入队人脸 / 对象索引任务。</p>
    </header>

    <!-- ===== 队列管理 ===== -->
    <section class="card">
      <div class="card-header">
        <div>
          <div class="card-title">扫描队列</div>
          <div class="card-subtitle">
            {{ scan.folders.length }} 个目录
            <span v-if="scan.currentFolderIndex >= 0" class="muted">
              · 当前 #{{ scan.currentFolderIndex + 1 }}
            </span>
          </div>
        </div>
        <button
          v-if="scan.folders.length && !busy"
          class="btn sm ghost"
          @click="clearQueue"
        >清空</button>
      </div>

      <div class="row gap-2" style="margin-bottom: var(--space-3);">
        <input
          v-model="newPath"
          class="input flex-1"
          placeholder="/Users/me/Pictures"
          :disabled="busy"
          @keydown.enter="addFromInput"
        />
        <button class="btn ghost" :disabled="busy" @click="addFromPicker">浏览…</button>
        <button class="btn ghost" :disabled="busy || !newPath.trim()" @click="addFromInput">添加</button>
      </div>

      <div v-if="scan.folders.length" class="folder-list">
        <div
          v-for="(path, idx) in scan.folders"
          :key="path"
          :class="['folder-row', { active: idx === scan.currentFolderIndex }]"
        >
          <span class="folder-idx font-mono">#{{ idx + 1 }}</span>
          <span class="folder-path font-mono" :title="path">{{ path }}</span>
          <button
            class="btn sm ghost"
            :disabled="busy"
            @click="removeOne(path)"
          >移除</button>
        </div>
      </div>

      <div class="row gap-3" style="margin-top: var(--space-3);">
        <button
          class="btn primary"
          :disabled="busy || scan.folders.length === 0"
          @click="startScanAll"
        >{{ busy ? '扫描中…' : '扫描全部' }}</button>
        <button class="btn ghost" :disabled="!busy" @click="stopScan">停止</button>
      </div>
    </section>

    <!-- ===== 实时进度 ===== -->
    <section v-if="busy || scan.currentFile" class="card">
      <div class="card-header">
        <div>
          <div class="card-title">扫描进度</div>
          <div class="card-subtitle font-mono text-sm">
            {{ scan.currentFile || '准备中…' }}
          </div>
        </div>
        <span v-if="scan.currentPath" class="muted text-sm font-mono">
          {{ scan.currentPath }}
        </span>
      </div>
      <ProgressBar
        :value="scan.progress()"
        :hint="`${scan.scanned} / ${scan.candidateCount} 文件`"
      />
    </section>

    <!-- ===== 后台处理实时状态 ===== -->
    <section v-if="scan.scanStatus.is_scanning || scan.processingStatus.current_image" class="card">
      <div class="card-header">
        <div>
          <div class="card-title">后台处理</div>
          <div class="card-subtitle font-mono text-sm">
            {{ scan.processingStatus.current_image || '空闲' }}
          </div>
        </div>
        <span v-if="scan.processingStatus.log_message" class="muted text-sm">
          {{ scan.processingStatus.log_message }}
        </span>
      </div>
      <div class="stat-grid">
        <StatCard label="已处理" :value="`${scan.scanStatus.processed_images}/${scan.scanStatus.total_images}`" />
        <StatCard label="待处理" :value="scan.scanStatus.pending_tasks" />
        <StatCard label="当前图 faces" :value="scan.processingStatus.current_faces" />
        <StatCard label="当前图 objects" :value="scan.processingStatus.current_objects" />
        <StatCard label="当前图 patches" :value="scan.processingStatus.current_patches" />
      </div>
      <div v-if="scan.processingStatus.last_completion_message" class="muted text-sm" style="margin-top: var(--space-3);">
        上次完成: {{ scan.processingStatus.last_completion_message }}
      </div>
    </section>

    <!-- ===== 最近一次结果 ===== -->
    <section v-if="scan.lastResult" class="card">
      <div class="card-header">
        <div>
          <div class="card-title">最近一次结果</div>
          <div class="card-subtitle font-mono text-sm">{{ scan.lastResult.path }}</div>
        </div>
        <span v-if="scan.lastError" class="tag danger">失败</span>
        <span v-else class="tag success">成功</span>
      </div>

      <div v-if="scan.lastError" class="text-sm" style="color: var(--color-danger);">
        {{ scan.lastError }}
      </div>
      <div v-else class="stat-grid">
        <StatCard label="候选" :value="scan.lastResult.total_found" />
        <StatCard label="新增" :value="scan.lastResult.new_images" />
        <StatCard label="跳过" :value="scan.lastResult.skipped" />
        <StatCard label="被拒" :value="scan.lastResult.filtered" hint="非 Photo (截图/二次元/海报)" />
        <StatCard label="低bpp" :value="scan.lastResult.filtered_bpp" hint="vector / icon" />
        <StatCard label="人脸任务" :value="scan.lastResult.queued_face_tasks" />
        <StatCard label="对象任务" :value="scan.lastResult.queued_object_tasks" />
      </div>
    </section>

    <!-- ===== 图片列表（含 face/object counts）===== -->
    <section class="card">
      <div class="card-header">
        <div>
          <div class="card-title">已扫描图片</div>
          <div class="card-subtitle text-sm muted">
            <span v-if="busy && scan.discoveredImages.length > 0">
              实时入库 {{ scan.discoveredImages.length }} 张
              <span class="text-accent">（扫描中…）</span>
            </span>
            <span v-else>
              {{ images.length }} 张
              <span v-if="loadingImages">（加载中…）</span>
            </span>
          </div>
        </div>
        <button class="btn sm ghost" :disabled="loadingImages" @click="loadImages">刷新</button>
      </div>

      <!-- 扫描中：实时显示入库进度 -->
      <div v-if="busy && scan.discoveredImages.length > 0" class="scan-progress-list">
        <div class="scan-progress-header">
          <span class="text-sm muted">入库进度</span>
          <span class="text-sm font-mono">{{ scan.discoveredImages.length }} / {{ scan.candidateCount || '?' }}</span>
        </div>
        <div class="scan-progress-bar-wrap">
          <div
            class="scan-progress-bar-fill"
            :style="{ width: scan.candidateCount > 0 ? `${Math.min(100, (scan.discoveredImages.length / scan.candidateCount) * 100)}%` : '0%' }"
          />
        </div>
        <div class="scan-image-grid">
          <div
            v-for="img in scan.discoveredImages.slice(-24)"
            :key="img.image_id"
            class="scan-image-item"
            :title="img.path"
          >
            <div class="scan-image-name">{{ img.path.split('/').pop() }}</div>
          </div>
        </div>
        <div v-if="scan.discoveredImages.length > 24" class="text-xs muted">
          还有 {{ scan.discoveredImages.length - 24 }} 张…
        </div>
      </div>

      <div v-else-if="loadingImages && images.length === 0" class="text-sm muted" style="padding: var(--space-3);">
        加载中…
      </div>
      <div v-else-if="images.length === 0" class="text-sm muted" style="padding: var(--space-3);">
        暂无图片，请先扫描目录
      </div>
      <div v-else class="image-table-wrap">
        <table class="image-table">
          <thead>
            <tr>
              <th>图片</th>
              <th>状态</th>
              <th>人脸</th>
              <th>物品</th>
            </tr>
          </thead>
          <tbody>
            <tr v-for="img in images" :key="img.id">
              <td class="img-name" :title="img.path">{{ imageName(img.path) }}</td>
              <td>
                <span :class="['tag', img.scan_status === 'indexed' ? 'success' : 'muted']">
                  {{ img.scan_status }}
                </span>
              </td>
              <td class="num">
                <span v-if="img.face_count > 0" class="face-count">{{ img.face_count }}</span>
                <span v-else class="muted">–</span>
              </td>
              <td class="num">
                <span v-if="img.object_count > 0" class="obj-count">{{ img.object_count }}</span>
                <span v-else class="muted">–</span>
              </td>
            </tr>
          </tbody>
        </table>
      </div>
    </section>

    <!-- ===== 滚动日志 ===== -->
    <section class="card">
      <ScanLog />
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
.folder-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
  max-height: 240px;
  overflow-y: auto;
  background: var(--color-bg-2);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  padding: var(--space-2);
}
.folder-row {
  display: grid;
  grid-template-columns: 40px 1fr auto;
  gap: var(--space-2);
  align-items: center;
  padding: var(--space-2) var(--space-3);
  background: var(--color-bg-1);
  border-radius: var(--radius-sm);
}
.folder-row.active {
  border-left: 2px solid var(--color-accent);
  background: var(--color-accent-soft);
}
.folder-idx {
  color: var(--color-fg-2);
  font-size: var(--text-sm);
}
.folder-path {
  font-size: var(--text-sm);
  color: var(--color-fg-1);
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
}
.image-table-wrap {
  overflow-x: auto;
  max-height: 400px;
  overflow-y: auto;
}
.image-table {
  width: 100%;
  border-collapse: collapse;
  font-size: var(--text-sm);
}
.image-table th {
  text-align: left;
  padding: var(--space-2) var(--space-3);
  border-bottom: 1px solid var(--color-border);
  color: var(--color-fg-2);
  font-weight: 500;
  position: sticky;
  top: 0;
  background: var(--color-bg-2);
}
.image-table td {
  padding: var(--space-2) var(--space-3);
  border-bottom: 1px solid var(--color-border);
}
.image-table tr:last-child td {
  border-bottom: none;
}
.img-name {
  max-width: 200px;
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  color: var(--color-fg-1);
}
.num {
  text-align: right;
  font-variant-numeric: tabular-nums;
}
.face-count {
  color: var(--color-accent);
  font-weight: 600;
}
.obj-count {
  color: var(--color-success);
  font-weight: 600;
}

/* 扫描中实时入库进度 */
.scan-progress-list {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
  padding: var(--space-3);
  background: var(--color-bg-2);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
}
.scan-progress-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
}
.scan-progress-bar-wrap {
  height: 6px;
  background: var(--color-bg-3);
  border-radius: 3px;
  overflow: hidden;
}
.scan-progress-bar-fill {
  height: 100%;
  background: var(--color-accent);
  border-radius: 3px;
  transition: width 0.3s ease;
}
.scan-image-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(140px, 1fr));
  gap: var(--space-1);
  max-height: 200px;
  overflow-y: auto;
}
.scan-image-item {
  padding: var(--space-1) var(--space-2);
  background: var(--color-bg-1);
  border-radius: var(--radius-sm);
  font-size: var(--text-xs);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.scan-image-name {
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
</style>