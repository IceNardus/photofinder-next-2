// Scan store — 扫描状态 + 进度 + 多文件夹队列 + 日志。
//
// Phase 5 升级：
// - 多文件夹队列（`folders: string[]`）
// - 顺序扫描（用一个 `pointer` 指向下一个要扫的文件夹）
// - 滚动日志（`logEntries`，最多保留 100 条）
// - 1s 轮询 `get_scan_status` + `get_processing_status`

import { defineStore } from 'pinia';
import { onScopeDispose, ref, watch } from 'vue';
import { getProcessingStatus, getScanStatus, scanFolder, stopScan } from '@/api';
import {
  onScanCompleted,
  onScanImageDiscovered,
  onScanProgress,
  type ScanCompletedPayload,
  type ScanImageDiscoveredPayload,
  type ScanProgressPayload,
} from '@/api/events';
import { useSettingsStore } from '@/stores/settings';

const MAX_LOG_ENTRIES_DEFAULT = 100;

export type LogLevel = 'info' | 'success' | 'warn' | 'error';

export interface LogEntry {
  /** 时间戳（ms since epoch） */
  ts: number;
  level: LogLevel;
  message: string;
}

export const useScanStore = defineStore('scan', () => {
  // ---- 单文件夹扫描状态 ----
  const isScanning = ref(false);
  const currentPath = ref('');
  const currentFile = ref('');
  const candidateCount = ref(0);
  const scanned = ref(0);

  const lastResult = ref<ScanCompletedPayload | null>(null);
  const lastError = ref<string | null>(null);

  // ---- Phase 5：多文件夹队列 ----
  const folders = ref<string[]>([]);
  const currentFolderIndex = ref(-1);

  // ---- 滚动日志 ----
  const logEntries = ref<LogEntry[]>([]);

  // ---- 实时发现的图片（扫描过程中逐张追加）----
  interface DiscoveredImage {
    image_id: number;
    path: string;
    /** 入库时间 */
    ts: number;
  }
  const discoveredImages = ref<DiscoveredImage[]>([]);

  // ---- 实时状态（来自 get_scan_status / get_processing_status） ----
  const scanStatus = ref({
    is_scanning: false,
    total_images: 0,
    processed_images: 0,
    pending_tasks: 0,
    current_file: '',
    current_faces: 0,
    current_objects: 0,
    current_patches: 0,
  });
  const processingStatus = ref({
    current_image: '',
    current_faces: 0,
    current_patches: 0,
    current_objects: 0,
    log_message: '',
    last_completion_message: '',
  });

  // ---- Subscriptions ----
  let unlistenProgress: (() => void) | null = null;
  let unlistenCompleted: (() => void) | null = null;
  let unlistenImageDiscovered: (() => void) | null = null;
  let pollTimer: ReturnType<typeof setInterval> | null = null;

  function appendLog(level: LogLevel, message: string) {
    logEntries.value.push({ ts: Date.now(), level, message });
    const max = getMaxLog();
    if (logEntries.value.length > max) {
      logEntries.value = logEntries.value.slice(-max);
    }
  }

  function getMaxLog(): number {
    try {
      return useSettingsStore().settings.logBufferSize;
    } catch {
      return MAX_LOG_ENTRIES_DEFAULT;
    }
  }

  async function startListeners() {
    if (unlistenProgress && unlistenCompleted && unlistenImageDiscovered) return;
    const [u1, u2, u3] = await Promise.all([
      onScanProgress((p: ScanProgressPayload) => {
        currentPath.value = p.path;
        currentFile.value = p.current_file;
        candidateCount.value = p.candidate_count;
        scanned.value = p.scanned;
      }),
      onScanCompleted((p: ScanCompletedPayload) => {
        isScanning.value = false;
        lastResult.value = p;
        lastError.value = p.error;
        if (p.error) {
          appendLog('error', `扫描 ${p.path} 失败: ${p.error}`);
        } else {
          appendLog(
            'success',
            `扫描 ${p.path} 完成: ${p.new_images} 新增, ${p.skipped} 跳过, ${p.queued_face_tasks} face, ${p.queued_object_tasks} object`,
          );
        }
      }),
      onScanImageDiscovered((p: ScanImageDiscoveredPayload) => {
        const fileName = p.path.split('/').pop() ?? p.path;
        discoveredImages.value.push({ image_id: p.image_id, path: p.path, ts: Date.now() });
        appendLog('info', `入库: ${fileName}`);
      }),
    ]);
    unlistenProgress = u1;
    unlistenCompleted = u2;
    unlistenImageDiscovered = u3;
  }

  function stopListeners() {
    unlistenProgress?.();
    unlistenCompleted?.();
    unlistenImageDiscovered?.();
    unlistenProgress = null;
    unlistenCompleted = null;
    unlistenImageDiscovered = null;
  }

  function startPolling() {
    if (pollTimer) return;
    pollTimer = setInterval(() => {
      void refreshStatus();
    }, getScanPollMs());
  }

  function restartPolling() {
    stopPolling();
    startPolling();
  }

  function getScanPollMs(): number {
    try {
      return useSettingsStore().settings.scanPollMs;
    } catch {
      return 1000;
    }
  }

  function stopPolling() {
    if (pollTimer) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }

  async function refreshStatus() {
    try {
      const [s, p] = await Promise.all([getScanStatus(), getProcessingStatus()]);
      scanStatus.value = s;
      processingStatus.value = p;
    } catch {
      // ignore polling errors
    }
  }

  // ---- Actions ----
  function addFolder(path: string) {
    if (path && !folders.value.includes(path)) {
      folders.value.push(path);
      appendLog('info', `已添加文件夹: ${path}`);
    }
  }

  function removeFolder(path: string) {
    const idx = folders.value.indexOf(path);
    if (idx >= 0) {
      folders.value.splice(idx, 1);
      if (currentFolderIndex.value > idx) currentFolderIndex.value--;
      appendLog('info', `已移除文件夹: ${path}`);
    }
  }

  function clearLog() {
    logEntries.value = [];
  }

  /** 扫描单个文件夹（不进入队列） */
  async function scan(path: string) {
    isScanning.value = true;
    currentPath.value = path;
    currentFile.value = '';
    candidateCount.value = 0;
    scanned.value = 0;
    lastResult.value = null;
    lastError.value = null;
    discoveredImages.value = []; // 清空上次的发现记录
    await startListeners(); // 必须等待 listener 注册完成
    startPolling();
    appendLog('info', `开始扫描: ${path}`);
    try {
      const r = await scanFolder(path);
      lastResult.value = {
        path: r.path,
        total_found: r.total_found,
        new_images: r.new_images,
        skipped: r.skipped,
        filtered: r.filtered,
        filtered_bpp: r.filtered_bpp,
        queued_face_tasks: r.queued_face_tasks,
        queued_object_tasks: r.queued_object_tasks,
        error: null,
      };
      return r;
    } catch (e) {
      lastError.value = String(e);
      appendLog('error', `${path}: ${e}`);
      throw e;
    } finally {
      isScanning.value = false;
      stopPolling();
    }
  }

  /** 顺序扫描整个文件夹队列 */
  async function scanAll() {
    if (folders.value.length === 0) {
      appendLog('warn', '文件夹队列为空');
      return;
    }
    isScanning.value = true;
    await startListeners();
    startPolling();
    appendLog('info', `开始扫描 ${folders.value.length} 个文件夹`);
    try {
      for (let i = 0; i < folders.value.length; i++) {
        currentFolderIndex.value = i;
        const path = folders.value[i];
        if (!path) continue;
        await scan(path);
      }
      appendLog('success', '所有文件夹扫描完成');
    } finally {
      isScanning.value = false;
      currentFolderIndex.value = -1;
      stopPolling();
    }
  }

  async function stop() {
    await stopScan();
    isScanning.value = false;
    stopPolling();
    appendLog('warn', '扫描已停止');
  }

  const progress = (): number => {
    if (candidateCount.value === 0) return 0;
    return Math.min(1, scanned.value / candidateCount.value);
  };

  onScopeDispose(() => {
    stopListeners();
    stopPolling();
  });

  // 用户改了 polling 间隔 / log buffer → 立即生效
  watch(
    () => {
      try {
        return useSettingsStore().settings.scanPollMs;
      } catch {
        return 1000;
      }
    },
    () => restartPolling(),
  );
  watch(
    () => {
      try {
        return useSettingsStore().settings.logBufferSize;
      } catch {
        return MAX_LOG_ENTRIES_DEFAULT;
      }
    },
    (n) => {
      if (logEntries.value.length > n) {
        logEntries.value = logEntries.value.slice(-n);
      }
    },
  );

  return {
    isScanning,
    currentPath,
    currentFile,
    candidateCount,
    scanned,
    lastResult,
    lastError,
    scan,
    scanAll,
    stop,
    progress,
    folders,
    currentFolderIndex,
    addFolder,
    removeFolder,
    logEntries,
    clearLog,
    appendLog,
    discoveredImages,
    scanStatus,
    processingStatus,
    refreshStatus,
  };
});
