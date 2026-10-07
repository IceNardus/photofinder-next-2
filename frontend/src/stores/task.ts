// Task store — 后台任务队列。
//
// 监听 `pf:task:progress` / `pf:task:completed` / `pf:task:failed`，
// 提供给 SettingsView 显示运行 / 待办 / 失败任务。

import { defineStore } from 'pinia';
import { onScopeDispose, ref } from 'vue';
import { cancelTask, listTasks, type Task, type TaskStatusString } from '@/api';
import {
  onTaskCompleted,
  onTaskFailed,
  onTaskProgress,
  type TaskCompletedPayload,
  type TaskFailedPayload,
  type TaskProgressPayload,
} from '@/api/events';

export const useTaskStore = defineStore('task', () => {
  const tasks = ref<Task[]>([]);
  const isLoading = ref(false);
  const lastError = ref<string | null>(null);
  const filter = ref<TaskStatusString | null>(null);

  // 内存中的进度信息（按 task_id）
  const progressMap = ref<Map<number, { progress: number; message: string | null }>>(
    new Map(),
  );

  // ---- subscriptions ----
  let unlistenProgress: (() => void) | null = null;
  let unlistenCompleted: (() => void) | null = null;
  let unlistenFailed: (() => void) | null = null;

  function startListeners() {
    if (unlistenProgress) return;
    void onTaskProgress((p: TaskProgressPayload) => {
      progressMap.value.set(p.task_id, {
        progress: p.progress,
        message: p.message,
      });
    }).then((u) => (unlistenProgress = u));
    void onTaskCompleted((_p: TaskCompletedPayload) => {
      void refresh();
    }).then((u) => (unlistenCompleted = u));
    void onTaskFailed((_p: TaskFailedPayload) => {
      void refresh();
    }).then((u) => (unlistenFailed = u));
  }

  function stopListeners() {
    unlistenProgress?.();
    unlistenCompleted?.();
    unlistenFailed?.();
    unlistenProgress = null;
    unlistenCompleted = null;
    unlistenFailed = null;
  }

  async function refresh(limit = 200) {
    isLoading.value = true;
    lastError.value = null;
    try {
      tasks.value = await listTasks(filter.value, limit);
    } catch (e) {
      lastError.value = String(e);
    } finally {
      isLoading.value = false;
    }
  }

  async function cancel(id: number) {
    await cancelTask(id);
    await refresh();
  }

  function getProgress(taskId: number): { progress: number; message: string | null } | null {
    return progressMap.value.get(taskId) ?? null;
  }

  onScopeDispose(() => stopListeners());

  return {
    tasks,
    isLoading,
    lastError,
    filter,
    progressMap,
    refresh,
    cancel,
    getProgress,
    startListeners,
    stopListeners,
  };
});
