// Task store — 后台任务队列。
//
// 监听 `pf:task:progress` / `pf:task:completed` / `pf:task:failed`，
// 提供给 SettingsView 显示运行 / 待办 / 失败任务。
// 由于 backend 事件未发射，使用轮询保证实时更新。
import { defineStore } from 'pinia';
import { onScopeDispose, ref } from 'vue';
import { cancelTask, listTasks } from '@/api';
import { onTaskCompleted, onTaskFailed, onTaskProgress, } from '@/api/events';
const TASK_POLL_MS = 2000; // 2秒轮询
export const useTaskStore = defineStore('task', () => {
    const tasks = ref([]);
    const isLoading = ref(false);
    const lastError = ref(null);
    const filter = ref(null);
    // 内存中的进度信息（按 task_id）
    const progressMap = ref(new Map());
    // ---- subscriptions ----
    let unlistenProgress = null;
    let unlistenCompleted = null;
    let unlistenFailed = null;
    // ---- polling ----
    let pollTimer = null;
    function startListeners() {
        if (unlistenProgress)
            return;
        void onTaskProgress((p) => {
            progressMap.value.set(p.task_id, {
                progress: p.progress,
                message: p.message,
            });
        }).then((u) => (unlistenProgress = u));
        void onTaskCompleted((_p) => {
            void refresh();
        }).then((u) => (unlistenCompleted = u));
        void onTaskFailed((_p) => {
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
    function startPolling() {
        if (pollTimer)
            return;
        pollTimer = setInterval(() => {
            void refresh();
        }, TASK_POLL_MS);
    }
    function stopPolling() {
        if (pollTimer) {
            clearInterval(pollTimer);
            pollTimer = null;
        }
    }
    async function refresh(limit = 200) {
        isLoading.value = true;
        lastError.value = null;
        try {
            tasks.value = await listTasks(filter.value, limit);
        }
        catch (e) {
            lastError.value = String(e);
        }
        finally {
            isLoading.value = false;
        }
    }
    async function cancel(id) {
        await cancelTask(id);
        await refresh();
    }
    function getProgress(taskId) {
        return progressMap.value.get(taskId) ?? null;
    }
    onScopeDispose(() => {
        stopListeners();
        stopPolling();
    });
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
        startPolling,
        stopPolling,
    };
});
