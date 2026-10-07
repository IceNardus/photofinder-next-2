// App store — 应用基本信息 + 统计 + 模型状态。
//
// 在 App 启动时由 main.ts 触发一次 refresh()，所有 view 共享。

import { defineStore } from 'pinia';
import { ref } from 'vue';
import {
  clearDatabase,
  getAppInfo,
  getModelsStatus,
  getStatistics,
  indexPending,
  type AppInfo,
  type ClearDatabaseResult,
  type ModelStatus,
  type Statistics,
} from '@/api';

export const useAppStore = defineStore('app', () => {
  const info = ref<AppInfo | null>(null);
  const stats = ref<Statistics | null>(null);
  const models = ref<ModelStatus[]>([]);

  const isLoading = ref(false);
  const lastError = ref<string | null>(null);

  async function refresh() {
    isLoading.value = true;
    lastError.value = null;
    try {
      const [i, s, m] = await Promise.all([
        getAppInfo(),
        getStatistics(),
        getModelsStatus(),
      ]);
      info.value = i;
      stats.value = s;
      models.value = m;
    } catch (e) {
      lastError.value = String(e);
    } finally {
      isLoading.value = false;
    }
  }

  async function refreshStats() {
    try {
      stats.value = await getStatistics();
    } catch (e) {
      lastError.value = String(e);
    }
  }

  async function enqueuePending(): Promise<number> {
    return await indexPending();
  }

  async function clearDb(): Promise<ClearDatabaseResult> {
    const r = await clearDatabase();
    await refresh();
    return r;
  }

  return {
    info,
    stats,
    models,
    isLoading,
    lastError,
    refresh,
    refreshStats,
    enqueuePending,
    clearDb,
  };
});
