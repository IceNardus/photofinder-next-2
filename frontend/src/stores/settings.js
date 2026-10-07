// Settings store — 用户可调参数。
//
// 持久化到 localStorage；启动时 hydrate；改值即时生效（其他 store 可 watch）。
// 数值约束：top_k ∈ [1, 200]，page size ∈ [1, 100]，polling ≥ 500ms，log ≥ 10。
import { defineStore } from 'pinia';
import { ref, watch } from 'vue';
const STORAGE_KEY = 'photofinder.settings.v1';
export const DEFAULT_SETTINGS = {
    topK: 30,
    resultsPerPage: 20,
    similarityHigh: 0.7,
    similarityMid: 0.5,
    sidebarPollMs: 5000,
    scanPollMs: 1000,
    logBufferSize: 100,
};
function clampSettings(s) {
    return {
        topK: Math.max(1, Math.min(200, Math.round(s.topK))),
        resultsPerPage: Math.max(1, Math.min(100, Math.round(s.resultsPerPage))),
        similarityHigh: Math.max(0, Math.min(1, s.similarityHigh)),
        similarityMid: Math.max(0, Math.min(1, s.similarityMid)),
        sidebarPollMs: Math.max(500, Math.round(s.sidebarPollMs)),
        scanPollMs: Math.max(500, Math.round(s.scanPollMs)),
        logBufferSize: Math.max(10, Math.min(1000, Math.round(s.logBufferSize))),
    };
}
function loadFromStorage() {
    try {
        const raw = localStorage.getItem(STORAGE_KEY);
        if (!raw)
            return { ...DEFAULT_SETTINGS };
        const parsed = JSON.parse(raw);
        return clampSettings({ ...DEFAULT_SETTINGS, ...parsed });
    }
    catch {
        return { ...DEFAULT_SETTINGS };
    }
}
export const useSettingsStore = defineStore('settings', () => {
    const settings = ref(loadFromStorage());
    // 持久化
    watch(settings, (s) => {
        try {
            localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
        }
        catch {
            // ignore quota errors
        }
    }, { deep: true });
    function reset() {
        settings.value = { ...DEFAULT_SETTINGS };
    }
    function patch(partial) {
        settings.value = clampSettings({ ...settings.value, ...partial });
    }
    return { settings, reset, patch };
});
