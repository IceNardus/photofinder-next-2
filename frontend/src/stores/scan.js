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
import { onScanCompleted, onScanImageDiscovered, onScanProgress, onTaskProgress, onTaskCompleted, onTaskFailed, } from '@/api/events';
import { useSettingsStore } from '@/stores/settings';
const MAX_LOG_ENTRIES_DEFAULT = 100;
export const useScanStore = defineStore('scan', () => {
    // ---- 单文件夹扫描状态 ----
    const isScanning = ref(false);
    const currentPath = ref('');
    const currentFile = ref('');
    const candidateCount = ref(0);
    const scanned = ref(0);
    const lastResult = ref(null);
    const lastError = ref(null);
    // ---- Phase 5：多文件夹队列 ----
    const folders = ref([]);
    const currentFolderIndex = ref(-1);
    // ---- 滚动日志 ----
    const logEntries = ref([]);
    const discoveredImages = ref([]);
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
    // ---- 扫描阶段追踪（用于进度模态框）----
    // 'idle' | 'discovering' | 'indexing' | 'done'
    const scanPhase = ref('idle');
    // 当前正在扫描的文件夹索引（scanAll 用）
    const totalFolders = ref(0);
    // 所有文件夹发现是否完毕
    const allDiscoveryDone = ref(false);
    // 索引轮询 timer（discovery 阶段结束后继续追踪 pending_tasks）
    let indexPollTimer = null;
    function startIndexPoll() {
        if (indexPollTimer)
            return;
        indexPollTimer = setInterval(async () => {
            await refreshStatus();
            if (scanStatus.value.pending_tasks === 0 && allDiscoveryDone.value) {
                scanPhase.value = 'done';
                stopIndexPoll();
            }
        }, 1000);
    }
    function stopIndexPoll() {
        if (indexPollTimer) {
            clearInterval(indexPollTimer);
            indexPollTimer = null;
        }
    }
    // ---- Subscriptions ----
    let unlistenProgress = null;
    let unlistenCompleted = null;
    let unlistenImageDiscovered = null;
    let unlistenTaskProgress = null;
    let unlistenTaskCompleted = null;
    let unlistenTaskFailed = null;
    let pollTimer = null;
    const currentTask = ref(null);
    const completedTasks = ref([]);
    // 任务 kind → 中文标签
    const TASK_KIND_LABELS = {
        Scan: '扫描文件夹',
        IndexFace: '人脸索引',
        IndexObject: '对象索引',
        IndexImage: '图片索引',
        IndexPatch: 'Patch 索引',
        IndexVisualScan: '视觉扫描',
        BuildIndex: '构建索引',
        ClusterFaces: '人脸聚类',
    };
    function appendLog(level, message) {
        logEntries.value.push({ ts: Date.now(), level, message });
        const max = getMaxLog();
        if (logEntries.value.length > max) {
            logEntries.value = logEntries.value.slice(-max);
        }
    }
    function getMaxLog() {
        try {
            return useSettingsStore().settings.logBufferSize;
        }
        catch {
            return MAX_LOG_ENTRIES_DEFAULT;
        }
    }
    async function startListeners() {
        if (unlistenProgress && unlistenCompleted && unlistenImageDiscovered)
            return;
        const [u1, u2, u3] = await Promise.all([
            onScanProgress((p) => {
                currentPath.value = p.path;
                currentFile.value = p.current_file;
                candidateCount.value = p.candidate_count;
                scanned.value = p.scanned;
            }),
            onScanCompleted((p) => {
                isScanning.value = false;
                lastResult.value = p;
                lastError.value = p.error;
                if (p.error) {
                    appendLog('error', `扫描 ${p.path} 失败: ${p.error}`);
                }
                else {
                    appendLog('success', `扫描 ${p.path} 完成: ${p.new_images} 新增, ${p.skipped} 跳过, ${p.queued_face_tasks} face, ${p.queued_object_tasks} object`);
                }
            }),
            onScanImageDiscovered((p) => {
                const fileName = p.path.split('/').pop() ?? p.path;
                discoveredImages.value.push({ image_id: p.image_id, path: p.path, ts: Date.now() });
                appendLog('info', `入库: ${fileName}`);
            }),
        ]);
        unlistenProgress = u1;
        unlistenCompleted = u2;
        unlistenImageDiscovered = u3;
    }
    async function startTaskListeners() {
        if (unlistenTaskProgress && unlistenTaskCompleted && unlistenTaskFailed)
            return;
        const [u1, u2, u3] = await Promise.all([
            onTaskProgress((p) => {
                const kindLabel = TASK_KIND_LABELS[p.kind] ?? p.kind;
                if (p.progress === 0) {
                    // 任务开始
                    currentTask.value = {
                        taskId: p.task_id,
                        kind: p.kind,
                        kindLabel,
                        progress: 0,
                        message: p.message ?? '开始',
                    };
                    appendLog('info', `${kindLabel} 开始`);
                }
                else {
                    currentTask.value = {
                        taskId: p.task_id,
                        kind: p.kind,
                        kindLabel,
                        progress: p.progress,
                        message: p.message ?? '',
                    };
                }
            }),
            onTaskCompleted((p) => {
                const kindLabel = TASK_KIND_LABELS[p.kind] ?? p.kind;
                appendLog('success', `${kindLabel} 完成 (${p.duration_ms}ms)`);
                completedTasks.value.push({ taskId: p.task_id, kind: p.kind, duration_ms: p.duration_ms });
                if (currentTask.value?.taskId === p.task_id) {
                    currentTask.value = null;
                }
            }),
            onTaskFailed((p) => {
                const kindLabel = TASK_KIND_LABELS[p.kind] ?? p.kind;
                appendLog('error', `${kindLabel} 失败: ${p.error}`);
                if (currentTask.value?.taskId === p.task_id) {
                    currentTask.value = null;
                }
            }),
        ]);
        unlistenTaskProgress = u1;
        unlistenTaskCompleted = u2;
        unlistenTaskFailed = u3;
    }
    function stopListeners() {
        unlistenProgress?.();
        unlistenCompleted?.();
        unlistenImageDiscovered?.();
        unlistenProgress = null;
        unlistenCompleted = null;
        unlistenImageDiscovered = null;
    }
    function stopTaskListeners() {
        unlistenTaskProgress?.();
        unlistenTaskCompleted?.();
        unlistenTaskFailed?.();
        unlistenTaskProgress = null;
        unlistenTaskCompleted = null;
        unlistenTaskFailed = null;
    }
    function startPolling() {
        if (pollTimer)
            return;
        pollTimer = setInterval(() => {
            void refreshStatus();
        }, getScanPollMs());
    }
    function restartPolling() {
        stopPolling();
        startPolling();
    }
    function getScanPollMs() {
        try {
            return useSettingsStore().settings.scanPollMs;
        }
        catch {
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
        }
        catch {
            // ignore polling errors
        }
    }
    // ---- Actions ----
    function addFolder(path) {
        if (path && !folders.value.includes(path)) {
            folders.value.push(path);
            appendLog('info', `已添加文件夹: ${path}`);
        }
    }
    function removeFolder(path) {
        const idx = folders.value.indexOf(path);
        if (idx >= 0) {
            folders.value.splice(idx, 1);
            if (currentFolderIndex.value > idx)
                currentFolderIndex.value--;
            appendLog('info', `已移除文件夹: ${path}`);
        }
    }
    function clearLog() {
        logEntries.value = [];
    }
    /** 扫描单个文件夹（不进入队列）。
     * discovery 完成后不停止轮询，等待调用方处理后续索引阶段。 */
    async function scan(path) {
        isScanning.value = true;
        currentPath.value = path;
        currentFile.value = '';
        candidateCount.value = 0;
        scanned.value = 0;
        lastResult.value = null;
        lastError.value = null;
        discoveredImages.value = []; // 清空上次的发现记录
        await startListeners(); // 必须等待 listener 注册完成
        await startTaskListeners();
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
        }
        catch (e) {
            lastError.value = String(e);
            appendLog('error', `${path}: ${e}`);
            throw e;
        }
        // 注意：不在这里停止轮询，由 scanAll 或调用方负责
    }
    /** 顺序扫描整个文件夹队列 + 追踪索引阶段直到全部完成。
     * 扫描模态框应基于 scanPhase !== 'idle' 显示。
     * 注意：scanFolder 现在是非阻塞的，需要等待 pf:scan:completed 事件才知道每个文件夹扫描完成。*/
    async function scanAll() {
        if (folders.value.length === 0) {
            appendLog('warn', '文件夹队列为空');
            return;
        }
        isScanning.value = true;
        totalFolders.value = folders.value.length;
        allDiscoveryDone.value = false;
        scanPhase.value = 'discovering';
        discoveredImages.value = []; // 清空上次的发现记录
        completedTasks.value = []; // 清空上次的任务记录
        currentTask.value = null;
        currentPath.value = '';
        currentFile.value = '';
        candidateCount.value = 0;
        scanned.value = 0;
        lastResult.value = null;
        lastError.value = null;
        await startListeners();
        await startTaskListeners();
        startPolling();
        appendLog('info', `开始扫描 ${folders.value.length} 个文件夹`);
        // 跟踪已完成扫描的文件夹数
        let completedCount = 0;
        // 等待扫描完成事件（最多等5分钟）
        const scanDonePromise = Promise.race([
            new Promise((resolve) => {
                const checkDone = setInterval(() => {
                    if (completedCount >= folders.value.length) {
                        clearInterval(checkDone);
                        resolve();
                    }
                }, 200);
            }),
            new Promise((_, reject) => setTimeout(() => { reject(new Error('扫描超时（5分钟）')); }, 5 * 60 * 1000)),
        ]);
        // 监听扫描完成事件
        const unlistenDone = await onScanCompleted((p) => {
            completedCount++;
            if (p.error) {
                appendLog('error', `扫描 ${p.path} 失败: ${p.error}`);
            }
            else {
                appendLog('success', `扫描 ${p.path} 完成`);
            }
        });
        try {
            // 对每个文件夹调用 scanFolder（非阻塞，立即返回）
            for (let i = 0; i < folders.value.length; i++) {
                currentFolderIndex.value = i;
                const path = folders.value[i];
                if (!path)
                    continue;
                // scanFolder 立即返回，扫描在后台运行
                void scanFolder(path);
            }
            // 等待所有文件夹扫描完成
            await scanDonePromise;
            // 所有文件夹发现完毕，进入索引追踪阶段
            allDiscoveryDone.value = true;
            scanPhase.value = 'indexing';
            appendLog('info', '文件夹扫描完成，开始处理索引任务…');
            // 继续轮询直到 pending_tasks === 0
            startIndexPoll();
            // 等待索引全部完成
            await new Promise((resolve) => {
                const checkDone = setInterval(() => {
                    if (scanPhase.value === 'done') {
                        clearInterval(checkDone);
                        resolve();
                    }
                }, 500);
            });
            appendLog('success', '所有文件夹扫描完成');
        }
        finally {
            unlistenDone();
            isScanning.value = false;
            currentFolderIndex.value = -1;
            stopPolling();
            stopTaskListeners();
            stopIndexPoll();
            scanPhase.value = 'idle';
        }
    }
    async function stop() {
        await stopScan();
        isScanning.value = false;
        allDiscoveryDone.value = true;
        scanPhase.value = 'done';
        stopPolling();
        stopTaskListeners();
        stopIndexPoll();
        appendLog('warn', '扫描已停止');
    }
    const progress = () => {
        if (candidateCount.value === 0)
            return 0;
        return Math.min(1, scanned.value / candidateCount.value);
    };
    onScopeDispose(() => {
        stopListeners();
        stopTaskListeners();
        stopPolling();
    });
    // 用户改了 polling 间隔 / log buffer → 立即生效
    watch(() => {
        try {
            return useSettingsStore().settings.scanPollMs;
        }
        catch {
            return 1000;
        }
    }, () => restartPolling());
    watch(() => {
        try {
            return useSettingsStore().settings.logBufferSize;
        }
        catch {
            return MAX_LOG_ENTRIES_DEFAULT;
        }
    }, (n) => {
        if (logEntries.value.length > n) {
            logEntries.value = logEntries.value.slice(-n);
        }
    });
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
        currentTask,
        completedTasks,
        scanPhase,
        totalFolders,
        allDiscoveryDone,
    };
});
