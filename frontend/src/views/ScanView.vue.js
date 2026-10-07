import { computed, onMounted, onScopeDispose, ref, watch } from 'vue';
import { open, ask } from '@tauri-apps/plugin-dialog';
import { useScanStore } from '@/stores/scan';
import { useAppStore } from '@/stores/app';
import { useToastStore } from '@/stores/toast';
import ScanLog from '@/components/ScanLog.vue';
import ScanProgressModal from '@/components/ScanProgressModal.vue';
import { listLibrary } from '@/api';
const scan = useScanStore();
const app = useAppStore();
const toast = useToastStore();
const newPath = ref('');
const busy = computed(() => scan.isScanning);
// 图片列表（最近50张）
const images = ref([]);
const loadingImages = ref(false);
// 扫描期间轮询图片列表
let imagePollTimer = null;
function startImagePolling() {
    if (imagePollTimer)
        return;
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
    }
    catch (e) {
        console.error('loadImages failed', e);
    }
    finally {
        loadingImages.value = false;
    }
}
// 扫描状态变化时自动启停轮询
watch(busy, (isBusy) => {
    if (isBusy) {
        startImagePolling();
    }
    else {
        stopImagePolling();
    }
});
function imageName(path) {
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
    }
    catch (e) {
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
function removeOne(path) {
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
    // 确认对话框，防止用户误触
    const confirmed = await ask(`即将扫描 ${scan.folders.length} 个目录。\n扫描将清除现有数据并重新索引。\n是否继续？`, { title: '确认扫描', kind: 'warning' });
    if (!confirmed) {
        return;
    }
    try {
        await scan.scanAll();
        await app.refreshStats();
        await loadImages(); // 刷新图片列表（显示 face/object counts）
        toast.success('所有目录扫描完成');
    }
    catch (e) {
        toast.danger('扫描失败: ' + String(e));
    }
}
async function stopScan() {
    const confirmed = await ask('确定要取消扫描吗？当前扫描进度将会丢失。', {
        title: '确认取消',
        kind: 'warning'
    });
    if (!confirmed) {
        return;
    }
    try {
        await scan.stop();
        toast.info('已请求停止扫描');
    }
    catch (e) {
        toast.danger('停止失败: ' + String(e));
    }
}
// 监听正在扫描的文件变化，实时写入扫描日志（避免重复日志）
let prevCurrentFile = '';
const stopWatchCurrentFile = watch(() => scan.currentFile, (file) => {
    if (file && file !== prevCurrentFile && scan.isScanning) {
        prevCurrentFile = file;
        // 提取文件名用于日志
        const fileName = file.split('/').pop() ?? file;
        scan.appendLog('info', `扫描: ${fileName}`);
    }
    if (!file) {
        prevCurrentFile = '';
    }
});
onScopeDispose(() => {
    stopWatchCurrentFile();
    stopImagePolling();
});
onMounted(() => { app.refresh(); loadImages(); });
debugger; /* PartiallyEnd: #3632/scriptSetup.vue */
const __VLS_ctx = {};
let __VLS_components;
let __VLS_directives;
/** @type {__VLS_StyleScopedClasses['folder-row']} */ ;
/** @type {__VLS_StyleScopedClasses['image-table']} */ ;
/** @type {__VLS_StyleScopedClasses['image-table']} */ ;
/** @type {__VLS_StyleScopedClasses['image-table']} */ ;
// CSS variable injection 
// CSS variable injection end 
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "view" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.header, __VLS_intrinsicElements.header)({
    ...{ class: "view-header" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.h1, __VLS_intrinsicElements.h1)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
    ...{ class: "muted text-sm" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.section, __VLS_intrinsicElements.section)({
    ...{ class: "card" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "card-header" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "card-title" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "card-subtitle" },
});
(__VLS_ctx.scan.folders.length);
if (__VLS_ctx.scan.currentFolderIndex >= 0) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
        ...{ class: "muted" },
    });
    (__VLS_ctx.scan.currentFolderIndex + 1);
}
if (__VLS_ctx.scan.folders.length && !__VLS_ctx.busy) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
        ...{ onClick: (__VLS_ctx.clearQueue) },
        ...{ class: "btn sm ghost" },
    });
}
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "row gap-2" },
    ...{ style: {} },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.input)({
    ...{ onKeydown: (__VLS_ctx.addFromInput) },
    ...{ class: "input flex-1" },
    placeholder: "/Users/me/Pictures",
    disabled: (__VLS_ctx.busy),
});
(__VLS_ctx.newPath);
__VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
    ...{ onClick: (__VLS_ctx.addFromPicker) },
    ...{ class: "btn ghost" },
    disabled: (__VLS_ctx.busy),
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
    ...{ onClick: (__VLS_ctx.addFromInput) },
    ...{ class: "btn ghost" },
    disabled: (__VLS_ctx.busy || !__VLS_ctx.newPath.trim()),
});
if (__VLS_ctx.scan.folders.length) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "folder-list" },
    });
    for (const [path, idx] of __VLS_getVForSourceType((__VLS_ctx.scan.folders))) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            key: (path),
            ...{ class: (['folder-row', { active: idx === __VLS_ctx.scan.currentFolderIndex }]) },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
            ...{ class: "folder-idx font-mono" },
        });
        (idx + 1);
        __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
            ...{ class: "folder-path font-mono" },
            title: (path),
        });
        (path);
        __VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
            ...{ onClick: (...[$event]) => {
                    if (!(__VLS_ctx.scan.folders.length))
                        return;
                    __VLS_ctx.removeOne(path);
                } },
            ...{ class: "btn sm ghost" },
            disabled: (__VLS_ctx.busy),
        });
    }
}
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "row gap-3" },
    ...{ style: {} },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
    ...{ onClick: (__VLS_ctx.startScanAll) },
    ...{ class: "btn primary" },
    disabled: (__VLS_ctx.busy || __VLS_ctx.scan.folders.length === 0),
});
(__VLS_ctx.busy ? '扫描中…' : '扫描全部');
__VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
    ...{ onClick: (__VLS_ctx.stopScan) },
    ...{ class: "btn ghost" },
    disabled: (!__VLS_ctx.busy),
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.section, __VLS_intrinsicElements.section)({
    ...{ class: "card" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "card-header" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "card-title" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "card-subtitle text-sm muted" },
});
if (__VLS_ctx.busy && __VLS_ctx.scan.discoveredImages.length > 0) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
    (__VLS_ctx.scan.discoveredImages.length);
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
        ...{ class: "text-accent" },
    });
}
else {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
    (__VLS_ctx.images.length);
    if (__VLS_ctx.loadingImages) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
    }
}
__VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
    ...{ onClick: (__VLS_ctx.loadImages) },
    ...{ class: "btn sm ghost" },
    disabled: (__VLS_ctx.loadingImages),
});
if (__VLS_ctx.busy && __VLS_ctx.scan.discoveredImages.length > 0) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "scan-progress-list" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "scan-progress-header" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
        ...{ class: "text-sm muted" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
        ...{ class: "text-sm font-mono" },
    });
    (__VLS_ctx.scan.discoveredImages.length);
    (__VLS_ctx.scan.candidateCount || '?');
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "scan-progress-bar-wrap" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div)({
        ...{ class: "scan-progress-bar-fill" },
        ...{ style: ({ width: __VLS_ctx.scan.candidateCount > 0 ? `${Math.min(100, (__VLS_ctx.scan.discoveredImages.length / __VLS_ctx.scan.candidateCount) * 100)}%` : '0%' }) },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "scan-image-grid" },
    });
    for (const [img] of __VLS_getVForSourceType((__VLS_ctx.scan.discoveredImages.slice(-24)))) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            key: (img.image_id),
            ...{ class: "scan-image-item" },
            title: (img.path),
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "scan-image-name" },
        });
        (img.path.split('/').pop());
    }
    if (__VLS_ctx.scan.discoveredImages.length > 24) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "text-xs muted" },
        });
        (__VLS_ctx.scan.discoveredImages.length - 24);
    }
}
else if (__VLS_ctx.loadingImages && __VLS_ctx.images.length === 0) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "text-sm muted" },
        ...{ style: {} },
    });
}
else if (__VLS_ctx.images.length === 0) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "text-sm muted" },
        ...{ style: {} },
    });
}
else {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "image-table-wrap" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.table, __VLS_intrinsicElements.table)({
        ...{ class: "image-table" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.thead, __VLS_intrinsicElements.thead)({});
    __VLS_asFunctionalElement(__VLS_intrinsicElements.tr, __VLS_intrinsicElements.tr)({});
    __VLS_asFunctionalElement(__VLS_intrinsicElements.th, __VLS_intrinsicElements.th)({});
    __VLS_asFunctionalElement(__VLS_intrinsicElements.th, __VLS_intrinsicElements.th)({});
    __VLS_asFunctionalElement(__VLS_intrinsicElements.th, __VLS_intrinsicElements.th)({});
    __VLS_asFunctionalElement(__VLS_intrinsicElements.th, __VLS_intrinsicElements.th)({});
    __VLS_asFunctionalElement(__VLS_intrinsicElements.tbody, __VLS_intrinsicElements.tbody)({});
    for (const [img] of __VLS_getVForSourceType((__VLS_ctx.images))) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.tr, __VLS_intrinsicElements.tr)({
            key: (img.id),
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.td, __VLS_intrinsicElements.td)({
            ...{ class: "img-name" },
            title: (img.path),
        });
        (__VLS_ctx.imageName(img.path));
        __VLS_asFunctionalElement(__VLS_intrinsicElements.td, __VLS_intrinsicElements.td)({});
        __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
            ...{ class: (['tag', img.scan_status === 'indexed' ? 'success' : 'muted']) },
        });
        (img.scan_status);
        __VLS_asFunctionalElement(__VLS_intrinsicElements.td, __VLS_intrinsicElements.td)({
            ...{ class: "num" },
        });
        if (img.face_count > 0) {
            __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
                ...{ class: "face-count" },
            });
            (img.face_count);
        }
        else {
            __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
                ...{ class: "muted" },
            });
        }
        __VLS_asFunctionalElement(__VLS_intrinsicElements.td, __VLS_intrinsicElements.td)({
            ...{ class: "num" },
        });
        if (img.object_count > 0) {
            __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
                ...{ class: "obj-count" },
            });
            (img.object_count);
        }
        else {
            __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
                ...{ class: "muted" },
            });
        }
    }
}
__VLS_asFunctionalElement(__VLS_intrinsicElements.section, __VLS_intrinsicElements.section)({
    ...{ class: "card" },
});
/** @type {[typeof ScanLog, ]} */ ;
// @ts-ignore
const __VLS_0 = __VLS_asFunctionalComponent(ScanLog, new ScanLog({}));
const __VLS_1 = __VLS_0({}, ...__VLS_functionalComponentArgsRest(__VLS_0));
/** @type {[typeof ScanProgressModal, ]} */ ;
// @ts-ignore
const __VLS_3 = __VLS_asFunctionalComponent(ScanProgressModal, new ScanProgressModal({}));
const __VLS_4 = __VLS_3({}, ...__VLS_functionalComponentArgsRest(__VLS_3));
/** @type {__VLS_StyleScopedClasses['view']} */ ;
/** @type {__VLS_StyleScopedClasses['view-header']} */ ;
/** @type {__VLS_StyleScopedClasses['muted']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['card']} */ ;
/** @type {__VLS_StyleScopedClasses['card-header']} */ ;
/** @type {__VLS_StyleScopedClasses['card-title']} */ ;
/** @type {__VLS_StyleScopedClasses['card-subtitle']} */ ;
/** @type {__VLS_StyleScopedClasses['muted']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['sm']} */ ;
/** @type {__VLS_StyleScopedClasses['ghost']} */ ;
/** @type {__VLS_StyleScopedClasses['row']} */ ;
/** @type {__VLS_StyleScopedClasses['gap-2']} */ ;
/** @type {__VLS_StyleScopedClasses['input']} */ ;
/** @type {__VLS_StyleScopedClasses['flex-1']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['ghost']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['ghost']} */ ;
/** @type {__VLS_StyleScopedClasses['folder-list']} */ ;
/** @type {__VLS_StyleScopedClasses['folder-idx']} */ ;
/** @type {__VLS_StyleScopedClasses['font-mono']} */ ;
/** @type {__VLS_StyleScopedClasses['folder-path']} */ ;
/** @type {__VLS_StyleScopedClasses['font-mono']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['sm']} */ ;
/** @type {__VLS_StyleScopedClasses['ghost']} */ ;
/** @type {__VLS_StyleScopedClasses['row']} */ ;
/** @type {__VLS_StyleScopedClasses['gap-3']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['primary']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['ghost']} */ ;
/** @type {__VLS_StyleScopedClasses['card']} */ ;
/** @type {__VLS_StyleScopedClasses['card-header']} */ ;
/** @type {__VLS_StyleScopedClasses['card-title']} */ ;
/** @type {__VLS_StyleScopedClasses['card-subtitle']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['muted']} */ ;
/** @type {__VLS_StyleScopedClasses['text-accent']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['sm']} */ ;
/** @type {__VLS_StyleScopedClasses['ghost']} */ ;
/** @type {__VLS_StyleScopedClasses['scan-progress-list']} */ ;
/** @type {__VLS_StyleScopedClasses['scan-progress-header']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['muted']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['font-mono']} */ ;
/** @type {__VLS_StyleScopedClasses['scan-progress-bar-wrap']} */ ;
/** @type {__VLS_StyleScopedClasses['scan-progress-bar-fill']} */ ;
/** @type {__VLS_StyleScopedClasses['scan-image-grid']} */ ;
/** @type {__VLS_StyleScopedClasses['scan-image-item']} */ ;
/** @type {__VLS_StyleScopedClasses['scan-image-name']} */ ;
/** @type {__VLS_StyleScopedClasses['text-xs']} */ ;
/** @type {__VLS_StyleScopedClasses['muted']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['muted']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['muted']} */ ;
/** @type {__VLS_StyleScopedClasses['image-table-wrap']} */ ;
/** @type {__VLS_StyleScopedClasses['image-table']} */ ;
/** @type {__VLS_StyleScopedClasses['img-name']} */ ;
/** @type {__VLS_StyleScopedClasses['num']} */ ;
/** @type {__VLS_StyleScopedClasses['face-count']} */ ;
/** @type {__VLS_StyleScopedClasses['muted']} */ ;
/** @type {__VLS_StyleScopedClasses['num']} */ ;
/** @type {__VLS_StyleScopedClasses['obj-count']} */ ;
/** @type {__VLS_StyleScopedClasses['muted']} */ ;
/** @type {__VLS_StyleScopedClasses['card']} */ ;
var __VLS_dollars;
const __VLS_self = (await import('vue')).defineComponent({
    setup() {
        return {
            ScanLog: ScanLog,
            ScanProgressModal: ScanProgressModal,
            scan: scan,
            newPath: newPath,
            busy: busy,
            images: images,
            loadingImages: loadingImages,
            loadImages: loadImages,
            imageName: imageName,
            addFromInput: addFromInput,
            addFromPicker: addFromPicker,
            removeOne: removeOne,
            clearQueue: clearQueue,
            startScanAll: startScanAll,
            stopScan: stopScan,
        };
    },
});
export default (await import('vue')).defineComponent({
    setup() {
        return {};
    },
});
; /* PartiallyEnd: #4569/main.vue */
