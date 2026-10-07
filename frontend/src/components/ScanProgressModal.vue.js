import { computed } from 'vue';
import { useScanStore } from '@/stores/scan';
import ProgressBar from '@/components/ProgressBar.vue';
const scan = useScanStore();
const isOpen = computed(() => scan.scanPhase !== 'idle');
const isDiscovering = computed(() => scan.scanPhase === 'discovering');
const isIndexing = computed(() => scan.scanPhase === 'indexing');
const isDone = computed(() => scan.scanPhase === 'done');
// 发现进度（文件夹扫描阶段）
const discoveryProgress = computed(() => {
    if (scan.scanStatus.total_images === 0)
        return 0;
    return scan.scanStatus.processed_images / scan.scanStatus.total_images;
});
// 索引进度：已完成任务数 / (已完成 + 待处理任务数)
const indexProgress = computed(() => {
    const completed = scan.completedTasks.length;
    const pending = scan.scanStatus.pending_tasks;
    const total = completed + pending;
    if (total === 0)
        return 0;
    return completed / total;
});
// 总任务数（已完成 + 待处理）
const totalTasks = computed(() => scan.completedTasks.length + scan.scanStatus.pending_tasks);
// 阶段描述
const phaseLabel = computed(() => {
    if (isDiscovering.value)
        return '正在扫描文件夹…';
    if (isIndexing.value)
        return '正在处理索引任务…';
    if (isDone.value)
        return '扫描完成！';
    return '';
});
// 当前步骤描述
const stepHint = computed(() => {
    if (isDiscovering.value) {
        if (scan.scanStatus.current_file) {
            return `处理: ${scan.scanStatus.current_file}`;
        }
        return `已入库 ${scan.discoveredImages.length} 张`;
    }
    if (isIndexing.value) {
        if (scan.currentTask) {
            return `${scan.currentTask.kindLabel} — ${scan.currentTask.message}`;
        }
        return `已处理 ${scan.completedTasks.length} 个任务`;
    }
    if (isDone.value) {
        return `共入库 ${scan.discoveredImages.length} 张`;
    }
    return '';
});
// 取消按钮
function onCancel() {
    void scan.stop();
}
debugger; /* PartiallyEnd: #3632/scriptSetup.vue */
const __VLS_ctx = {};
let __VLS_components;
let __VLS_directives;
/** @type {__VLS_StyleScopedClasses['scan-modal-close']} */ ;
// CSS variable injection 
// CSS variable injection end 
const __VLS_0 = {}.Teleport;
/** @type {[typeof __VLS_components.Teleport, typeof __VLS_components.Teleport, ]} */ ;
// @ts-ignore
const __VLS_1 = __VLS_asFunctionalComponent(__VLS_0, new __VLS_0({
    to: "body",
}));
const __VLS_2 = __VLS_1({
    to: "body",
}, ...__VLS_functionalComponentArgsRest(__VLS_1));
__VLS_3.slots.default;
if (__VLS_ctx.isOpen) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "scan-modal-backdrop" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "scan-modal" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "scan-modal-header" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "scan-modal-title" },
    });
    if (__VLS_ctx.isDone) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ onClick: (...[$event]) => {
                    if (!(__VLS_ctx.isOpen))
                        return;
                    if (!(__VLS_ctx.isDone))
                        return;
                    __VLS_ctx.scan.scanPhase = 'idle';
                } },
            ...{ class: "scan-modal-close" },
        });
    }
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "scan-phase-label" },
    });
    (__VLS_ctx.phaseLabel);
    if (__VLS_ctx.isDiscovering) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "scan-modal-section" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "scan-modal-hint" },
        });
        (__VLS_ctx.stepHint);
        /** @type {[typeof ProgressBar, ]} */ ;
        // @ts-ignore
        const __VLS_4 = __VLS_asFunctionalComponent(ProgressBar, new ProgressBar({
            value: (__VLS_ctx.discoveryProgress),
            hint: (`${__VLS_ctx.scan.scanStatus.processed_images} / ${__VLS_ctx.scan.scanStatus.total_images} 文件`),
        }));
        const __VLS_5 = __VLS_4({
            value: (__VLS_ctx.discoveryProgress),
            hint: (`${__VLS_ctx.scan.scanStatus.processed_images} / ${__VLS_ctx.scan.scanStatus.total_images} 文件`),
        }, ...__VLS_functionalComponentArgsRest(__VLS_4));
        if (__VLS_ctx.scan.scanStatus.pending_tasks > 0) {
            __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
                ...{ class: "scan-modal-sub" },
            });
            (__VLS_ctx.scan.scanStatus.pending_tasks);
        }
    }
    if (__VLS_ctx.isIndexing) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "scan-modal-section" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "scan-modal-hint" },
        });
        (__VLS_ctx.stepHint);
        /** @type {[typeof ProgressBar, ]} */ ;
        // @ts-ignore
        const __VLS_7 = __VLS_asFunctionalComponent(ProgressBar, new ProgressBar({
            value: (__VLS_ctx.indexProgress),
            hint: (`${__VLS_ctx.scan.completedTasks.length} / ${__VLS_ctx.totalTasks} 任务`),
        }));
        const __VLS_8 = __VLS_7({
            value: (__VLS_ctx.indexProgress),
            hint: (`${__VLS_ctx.scan.completedTasks.length} / ${__VLS_ctx.totalTasks} 任务`),
        }, ...__VLS_functionalComponentArgsRest(__VLS_7));
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "scan-modal-sub" },
        });
        (__VLS_ctx.scan.completedTasks.length);
        if (__VLS_ctx.scan.scanStatus.pending_tasks > 0) {
            __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
            (__VLS_ctx.scan.scanStatus.pending_tasks);
        }
    }
    if (__VLS_ctx.isDone) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "scan-modal-section done-section" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "scan-done-icon" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "scan-done-text" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "scan-done-summary" },
        });
        (__VLS_ctx.scan.discoveredImages.length);
        __VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
            ...{ onClick: (...[$event]) => {
                    if (!(__VLS_ctx.isOpen))
                        return;
                    if (!(__VLS_ctx.isDone))
                        return;
                    __VLS_ctx.scan.scanPhase = 'idle';
                } },
            ...{ class: "btn primary" },
        });
    }
    if (!__VLS_ctx.isDone) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "scan-modal-footer" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
            ...{ onClick: (__VLS_ctx.onCancel) },
            ...{ class: "btn ghost" },
        });
    }
}
var __VLS_3;
/** @type {__VLS_StyleScopedClasses['scan-modal-backdrop']} */ ;
/** @type {__VLS_StyleScopedClasses['scan-modal']} */ ;
/** @type {__VLS_StyleScopedClasses['scan-modal-header']} */ ;
/** @type {__VLS_StyleScopedClasses['scan-modal-title']} */ ;
/** @type {__VLS_StyleScopedClasses['scan-modal-close']} */ ;
/** @type {__VLS_StyleScopedClasses['scan-phase-label']} */ ;
/** @type {__VLS_StyleScopedClasses['scan-modal-section']} */ ;
/** @type {__VLS_StyleScopedClasses['scan-modal-hint']} */ ;
/** @type {__VLS_StyleScopedClasses['scan-modal-sub']} */ ;
/** @type {__VLS_StyleScopedClasses['scan-modal-section']} */ ;
/** @type {__VLS_StyleScopedClasses['scan-modal-hint']} */ ;
/** @type {__VLS_StyleScopedClasses['scan-modal-sub']} */ ;
/** @type {__VLS_StyleScopedClasses['scan-modal-section']} */ ;
/** @type {__VLS_StyleScopedClasses['done-section']} */ ;
/** @type {__VLS_StyleScopedClasses['scan-done-icon']} */ ;
/** @type {__VLS_StyleScopedClasses['scan-done-text']} */ ;
/** @type {__VLS_StyleScopedClasses['scan-done-summary']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['primary']} */ ;
/** @type {__VLS_StyleScopedClasses['scan-modal-footer']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['ghost']} */ ;
var __VLS_dollars;
const __VLS_self = (await import('vue')).defineComponent({
    setup() {
        return {
            ProgressBar: ProgressBar,
            scan: scan,
            isOpen: isOpen,
            isDiscovering: isDiscovering,
            isIndexing: isIndexing,
            isDone: isDone,
            discoveryProgress: discoveryProgress,
            indexProgress: indexProgress,
            totalTasks: totalTasks,
            phaseLabel: phaseLabel,
            stepHint: stepHint,
            onCancel: onCancel,
        };
    },
});
export default (await import('vue')).defineComponent({
    setup() {
        return {};
    },
});
; /* PartiallyEnd: #4569/main.vue */
