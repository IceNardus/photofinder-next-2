import { computed, ref } from 'vue';
import { open as openDialog } from '@tauri-apps/plugin-dialog';
import { useSearchStore } from '@/stores/search';
import { useToastStore } from '@/stores/toast';
const search = useSearchStore();
const toast = useToastStore();
const exporting = ref(false);
const selectedCount = computed(() => search.selectedIds.size);
const totalCount = computed(() => search.hits.length);
const pageSelectedCount = computed(() => search.pageHits.filter((h) => search.isSelected(h.image_id)).length);
function selectAll() {
    search.selectAll();
}
function unselectAll() {
    search.clearSelection();
}
function selectPage() {
    search.selectPage();
}
function unselectPage() {
    search.unselectPage();
}
async function pickAndExport() {
    if (selectedCount.value === 0) {
        toast.warning('请先选中要导出的图片');
        return;
    }
    let dir = null;
    try {
        const r = await openDialog({ directory: true, multiple: false, title: '选择导出目录' });
        if (typeof r === 'string')
            dir = r;
    }
    catch (e) {
        toast.danger('打开目录选择失败: ' + String(e));
        return;
    }
    if (!dir)
        return;
    exporting.value = true;
    try {
        const results = await search.exportSelected(dir);
        const ok = results.filter((r) => r.success).length;
        const fail = results.length - ok;
        if (fail === 0) {
            toast.success(`已导出 ${ok} 个文件到 ${dir}`);
        }
        else {
            toast.warning(`导出 ${ok} 个，失败 ${fail} 个`);
        }
    }
    catch (e) {
        toast.danger('导出失败: ' + String(e));
    }
    finally {
        exporting.value = false;
    }
}
debugger; /* PartiallyEnd: #3632/scriptSetup.vue */
const __VLS_ctx = {};
let __VLS_components;
let __VLS_directives;
// CSS variable injection 
// CSS variable injection end 
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "export-bar" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "export-summary" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
    ...{ class: "text-sm muted" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
    ...{ class: "font-mono count" },
});
(__VLS_ctx.selectedCount);
(__VLS_ctx.totalCount);
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "export-actions" },
});
if (__VLS_ctx.pageSelectedCount > 0) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
        ...{ onClick: (__VLS_ctx.unselectPage) },
        ...{ class: "btn sm ghost" },
    });
}
else {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
        ...{ onClick: (__VLS_ctx.selectPage) },
        ...{ class: "btn sm ghost" },
        disabled: (__VLS_ctx.search.pageHits.length === 0),
    });
}
if (__VLS_ctx.selectedCount > 0) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
        ...{ onClick: (__VLS_ctx.unselectAll) },
        ...{ class: "btn sm ghost" },
    });
}
else {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
        ...{ onClick: (__VLS_ctx.selectAll) },
        ...{ class: "btn sm ghost" },
        disabled: (__VLS_ctx.totalCount === 0),
    });
}
__VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
    ...{ onClick: (__VLS_ctx.pickAndExport) },
    ...{ class: "btn sm primary" },
    disabled: (__VLS_ctx.selectedCount === 0 || __VLS_ctx.exporting),
});
(__VLS_ctx.exporting ? '导出中…' : `导出 (${__VLS_ctx.selectedCount})`);
/** @type {__VLS_StyleScopedClasses['export-bar']} */ ;
/** @type {__VLS_StyleScopedClasses['export-summary']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['muted']} */ ;
/** @type {__VLS_StyleScopedClasses['font-mono']} */ ;
/** @type {__VLS_StyleScopedClasses['count']} */ ;
/** @type {__VLS_StyleScopedClasses['export-actions']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['sm']} */ ;
/** @type {__VLS_StyleScopedClasses['ghost']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['sm']} */ ;
/** @type {__VLS_StyleScopedClasses['ghost']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['sm']} */ ;
/** @type {__VLS_StyleScopedClasses['ghost']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['sm']} */ ;
/** @type {__VLS_StyleScopedClasses['ghost']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['sm']} */ ;
/** @type {__VLS_StyleScopedClasses['primary']} */ ;
var __VLS_dollars;
const __VLS_self = (await import('vue')).defineComponent({
    setup() {
        return {
            search: search,
            exporting: exporting,
            selectedCount: selectedCount,
            totalCount: totalCount,
            pageSelectedCount: pageSelectedCount,
            selectAll: selectAll,
            unselectAll: unselectAll,
            selectPage: selectPage,
            unselectPage: unselectPage,
            pickAndExport: pickAndExport,
        };
    },
});
export default (await import('vue')).defineComponent({
    setup() {
        return {};
    },
});
; /* PartiallyEnd: #4569/main.vue */
