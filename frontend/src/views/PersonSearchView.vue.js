import { onMounted, ref } from 'vue';
import { useRouter } from 'vue-router';
import { useSearchStore } from '@/stores/search';
import { useSettingsStore } from '@/stores/settings';
import { useToastStore } from '@/stores/toast';
import ImageGrid from '@/components/ImageGrid.vue';
import EmptyState from '@/components/EmptyState.vue';
import ExportBar from '@/components/ExportBar.vue';
const router = useRouter();
const search = useSearchStore();
const settings = useSettingsStore();
const toast = useToastStore();
const fileInput = ref(null);
const previewUrl = ref(null);
const topK = ref(settings.settings.topK);
onMounted(() => search.clear());
async function pickFile() {
    fileInput.value?.click();
}
async function onFile(e) {
    const input = e.target;
    const file = input.files?.[0];
    if (!file)
        return;
    await runWith(file);
    input.value = '';
}
function isLicenseError(e) {
    const msg = String(e).toLowerCase();
    return msg.includes('not_logged_in') ||
        msg.includes('not_activated') ||
        msg.includes('license') ||
        msg.includes('expired');
}
async function runWith(file) {
    try {
        const buf = new Uint8Array(await file.arrayBuffer());
        previewUrl.value = URL.createObjectURL(file);
        await search.searchByImage(buf, topK.value);
        if (search.hits.length === 0) {
            toast.info('未找到相似人脸（库中可能还没有人脸数据）');
        }
        else {
            toast.success(`找到 ${search.hits.length} 个相似结果`);
        }
    }
    catch (e) {
        if (isLicenseError(e)) {
            toast.warning('会员未激活，请先兑换会员');
            router.push('/membership');
        }
        else {
            toast.danger('检索失败: ' + String(e));
        }
    }
}
const dragOver = ref(false);
function onDragOver(e) {
    e.preventDefault();
    dragOver.value = true;
}
function onDragLeave() {
    dragOver.value = false;
}
async function onDrop(e) {
    e.preventDefault();
    dragOver.value = false;
    const file = e.dataTransfer?.files?.[0];
    if (file && file.type.startsWith('image/')) {
        await runWith(file);
    }
    else {
        toast.warning('请拖入图片文件');
    }
}
debugger; /* PartiallyEnd: #3632/scriptSetup.vue */
const __VLS_ctx = {};
let __VLS_components;
let __VLS_directives;
/** @type {__VLS_StyleScopedClasses['dropzone']} */ ;
/** @type {__VLS_StyleScopedClasses['dropzone']} */ ;
// CSS variable injection 
// CSS variable injection end 
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "view" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.header, __VLS_intrinsicElements.header)({
    ...{ class: "view-header" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.h1, __VLS_intrinsicElements.h1)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.section, __VLS_intrinsicElements.section)({
    ...{ class: "card" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "card-header" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "card-title" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ onClick: (__VLS_ctx.pickFile) },
    ...{ onDragover: (__VLS_ctx.onDragOver) },
    ...{ onDragleave: (__VLS_ctx.onDragLeave) },
    ...{ onDrop: (__VLS_ctx.onDrop) },
    ...{ class: (['dropzone', { over: __VLS_ctx.dragOver }]) },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.input)({
    ...{ onChange: (__VLS_ctx.onFile) },
    ref: "fileInput",
    type: "file",
    accept: "image/*",
    ...{ style: {} },
});
/** @type {typeof __VLS_ctx.fileInput} */ ;
if (__VLS_ctx.previewUrl) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.img)({
        src: (__VLS_ctx.previewUrl),
        ...{ class: "preview" },
    });
}
else {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "dropzone-hint" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "text-lg" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "muted text-sm" },
    });
}
__VLS_asFunctionalElement(__VLS_intrinsicElements.section, __VLS_intrinsicElements.section)({
    ...{ class: "card" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "card-header" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "card-title" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "muted text-sm" },
});
(__VLS_ctx.search.isSearching ? '检索中…' : `${__VLS_ctx.search.hits.length} 个结果`);
if (__VLS_ctx.search.hits.length > 0) {
    /** @type {[typeof ExportBar, ]} */ ;
    // @ts-ignore
    const __VLS_0 = __VLS_asFunctionalComponent(ExportBar, new ExportBar({}));
    const __VLS_1 = __VLS_0({}, ...__VLS_functionalComponentArgsRest(__VLS_0));
}
if (__VLS_ctx.search.hits.length > 0) {
    /** @type {[typeof ImageGrid, ]} */ ;
    // @ts-ignore
    const __VLS_3 = __VLS_asFunctionalComponent(ImageGrid, new ImageGrid({
        hits: (__VLS_ctx.search.hits),
    }));
    const __VLS_4 = __VLS_3({
        hits: (__VLS_ctx.search.hits),
    }, ...__VLS_functionalComponentArgsRest(__VLS_3));
}
else {
    /** @type {[typeof EmptyState, ]} */ ;
    // @ts-ignore
    const __VLS_6 = __VLS_asFunctionalComponent(EmptyState, new EmptyState({
        title: "还没有结果",
        hint: "上传一张图片开始检索",
    }));
    const __VLS_7 = __VLS_6({
        title: "还没有结果",
        hint: "上传一张图片开始检索",
    }, ...__VLS_functionalComponentArgsRest(__VLS_6));
}
/** @type {__VLS_StyleScopedClasses['view']} */ ;
/** @type {__VLS_StyleScopedClasses['view-header']} */ ;
/** @type {__VLS_StyleScopedClasses['card']} */ ;
/** @type {__VLS_StyleScopedClasses['card-header']} */ ;
/** @type {__VLS_StyleScopedClasses['card-title']} */ ;
/** @type {__VLS_StyleScopedClasses['preview']} */ ;
/** @type {__VLS_StyleScopedClasses['dropzone-hint']} */ ;
/** @type {__VLS_StyleScopedClasses['text-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['muted']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['card']} */ ;
/** @type {__VLS_StyleScopedClasses['card-header']} */ ;
/** @type {__VLS_StyleScopedClasses['card-title']} */ ;
/** @type {__VLS_StyleScopedClasses['muted']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
var __VLS_dollars;
const __VLS_self = (await import('vue')).defineComponent({
    setup() {
        return {
            ImageGrid: ImageGrid,
            EmptyState: EmptyState,
            ExportBar: ExportBar,
            search: search,
            fileInput: fileInput,
            previewUrl: previewUrl,
            pickFile: pickFile,
            onFile: onFile,
            dragOver: dragOver,
            onDragOver: onDragOver,
            onDragLeave: onDragLeave,
            onDrop: onDrop,
        };
    },
});
export default (await import('vue')).defineComponent({
    setup() {
        return {};
    },
});
; /* PartiallyEnd: #4569/main.vue */
