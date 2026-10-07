import { onMounted, ref } from 'vue';
import { useRouter } from 'vue-router';
import { useSearchStore } from '@/stores/search';
import { useSettingsStore } from '@/stores/settings';
import { useToastStore } from '@/stores/toast';
import ImageGrid from '@/components/ImageGrid.vue';
import EmptyState from '@/components/EmptyState.vue';
import ExportBar from '@/components/ExportBar.vue';
import CropModal from '@/components/CropModal.vue';
const router = useRouter();
const search = useSearchStore();
const settings = useSettingsStore();
const toast = useToastStore();
onMounted(() => search.clear());
function isLicenseError(e) {
    const msg = String(e).toLowerCase();
    return msg.includes('not_logged_in') ||
        msg.includes('not_activated') ||
        msg.includes('license') ||
        msg.includes('expired');
}
const limit = ref(settings.settings.topK);
// ===== 图片查询（crop modal）=====
const fileInput = ref(null);
const dragOver = ref(false);
const cropSrc = ref(null);
const croppedPreview = ref(null);
async function pickFile() {
    fileInput.value?.click();
}
async function onFile(e) {
    const input = e.target;
    const file = input.files?.[0];
    if (file)
        await openCrop(file);
    input.value = '';
}
async function onDrop(e) {
    e.preventDefault();
    dragOver.value = false;
    const file = e.dataTransfer?.files?.[0];
    if (file && file.type.startsWith('image/')) {
        await openCrop(file);
    }
    else {
        toast.warning('请拖入图片文件');
    }
}
function onDragOver(e) {
    e.preventDefault();
    dragOver.value = true;
}
function onDragLeave() {
    dragOver.value = false;
}
async function openCrop(file) {
    const url = URL.createObjectURL(file);
    const img = new Image();
    img.src = url;
    await new Promise((res, rej) => {
        img.onload = () => res();
        img.onerror = () => rej(new Error('load failed'));
    });
    cropSrc.value = {
        url,
        mime: file.type || 'image/png',
        width: img.naturalWidth,
        height: img.naturalHeight,
    };
}
function cancelCrop() {
    if (cropSrc.value) {
        URL.revokeObjectURL(cropSrc.value.url);
    }
    cropSrc.value = null;
}
async function confirmCrop(payload) {
    // 先显示预览，再关 modal + 搜索
    croppedPreview.value = payload.previewUrl;
    cancelCrop();
    try {
        const buf = new Uint8Array(await payload.imageBlob.arrayBuffer());
        await search.searchByObjectImage(buf, payload.mime, limit.value);
        if (search.hits.length === 0) {
            toast.info('未找到相似对象');
        }
        else {
            toast.success(`对象检索: ${search.hits.length} 个相似结果`);
        }
    }
    catch (e) {
        if (isLicenseError(e)) {
            toast.warning('会员未激活，请先兑换会员');
            router.push('/membership');
        }
        else {
            toast.danger('对象检索失败: ' + String(e));
        }
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
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({});
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
if (__VLS_ctx.croppedPreview) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.img)({
        src: (__VLS_ctx.croppedPreview),
        alt: "裁剪预览",
        ...{ class: "crop-thumb" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "muted text-sm" },
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
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "card-title" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "card-subtitle" },
});
(__VLS_ctx.search.isSearching ? '加载中…' : `${__VLS_ctx.search.hits.length} 张`);
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
        hint: "拖入图片开始检索",
    }));
    const __VLS_7 = __VLS_6({
        title: "还没有结果",
        hint: "拖入图片开始检索",
    }, ...__VLS_functionalComponentArgsRest(__VLS_6));
}
if (__VLS_ctx.cropSrc) {
    /** @type {[typeof CropModal, ]} */ ;
    // @ts-ignore
    const __VLS_9 = __VLS_asFunctionalComponent(CropModal, new CropModal({
        ...{ 'onCancel': {} },
        ...{ 'onConfirm': {} },
        imageSrc: (__VLS_ctx.cropSrc.url),
        mime: (__VLS_ctx.cropSrc.mime),
        imageWidth: (__VLS_ctx.cropSrc.width),
        imageHeight: (__VLS_ctx.cropSrc.height),
    }));
    const __VLS_10 = __VLS_9({
        ...{ 'onCancel': {} },
        ...{ 'onConfirm': {} },
        imageSrc: (__VLS_ctx.cropSrc.url),
        mime: (__VLS_ctx.cropSrc.mime),
        imageWidth: (__VLS_ctx.cropSrc.width),
        imageHeight: (__VLS_ctx.cropSrc.height),
    }, ...__VLS_functionalComponentArgsRest(__VLS_9));
    let __VLS_12;
    let __VLS_13;
    let __VLS_14;
    const __VLS_15 = {
        onCancel: (__VLS_ctx.cancelCrop)
    };
    const __VLS_16 = {
        onConfirm: (__VLS_ctx.confirmCrop)
    };
    var __VLS_11;
}
/** @type {__VLS_StyleScopedClasses['view']} */ ;
/** @type {__VLS_StyleScopedClasses['view-header']} */ ;
/** @type {__VLS_StyleScopedClasses['card']} */ ;
/** @type {__VLS_StyleScopedClasses['card-header']} */ ;
/** @type {__VLS_StyleScopedClasses['card-title']} */ ;
/** @type {__VLS_StyleScopedClasses['crop-thumb']} */ ;
/** @type {__VLS_StyleScopedClasses['muted']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['dropzone-hint']} */ ;
/** @type {__VLS_StyleScopedClasses['text-lg']} */ ;
/** @type {__VLS_StyleScopedClasses['muted']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['card']} */ ;
/** @type {__VLS_StyleScopedClasses['card-header']} */ ;
/** @type {__VLS_StyleScopedClasses['card-title']} */ ;
/** @type {__VLS_StyleScopedClasses['card-subtitle']} */ ;
var __VLS_dollars;
const __VLS_self = (await import('vue')).defineComponent({
    setup() {
        return {
            ImageGrid: ImageGrid,
            EmptyState: EmptyState,
            ExportBar: ExportBar,
            CropModal: CropModal,
            search: search,
            fileInput: fileInput,
            dragOver: dragOver,
            cropSrc: cropSrc,
            croppedPreview: croppedPreview,
            pickFile: pickFile,
            onFile: onFile,
            onDrop: onDrop,
            onDragOver: onDragOver,
            onDragLeave: onDragLeave,
            cancelCrop: cancelCrop,
            confirmCrop: confirmCrop,
        };
    },
});
export default (await import('vue')).defineComponent({
    setup() {
        return {};
    },
});
; /* PartiallyEnd: #4569/main.vue */
