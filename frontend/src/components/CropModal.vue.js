import { computed, onMounted, onUnmounted, ref } from 'vue';
const props = defineProps();
const emit = defineEmits();
const HANDLE_SIZE = 14;
const MIN_DRAG = 20;
const MIN_RUST = 32;
const stage = ref(null);
const stageSize = ref({ w: 0, h: 0 });
const displayed = computed(() => {
    const sw = stageSize.value.w;
    const sh = stageSize.value.h;
    if (sw === 0 || sh === 0 || props.imageWidth === 0 || props.imageHeight === 0) {
        return { w: 0, h: 0, offsetX: 0, offsetY: 0 };
    }
    const s = Math.min(sw / props.imageWidth, sh / props.imageHeight);
    const dw = Math.round(props.imageWidth * s);
    const dh = Math.round(props.imageHeight * s);
    return { w: dw, h: dh, offsetX: (sw - dw) / 2, offsetY: (sh - dh) / 2 };
});
const scale = computed(() => {
    if (props.imageWidth === 0)
        return 1;
    return displayed.value.w / props.imageWidth;
});
const cropImageSize = computed(() => {
    const sx = 1 / scale.value;
    return {
        w: Math.round(crop.value.w * sx),
        h: Math.round(crop.value.h * sx),
    };
});
const crop = ref({ x: 0, y: 0, w: 0, h: 0 });
const drag = ref(null);
const cropStyle = computed(() => ({
    left: `${crop.value.x}px`,
    top: `${crop.value.y}px`,
    width: `${crop.value.w}px`,
    height: `${crop.value.h}px`,
}));
function onResize() {
    const el = stage.value;
    if (!el)
        return;
    stageSize.value = { w: el.clientWidth, h: el.clientHeight };
}
onMounted(() => {
    onResize();
    window.addEventListener('resize', onResize);
    const d = displayed.value;
    crop.value = {
        x: d.offsetX + d.w * 0.2,
        y: d.offsetY + d.h * 0.2,
        w: d.w * 0.6,
        h: d.h * 0.6,
    };
});
onUnmounted(() => window.removeEventListener('resize', onResize));
function clampBox(b) {
    const d = displayed.value;
    const minX = d.offsetX;
    const minY = d.offsetY;
    const maxX = d.offsetX + d.w;
    const maxY = d.offsetY + d.h;
    const minSide = Math.max(HANDLE_SIZE, MIN_DRAG);
    let { x, y, w, h } = b;
    w = Math.max(minSide, Math.min(d.w, w));
    h = Math.max(minSide, Math.min(d.h, h));
    x = Math.max(minX, Math.min(maxX - w, x));
    y = Math.max(minY, Math.min(maxY - h, y));
    return { x, y, w, h };
}
function startDrag(mode, e) {
    e.preventDefault();
    e.stopPropagation();
    drag.value = { mode, startX: e.clientX, startY: e.clientY, startBox: { ...crop.value } };
    e.currentTarget.setPointerCapture?.(e.pointerId);
}
function onMove(e) {
    if (!drag.value)
        return;
    const dx = e.clientX - drag.value.startX;
    const dy = e.clientY - drag.value.startY;
    const sb = drag.value.startBox;
    let next;
    switch (drag.value.mode) {
        case 'move':
            next = { ...sb, x: sb.x + dx, y: sb.y + dy };
            break;
        case 'nw':
            next = { x: sb.x + dx, y: sb.y + dy, w: sb.w - dx, h: sb.h - dy };
            break;
        case 'ne':
            next = { y: sb.y + dy, w: sb.w + dx, h: sb.h - dy, x: sb.x };
            break;
        case 'sw':
            next = { x: sb.x + dx, w: sb.w - dx, h: sb.h + dy, y: sb.y };
            break;
        case 'se':
            next = { w: sb.w + dx, h: sb.h + dy, x: sb.x, y: sb.y };
            break;
        case 'n':
            next = { ...sb, y: sb.y + dy, h: sb.h - dy };
            break;
        case 's':
            next = { ...sb, h: sb.h + dy };
            break;
        case 'e':
            next = { ...sb, w: sb.w + dx };
            break;
        case 'w':
            next = { ...sb, x: sb.x + dx, w: sb.w - dx };
            break;
        default: return;
    }
    crop.value = clampBox(next);
}
function endDrag() { drag.value = null; }
async function onConfirm() {
    const bx = crop.value;
    const minBox = clampBox({ ...bx, w: Math.max(bx.w, MIN_RUST), h: Math.max(bx.h, MIN_RUST) });
    crop.value = minBox;
    const sx = 1 / scale.value;
    const d = displayed.value;
    const ix = Math.round((bx.x - d.offsetX) * sx);
    const iy = Math.round((bx.y - d.offsetY) * sx);
    const iw = Math.round(bx.w * sx);
    const ih = Math.round(bx.h * sx);
    const img = new Image();
    img.src = props.imageSrc;
    await new Promise((res, rej) => {
        img.onload = () => res();
        img.onerror = () => rej(new Error('image load failed'));
    });
    const canvas = document.createElement('canvas');
    canvas.width = iw;
    canvas.height = ih;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, ix, iy, iw, ih, 0, 0, iw, ih);
    const blob = await new Promise((res, rej) => canvas.toBlob((b) => b ? res(b) : rej(new Error('toBlob null')), props.mime, 0.92));
    const previewUrl = await new Promise((res) => {
        const reader = new FileReader();
        reader.onloadend = () => res(reader.result);
        reader.readAsDataURL(blob);
    });
    emit('confirm', { x: ix, y: iy, w: iw, h: ih, imageBlob: blob, mime: props.mime, previewUrl });
}
debugger; /* PartiallyEnd: #3632/scriptSetup.vue */
const __VLS_ctx = {};
let __VLS_components;
let __VLS_directives;
/** @type {__VLS_StyleScopedClasses['crop-modal-foot']} */ ;
/** @type {__VLS_StyleScopedClasses['crop-handle-move']} */ ;
// CSS variable injection 
// CSS variable injection end 
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ onClick: (...[$event]) => {
            __VLS_ctx.emit('cancel');
        } },
    ...{ class: "crop-modal-backdrop" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "crop-modal" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.header, __VLS_intrinsicElements.header)({
    ...{ class: "crop-modal-head" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "crop-modal-title" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "crop-info font-mono text-sm" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
    ...{ class: "crop-dim" },
});
(__VLS_ctx.cropImageSize.w);
(__VLS_ctx.cropImageSize.h);
if (__VLS_ctx.cropImageSize.w < __VLS_ctx.MIN_RUST || __VLS_ctx.cropImageSize.h < __VLS_ctx.MIN_RUST) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
        ...{ class: "crop-warn" },
    });
    (__VLS_ctx.MIN_RUST);
}
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
    ...{ class: "crop-hint muted" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ref: "stage",
    ...{ class: "crop-stage" },
});
/** @type {typeof __VLS_ctx.stage} */ ;
__VLS_asFunctionalElement(__VLS_intrinsicElements.img)({
    src: (__VLS_ctx.imageSrc),
    ...{ class: "crop-image" },
    alt: "crop source",
    draggable: "false",
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "crop-box" },
    ...{ style: (__VLS_ctx.cropStyle) },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "thirds-lines" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div)({
    ...{ class: "thirds-v" },
    ...{ style: ({ left: `${(1 / 3) * 100}%` }) },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div)({
    ...{ class: "thirds-v" },
    ...{ style: ({ left: `${(2 / 3) * 100}%` }) },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div)({
    ...{ class: "thirds-h" },
    ...{ style: ({ top: `${(1 / 3) * 100}%` }) },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div)({
    ...{ class: "thirds-h" },
    ...{ style: ({ top: `${(2 / 3) * 100}%` }) },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div)({
    ...{ onPointerdown: (...[$event]) => {
            __VLS_ctx.startDrag('move', $event);
        } },
    ...{ onPointermove: (__VLS_ctx.onMove) },
    ...{ onPointerup: (__VLS_ctx.endDrag) },
    ...{ onPointercancel: (__VLS_ctx.endDrag) },
    ...{ class: "crop-handle crop-handle-move" },
});
for (const [h] of __VLS_getVForSourceType(['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'])) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div)({
        ...{ onPointerdown: (...[$event]) => {
                __VLS_ctx.startDrag(h, $event);
            } },
        ...{ onPointermove: (__VLS_ctx.onMove) },
        ...{ onPointerup: (__VLS_ctx.endDrag) },
        ...{ onPointercancel: (__VLS_ctx.endDrag) },
        key: (h),
        ...{ class: (['crop-handle', `crop-handle-${h}`]) },
    });
}
__VLS_asFunctionalElement(__VLS_intrinsicElements.footer, __VLS_intrinsicElements.footer)({
    ...{ class: "crop-modal-foot" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
    ...{ onClick: (...[$event]) => {
            __VLS_ctx.emit('cancel');
        } },
    ...{ class: "btn ghost" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
    ...{ onClick: (__VLS_ctx.onConfirm) },
    ...{ class: "btn primary" },
});
/** @type {__VLS_StyleScopedClasses['crop-modal-backdrop']} */ ;
/** @type {__VLS_StyleScopedClasses['crop-modal']} */ ;
/** @type {__VLS_StyleScopedClasses['crop-modal-head']} */ ;
/** @type {__VLS_StyleScopedClasses['crop-modal-title']} */ ;
/** @type {__VLS_StyleScopedClasses['crop-info']} */ ;
/** @type {__VLS_StyleScopedClasses['font-mono']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['crop-dim']} */ ;
/** @type {__VLS_StyleScopedClasses['crop-warn']} */ ;
/** @type {__VLS_StyleScopedClasses['crop-hint']} */ ;
/** @type {__VLS_StyleScopedClasses['muted']} */ ;
/** @type {__VLS_StyleScopedClasses['crop-stage']} */ ;
/** @type {__VLS_StyleScopedClasses['crop-image']} */ ;
/** @type {__VLS_StyleScopedClasses['crop-box']} */ ;
/** @type {__VLS_StyleScopedClasses['thirds-lines']} */ ;
/** @type {__VLS_StyleScopedClasses['thirds-v']} */ ;
/** @type {__VLS_StyleScopedClasses['thirds-v']} */ ;
/** @type {__VLS_StyleScopedClasses['thirds-h']} */ ;
/** @type {__VLS_StyleScopedClasses['thirds-h']} */ ;
/** @type {__VLS_StyleScopedClasses['crop-handle']} */ ;
/** @type {__VLS_StyleScopedClasses['crop-handle-move']} */ ;
/** @type {__VLS_StyleScopedClasses['crop-modal-foot']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['ghost']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['primary']} */ ;
var __VLS_dollars;
const __VLS_self = (await import('vue')).defineComponent({
    setup() {
        return {
            emit: emit,
            MIN_RUST: MIN_RUST,
            stage: stage,
            cropImageSize: cropImageSize,
            cropStyle: cropStyle,
            startDrag: startDrag,
            onMove: onMove,
            endDrag: endDrag,
            onConfirm: onConfirm,
        };
    },
    __typeEmits: {},
    __typeProps: {},
});
export default (await import('vue')).defineComponent({
    setup() {
        return {};
    },
    __typeEmits: {},
    __typeProps: {},
});
; /* PartiallyEnd: #4569/main.vue */
