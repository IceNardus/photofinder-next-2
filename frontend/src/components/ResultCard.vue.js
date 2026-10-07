import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import { useSearchStore } from '@/stores/search';
import { useSettingsStore } from '@/stores/settings';
const props = defineProps();
const search = useSearchStore();
const settings = useSettingsStore();
const url = ref(null);
const root = ref(null);
let observer = null;
const selected = computed(() => search.isSelected(props.hit.image_id));
const colorTier = computed(() => {
    const high = settings.settings.similarityHigh;
    const mid = settings.settings.similarityMid;
    if (props.hit.score >= high)
        return 'high';
    if (props.hit.score >= mid)
        return 'mid';
    return 'low';
});
function startObserve() {
    if (!root.value || observer)
        return;
    observer = new IntersectionObserver((entries) => {
        for (const e of entries) {
            if (e.isIntersecting) {
                observer?.disconnect();
                observer = null;
                void (async () => {
                    url.value = await search.loadThumbnail(props.hit.image_id);
                })();
                break;
            }
        }
    }, { rootMargin: '200px' });
    observer.observe(root.value);
}
function onSelect(e) {
    e.stopPropagation();
    search.toggleSelected(props.hit.image_id);
}
onMounted(startObserve);
onUnmounted(() => observer?.disconnect());
watch(() => props.hit.image_id, startObserve);
debugger; /* PartiallyEnd: #3632/scriptSetup.vue */
const __VLS_ctx = {};
let __VLS_components;
let __VLS_directives;
/** @type {__VLS_StyleScopedClasses['tier-high']} */ ;
/** @type {__VLS_StyleScopedClasses['tier-mid']} */ ;
/** @type {__VLS_StyleScopedClasses['image-cell-overlay']} */ ;
/** @type {__VLS_StyleScopedClasses['tier-low']} */ ;
/** @type {__VLS_StyleScopedClasses['image-cell-overlay']} */ ;
/** @type {__VLS_StyleScopedClasses['tier-high']} */ ;
/** @type {__VLS_StyleScopedClasses['tier-mid']} */ ;
/** @type {__VLS_StyleScopedClasses['score']} */ ;
/** @type {__VLS_StyleScopedClasses['tier-low']} */ ;
/** @type {__VLS_StyleScopedClasses['score']} */ ;
/** @type {__VLS_StyleScopedClasses['image-cell']} */ ;
/** @type {__VLS_StyleScopedClasses['select-toggle']} */ ;
// CSS variable injection 
// CSS variable injection end 
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ onClick: (__VLS_ctx.onSelect) },
    ref: "root",
    ...{ class: (['image-cell', `tier-${__VLS_ctx.colorTier}`, { selected: __VLS_ctx.selected }]) },
});
/** @type {typeof __VLS_ctx.root} */ ;
if (__VLS_ctx.url) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.img)({
        src: (__VLS_ctx.url),
        alt: (`img-${__VLS_ctx.hit.image_id}`),
    });
}
else {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "placeholder muted center" },
    });
}
__VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
    ...{ onClick: (__VLS_ctx.onSelect) },
    ...{ class: (['select-toggle', { checked: __VLS_ctx.selected }]) },
    'aria-label': (__VLS_ctx.selected ? '取消选择' : '选择'),
});
if (__VLS_ctx.selected) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
}
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "image-cell-overlay" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
(__VLS_ctx.hit.image_id);
__VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
    ...{ class: "score font-mono" },
});
((__VLS_ctx.hit.score * 100).toFixed(1));
/** @type {__VLS_StyleScopedClasses['placeholder']} */ ;
/** @type {__VLS_StyleScopedClasses['muted']} */ ;
/** @type {__VLS_StyleScopedClasses['center']} */ ;
/** @type {__VLS_StyleScopedClasses['image-cell-overlay']} */ ;
/** @type {__VLS_StyleScopedClasses['score']} */ ;
/** @type {__VLS_StyleScopedClasses['font-mono']} */ ;
var __VLS_dollars;
const __VLS_self = (await import('vue')).defineComponent({
    setup() {
        return {
            url: url,
            root: root,
            selected: selected,
            colorTier: colorTier,
            onSelect: onSelect,
        };
    },
    __typeProps: {},
});
export default (await import('vue')).defineComponent({
    setup() {
        return {};
    },
    __typeProps: {},
});
; /* PartiallyEnd: #4569/main.vue */
