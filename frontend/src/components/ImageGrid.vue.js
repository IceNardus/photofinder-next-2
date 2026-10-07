import ResultCard from './ResultCard.vue';
import { useSearchStore } from '@/stores/search';
const search = useSearchStore();
// 注意：这些是 Pinia 自动解包的响应式值，直接在模板中使用即可
// 不要用 const xxx = xxx.length 提前捕获，会失去响应式
debugger; /* PartiallyEnd: #3632/scriptSetup.vue */
const __VLS_ctx = {};
let __VLS_components;
let __VLS_directives;
// CSS variable injection 
// CSS variable injection end 
if (__VLS_ctx.search.hits.length === 0) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "empty" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "empty-title" },
    });
}
else {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "grid-with-pages" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "image-grid" },
    });
    for (const [h] of __VLS_getVForSourceType((__VLS_ctx.search.pageHits))) {
        /** @type {[typeof ResultCard, ]} */ ;
        // @ts-ignore
        const __VLS_0 = __VLS_asFunctionalComponent(ResultCard, new ResultCard({
            key: (`${h.target_id}-${h.rank}-${__VLS_ctx.search.page}`),
            hit: (h),
        }));
        const __VLS_1 = __VLS_0({
            key: (`${h.target_id}-${h.rank}-${__VLS_ctx.search.page}`),
            hit: (h),
        }, ...__VLS_functionalComponentArgsRest(__VLS_0));
    }
    if (__VLS_ctx.search.totalPages > 1) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "pager" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
            ...{ onClick: (...[$event]) => {
                    if (!!(__VLS_ctx.search.hits.length === 0))
                        return;
                    if (!(__VLS_ctx.search.totalPages > 1))
                        return;
                    __VLS_ctx.search.prevPage();
                } },
            ...{ class: "btn sm ghost" },
            disabled: (__VLS_ctx.search.page <= 1),
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
            ...{ class: "pager-info font-mono" },
        });
        (__VLS_ctx.search.page);
        (__VLS_ctx.search.totalPages);
        (__VLS_ctx.search.hits.length);
        __VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
            ...{ onClick: (...[$event]) => {
                    if (!!(__VLS_ctx.search.hits.length === 0))
                        return;
                    if (!(__VLS_ctx.search.totalPages > 1))
                        return;
                    __VLS_ctx.search.nextPage();
                } },
            ...{ class: "btn sm ghost" },
            disabled: (__VLS_ctx.search.page >= __VLS_ctx.search.totalPages),
        });
    }
}
/** @type {__VLS_StyleScopedClasses['empty']} */ ;
/** @type {__VLS_StyleScopedClasses['empty-title']} */ ;
/** @type {__VLS_StyleScopedClasses['grid-with-pages']} */ ;
/** @type {__VLS_StyleScopedClasses['image-grid']} */ ;
/** @type {__VLS_StyleScopedClasses['pager']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['sm']} */ ;
/** @type {__VLS_StyleScopedClasses['ghost']} */ ;
/** @type {__VLS_StyleScopedClasses['pager-info']} */ ;
/** @type {__VLS_StyleScopedClasses['font-mono']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['sm']} */ ;
/** @type {__VLS_StyleScopedClasses['ghost']} */ ;
var __VLS_dollars;
const __VLS_self = (await import('vue')).defineComponent({
    setup() {
        return {
            ResultCard: ResultCard,
            search: search,
        };
    },
});
export default (await import('vue')).defineComponent({
    setup() {
        return {};
    },
});
; /* PartiallyEnd: #4569/main.vue */
