import { useToastStore } from '@/stores/toast';
const toast = useToastStore();
debugger; /* PartiallyEnd: #3632/scriptSetup.vue */
const __VLS_ctx = {};
let __VLS_components;
let __VLS_directives;
if (__VLS_ctx.toast.items.length) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "toast-stack" },
    });
    for (const [t] of __VLS_getVForSourceType((__VLS_ctx.toast.items))) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            key: (t.id),
            ...{ class: (['toast', t.kind]) },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "toast-msg" },
        });
        (t.message);
        __VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
            ...{ onClick: (...[$event]) => {
                    if (!(__VLS_ctx.toast.items.length))
                        return;
                    __VLS_ctx.toast.dismiss(t.id);
                } },
            ...{ class: "toast-close" },
        });
    }
}
/** @type {__VLS_StyleScopedClasses['toast-stack']} */ ;
/** @type {__VLS_StyleScopedClasses['toast-msg']} */ ;
/** @type {__VLS_StyleScopedClasses['toast-close']} */ ;
var __VLS_dollars;
const __VLS_self = (await import('vue')).defineComponent({
    setup() {
        return {
            toast: toast,
        };
    },
});
export default (await import('vue')).defineComponent({
    setup() {
        return {};
    },
});
; /* PartiallyEnd: #4569/main.vue */
