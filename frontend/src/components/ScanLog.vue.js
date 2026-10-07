import { computed, nextTick, onUnmounted, ref, watch } from 'vue';
import { useScanStore } from '@/stores/scan';
// helper outside <script setup> so template can call it as a method
function levelLabel(level) {
    switch (level) {
        case 'info': return 'INFO';
        case 'success': return 'OK';
        case 'warn': return 'WARN';
        case 'error': return 'ERR';
    }
}
export { levelLabel };
debugger; /* PartiallyEnd: #3632/both.vue */
export default await (async () => {
    const scan = useScanStore();
    const container = ref(null);
    const stickToBottom = ref(true);
    function timeLabel(ts) {
        const d = new Date(ts);
        const pad = (n) => String(n).padStart(2, '0');
        return `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`;
    }
    const rows = computed(() => scan.logEntries);
    function onScroll() {
        const el = container.value;
        if (!el)
            return;
        const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
        stickToBottom.value = distance < 16;
    }
    async function scrollToBottom() {
        await nextTick();
        const el = container.value;
        if (!el)
            return;
        el.scrollTop = el.scrollHeight;
    }
    watch(rows, async () => {
        if (stickToBottom.value) {
            void scrollToBottom();
        }
    });
    onUnmounted(() => {
        // store cleans up its own timer; nothing to do here
    });
    debugger; /* PartiallyEnd: #3632/scriptSetup.vue */
    const __VLS_ctx = {};
    let __VLS_components;
    let __VLS_directives;
    /** @type {__VLS_StyleScopedClasses['scan-log-row']} */ ;
    /** @type {__VLS_StyleScopedClasses['scan-log-msg']} */ ;
    /** @type {__VLS_StyleScopedClasses['scan-log-msg']} */ ;
    /** @type {__VLS_StyleScopedClasses['scan-log-msg']} */ ;
    // CSS variable injection 
    // CSS variable injection end 
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "scan-log" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "scan-log-head" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "scan-log-title" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "row gap-2" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
        ...{ class: "muted text-sm font-mono" },
    });
    (__VLS_ctx.rows.length);
    if (__VLS_ctx.rows.length) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
            ...{ onClick: (...[$event]) => {
                    if (!(__VLS_ctx.rows.length))
                        return;
                    __VLS_ctx.scan.clearLog();
                } },
            ...{ class: "btn sm ghost" },
        });
    }
    if (__VLS_ctx.rows.length === 0) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "scan-log-empty muted text-sm" },
        });
    }
    else {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ onScroll: (__VLS_ctx.onScroll) },
            ref: "container",
            ...{ class: "scan-log-body" },
        });
        /** @type {typeof __VLS_ctx.container} */ ;
        for (const [entry, idx] of __VLS_getVForSourceType((__VLS_ctx.rows))) {
            __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
                key: (`${entry.ts}-${idx}`),
                ...{ class: (['scan-log-row', `level-${entry.level}`]) },
            });
            __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
                ...{ class: "scan-log-ts font-mono" },
            });
            (__VLS_ctx.timeLabel(entry.ts));
            __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
                ...{ class: (['scan-log-tag', `tag-${entry.level}`]) },
            });
            (__VLS_ctx.levelLabel(entry.level));
            __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
                ...{ class: "scan-log-msg" },
            });
            (entry.message);
        }
    }
    /** @type {__VLS_StyleScopedClasses['scan-log']} */ ;
    /** @type {__VLS_StyleScopedClasses['scan-log-head']} */ ;
    /** @type {__VLS_StyleScopedClasses['scan-log-title']} */ ;
    /** @type {__VLS_StyleScopedClasses['row']} */ ;
    /** @type {__VLS_StyleScopedClasses['gap-2']} */ ;
    /** @type {__VLS_StyleScopedClasses['muted']} */ ;
    /** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
    /** @type {__VLS_StyleScopedClasses['font-mono']} */ ;
    /** @type {__VLS_StyleScopedClasses['btn']} */ ;
    /** @type {__VLS_StyleScopedClasses['sm']} */ ;
    /** @type {__VLS_StyleScopedClasses['ghost']} */ ;
    /** @type {__VLS_StyleScopedClasses['scan-log-empty']} */ ;
    /** @type {__VLS_StyleScopedClasses['muted']} */ ;
    /** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
    /** @type {__VLS_StyleScopedClasses['scan-log-body']} */ ;
    /** @type {__VLS_StyleScopedClasses['scan-log-ts']} */ ;
    /** @type {__VLS_StyleScopedClasses['font-mono']} */ ;
    /** @type {__VLS_StyleScopedClasses['scan-log-msg']} */ ;
    var __VLS_dollars;
    const __VLS_self = (await import('vue')).defineComponent({
        setup() {
            return {
                scan: scan,
                container: container,
                timeLabel: timeLabel,
                rows: rows,
                onScroll: onScroll,
                levelLabel: levelLabel,
            };
        },
    });
    return (await import('vue')).defineComponent({
        setup() {
            return {};
        },
    });
})(); /* PartiallyEnd: #4569/main.vue */
