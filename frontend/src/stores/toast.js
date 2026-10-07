// Toast store — 简单全局提示。
//
// 任何地方调用 `toast.success(msg)` / `toast.error(msg)` 即可。
// App.vue 渲染 stack。
import { defineStore } from 'pinia';
import { ref } from 'vue';
let nextId = 1;
export const useToastStore = defineStore('toast', () => {
    const items = ref([]);
    function push(kind, message, ttl = 3500) {
        const id = nextId++;
        items.value.push({ id, kind, message, ttl });
        setTimeout(() => dismiss(id), ttl);
    }
    function dismiss(id) {
        items.value = items.value.filter((t) => t.id !== id);
    }
    return {
        items,
        push,
        dismiss,
        info: (m) => push('info', m),
        success: (m) => push('success', m),
        warning: (m) => push('warning', m),
        danger: (m) => push('danger', m),
    };
});
