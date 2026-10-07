// Toast store — 简单全局提示。
//
// 任何地方调用 `toast.success(msg)` / `toast.error(msg)` 即可。
// App.vue 渲染 stack。

import { defineStore } from 'pinia';
import { ref } from 'vue';

export type ToastKind = 'info' | 'success' | 'warning' | 'danger';

export interface ToastItem {
  id: number;
  kind: ToastKind;
  message: string;
  ttl: number;
}

let nextId = 1;

export const useToastStore = defineStore('toast', () => {
  const items = ref<ToastItem[]>([]);

  function push(kind: ToastKind, message: string, ttl = 3500) {
    const id = nextId++;
    items.value.push({ id, kind, message, ttl });
    setTimeout(() => dismiss(id), ttl);
  }

  function dismiss(id: number) {
    items.value = items.value.filter((t) => t.id !== id);
  }

  return {
    items,
    push,
    dismiss,
    info:    (m: string) => push('info', m),
    success: (m: string) => push('success', m),
    warning: (m: string) => push('warning', m),
    danger:  (m: string) => push('danger', m),
  };
});
