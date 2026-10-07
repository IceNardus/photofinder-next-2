// PhotoFinder Next 2 — Tauri Event 订阅
//
// 后端通过 `app_handle.emit("pf:xxx", payload)` 推送事件。
// 前端用 `listen<T>(name, cb)` 订阅；返回的 unlisten 函数在组件 unmount 时调用。
//
// 命名规则：`pf:<domain>:<event>`（见 `apps/desktop/src-tauri/src/events.rs`）
import { listen } from '@tauri-apps/api/event';
// ----------------------------------------------------------------------------
// Channel names — 与后端保持一致
// ----------------------------------------------------------------------------
export const EVENT_SCAN_PROGRESS = 'pf:scan:progress';
export const EVENT_SCAN_COMPLETED = 'pf:scan:completed';
export const EVENT_SCAN_IMAGE_DISCOVERED = 'pf:scan:image_discovered';
export const EVENT_TASK_PROGRESS = 'pf:task:progress';
export const EVENT_TASK_COMPLETED = 'pf:task:completed';
export const EVENT_TASK_FAILED = 'pf:task:failed';
export const EVENT_APP_READY = 'pf:app:ready';
// ----------------------------------------------------------------------------
// Subscribe helpers — 内部用，统一返回 UnlistenFn
// ----------------------------------------------------------------------------
export function onScanProgress(cb) {
    return listen(EVENT_SCAN_PROGRESS, (e) => cb(e.payload));
}
export function onScanCompleted(cb) {
    return listen(EVENT_SCAN_COMPLETED, (e) => cb(e.payload));
}
export function onScanImageDiscovered(cb) {
    return listen(EVENT_SCAN_IMAGE_DISCOVERED, (e) => cb(e.payload));
}
export function onTaskProgress(cb) {
    return listen(EVENT_TASK_PROGRESS, (e) => cb(e.payload));
}
export function onTaskCompleted(cb) {
    return listen(EVENT_TASK_COMPLETED, (e) => cb(e.payload));
}
export function onTaskFailed(cb) {
    return listen(EVENT_TASK_FAILED, (e) => cb(e.payload));
}
export function onAppReady(cb) {
    return listen(EVENT_APP_READY, () => cb());
}
