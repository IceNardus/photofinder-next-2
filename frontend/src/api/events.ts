// PhotoFinder Next 2 — Tauri Event 订阅
//
// 后端通过 `app_handle.emit("pf:xxx", payload)` 推送事件。
// 前端用 `listen<T>(name, cb)` 订阅；返回的 unlisten 函数在组件 unmount 时调用。
//
// 命名规则：`pf:<domain>:<event>`（见 `apps/desktop/src-tauri/src/events.rs`）

import { listen, type UnlistenFn } from '@tauri-apps/api/event';

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
// Payload types
// ----------------------------------------------------------------------------

/** 扫描进度（folder scan 阶段）。 */
export interface ScanProgressPayload {
  path: string;
  candidate_count: number;
  scanned: number;
  current_file: string;
}

/** 扫描完成（一次 folder scan 结束）。 */
export interface ScanCompletedPayload {
  path: string;
  total_found: number;
  new_images: number;
  skipped: number;
  filtered: number;
  filtered_bpp: number;
  queued_face_tasks: number;
  queued_object_tasks: number;
  error: string | null;
}

/** 扫描时每发现一张图片（入库后立即推送）。 */
export interface ScanImageDiscoveredPayload {
  image_id: number;
  path: string;
}

/** 后台 task 进度。 */
export interface TaskProgressPayload {
  task_id: number;
  kind: string;
  progress: number; // 0..1
  message: string | null;
}

/** 后台 task 完成。 */
export interface TaskCompletedPayload {
  task_id: number;
  kind: string;
  duration_ms: number;
}

/** 后台 task 失败。 */
export interface TaskFailedPayload {
  task_id: number;
  kind: string;
  error: string;
  retry_count: number;
  will_retry: boolean;
}

// ----------------------------------------------------------------------------
// Subscribe helpers — 内部用，统一返回 UnlistenFn
// ----------------------------------------------------------------------------

export function onScanProgress(
  cb: (p: ScanProgressPayload) => void,
): Promise<UnlistenFn> {
  return listen<ScanProgressPayload>(EVENT_SCAN_PROGRESS, (e) => cb(e.payload));
}

export function onScanCompleted(
  cb: (p: ScanCompletedPayload) => void,
): Promise<UnlistenFn> {
  return listen<ScanCompletedPayload>(EVENT_SCAN_COMPLETED, (e) => cb(e.payload));
}

export function onScanImageDiscovered(
  cb: (p: ScanImageDiscoveredPayload) => void,
): Promise<UnlistenFn> {
  return listen<ScanImageDiscoveredPayload>(EVENT_SCAN_IMAGE_DISCOVERED, (e) => cb(e.payload));
}

export function onTaskProgress(
  cb: (p: TaskProgressPayload) => void,
): Promise<UnlistenFn> {
  return listen<TaskProgressPayload>(EVENT_TASK_PROGRESS, (e) => cb(e.payload));
}

export function onTaskCompleted(
  cb: (p: TaskCompletedPayload) => void,
): Promise<UnlistenFn> {
  return listen<TaskCompletedPayload>(EVENT_TASK_COMPLETED, (e) => cb(e.payload));
}

export function onTaskFailed(
  cb: (p: TaskFailedPayload) => void,
): Promise<UnlistenFn> {
  return listen<TaskFailedPayload>(EVENT_TASK_FAILED, (e) => cb(e.payload));
}

export function onAppReady(cb: () => void): Promise<UnlistenFn> {
  return listen<void>(EVENT_APP_READY, () => cb());
}
