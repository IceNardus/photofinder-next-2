// PhotoFinder Next 2 — Tauri IPC 客户端
//
// 所有 invoke() 调用集中在此，避免业务代码直接散落 Tauri API。
// 命名与 `apps/desktop/src-tauri/src/commands.rs` 一一对应。
//
// Tauri 2 IPC 约定：
// - 多参数命令直接以参数名传对象（`{ bytes, topK }`，Tauri 2 IPC 将 camelCase 转为 snake_case）
// - 单 args struct 命令以 `{ args: {...} }` 传递
//   （这是 Rust `fn(args: X)` 形式所要求的）

import { invoke } from '@tauri-apps/api/core';
import type {
  AppInfo,
  ClearDatabaseResult,
  ClusterSummary,
  CopyFileResult,
  CopyFilesArgs,
  Face,
  GetImageThumbnailArgs,
  GetThumbnailArgs,
  LibraryImage,
  ListByClassArgs,
  ListLibraryArgs,
  ListTasksArgs,
  ModelStatus,
  ObjectSearchHit,
  Person,
  ProcessingStatus,
  RebuildProgress,
  RenamePersonArgs,
  ScanFolderArgs,
  ScanResult,
  ScanStatus,
  SearchByPersonArgs,
  SearchHit,
  SearchObjectsArgs,
  Statistics,
  Task,
  TaskStatusString,
  WriteImageArgs,
} from './types';

// ----------------------------------------------------------------------------
// Scan
// ----------------------------------------------------------------------------

export function scanFolder(path: string): Promise<ScanResult> {
  return invoke<ScanResult>('scan_folder', { args: { path } satisfies ScanFolderArgs });
}

export function stopScan(): Promise<void> {
  return invoke<void>('stop_scan');
}

// ----------------------------------------------------------------------------
// Search
// ----------------------------------------------------------------------------

/** 用图片 bytes 检索相似人脸。bytes 是 ArrayBuffer / Uint8Array。 */
export function searchByFaceImage(bytes: Uint8Array, topK: number): Promise<SearchHit[]> {
  return invoke<SearchHit[]>('search_by_face_image', {
    bytes: Array.from(bytes),
    topK,
  });
}

/** 用 512 维 embedding 检索相似人脸。 */
export function searchByFaceEmbedding(embedding: number[], topK: number): Promise<SearchHit[]> {
  return invoke<SearchHit[]>('search_by_face_embedding', {
    embedding,
    topK,
  });
}

/** 按对象类别 ID 列出图片（不走向量搜索）。 */
export function listObjectsByClass(classId: number, limit: number): Promise<SearchHit[]> {
  return invoke<SearchHit[]>('list_objects_by_class', {
    args: { class_id: classId, limit } satisfies ListByClassArgs,
  });
}

// ----------------------------------------------------------------------------
// Persons
// ----------------------------------------------------------------------------

export function listPersons(): Promise<Person[]> {
  return invoke<Person[]>('list_persons');
}

export function renamePerson(personId: number, name: string | null): Promise<void> {
  return invoke<void>('rename_person', {
    args: { person_id: personId, name } satisfies RenamePersonArgs,
  });
}

export function facesOfPerson(personId: number): Promise<Face[]> {
  return invoke<Face[]>('faces_of_person', { personId });
}

/**
 * Phase 5: 查找 person 出现过的所有 image。
 *
 * 后端用 multi-prototype HNSW 聚合(frontal/left/right/high_quality/general),
 * 比 `searchByFaceImage` 召回更全 — 即使目标 person 在新图里是侧脸,
 * 也能用 left_profile prototype 召回。
 */
export function searchByPerson(
  personId: number,
  topK: number,
): Promise<SearchHit[]> {
  return invoke<SearchHit[]>('search_by_person', {
    args: { person_id: personId, top_k: topK } satisfies SearchByPersonArgs,
  });
}

/** 触发全量聚类（异步，返回新任务的 ID）。 */
export function clusterFaces(): Promise<number> {
  return invoke<number>('cluster_faces');
}

/** 重新分析人物聚类（同步执行，直接返回聚类结果）。 */
export function rebuildPersonClusters(): Promise<ClusterSummary> {
  return invoke<ClusterSummary>('rebuild_person_clusters');
}

// ----------------------------------------------------------------------------
// Library
// ----------------------------------------------------------------------------

export function listLibrary(limit: number, offset: number): Promise<LibraryImage[]> {
  return invoke<LibraryImage[]>('list_library', {
    args: { limit, offset } satisfies ListLibraryArgs,
  });
}

/** 取缩略图（data URL：`data:image/jpeg;base64,...`）。 */
export function getThumbnail(imageId: number, maxSize = 256): Promise<string> {
  return invoke<string>('get_thumbnail', {
    args: { image_id: imageId, max_size: maxSize } satisfies GetThumbnailArgs,
  });
}

// ----------------------------------------------------------------------------
// Statistics
// ----------------------------------------------------------------------------

export function getStatistics(): Promise<Statistics> {
  return invoke<Statistics>('get_statistics');
}

// ----------------------------------------------------------------------------
// Tasks
// ----------------------------------------------------------------------------

export function listTasks(status: TaskStatusString | null, limit = 200): Promise<Task[]> {
  return invoke<Task[]>('list_tasks', {
    args: { status, limit } satisfies ListTasksArgs,
  });
}

export function cancelTask(taskId: number): Promise<void> {
  return invoke<void>('cancel_task', { taskId });
}

export function indexPending(): Promise<number> {
  return invoke<number>('index_pending');
}

// ----------------------------------------------------------------------------
// Models
// ----------------------------------------------------------------------------

export function getModelsStatus(): Promise<ModelStatus[]> {
  return invoke<ModelStatus[]>('get_models_status');
}

// ----------------------------------------------------------------------------
// App info
// ----------------------------------------------------------------------------

export function getDataDir(): Promise<string> {
  return invoke<string>('get_data_dir');
}

export function getAppInfo(): Promise<AppInfo> {
  return invoke<AppInfo>('get_app_info');
}

export function clearDatabase(): Promise<ClearDatabaseResult> {
  return invoke<ClearDatabaseResult>('clear_database');
}

// ----------------------------------------------------------------------------
// Phase 4 — 新增命令
// ----------------------------------------------------------------------------

export function rebuildThumbnails(): Promise<RebuildProgress> {
  return invoke<RebuildProgress>('rebuild_thumbnails');
}

export function rebuildFaceIndex(): Promise<string> {
  return invoke<string>('rebuild_face_index');
}

export function getScanStatus(): Promise<ScanStatus> {
  return invoke<ScanStatus>('get_scan_status');
}

export function getProcessingStatus(): Promise<ProcessingStatus> {
  return invoke<ProcessingStatus>('get_processing_status');
}

export function copyFiles(args: CopyFilesArgs): Promise<CopyFileResult[]> {
  return invoke<CopyFileResult[]>('copy_files', { args });
}

export function initObjectSearch(): Promise<string> {
  return invoke<string>('init_object_search');
}

export function searchObjects(args: SearchObjectsArgs): Promise<ObjectSearchHit[]> {
  return invoke<ObjectSearchHit[]>('search_objects', { args });
}

export function indexImagesForObject(): Promise<number> {
  return invoke<number>('index_images_for_object');
}

export function writeQueryImage(args: WriteImageArgs): Promise<string> {
  return invoke<string>('write_query_image', { args });
}

export function writeCroppedImage(args: WriteImageArgs): Promise<string> {
  return invoke<string>('write_cropped_image', { args });
}

export function getImageThumbnail(args: GetImageThumbnailArgs): Promise<string> {
  return invoke<string>('get_image_thumbnail', { args });
}

// ----------------------------------------------------------------------------
// License / Account (stub implementations)
// ----------------------------------------------------------------------------

export interface Account {
  id: number;
  username: string;
  email: string;
  created_at: number;  // timestamp
}

export interface License {
  status: string;
  plan: string;
  expires_at: number | null;
  remaining_days: number;
}

export function checkLicense(): Promise<boolean> {
  return Promise.resolve(true);
}

export function getAccount(): Promise<Account | null> {
  return Promise.resolve(null);
}

export function getLicenseStatus(): Promise<License | null> {
  return Promise.resolve(null);
}

export function loginAccount(_username: string, _password: string): Promise<Account> {
  return Promise.reject(new Error('Not implemented'));
}

export function logoutAccount(): Promise<void> {
  return Promise.resolve();
}

export function redeemCode(_code: string): Promise<License> {
  return Promise.reject(new Error('Not implemented'));
}

export function registerAccount(_email: string, _password: string): Promise<Account> {
  return Promise.reject(new Error('Not implemented'));
}

// ----------------------------------------------------------------------------
// Re-export types for convenience
// ----------------------------------------------------------------------------

export type {
  AppInfo,
  ClearDatabaseResult,
  ClusterSummary,
  CopyFileResult,
  Face,
  LibraryImage,
  ModelStatus,
  ObjectSearchHit,
  Person,
  ScanResult,
  SearchHit,
  Statistics,
  Task,
  TaskStatusString,
} from './types';
