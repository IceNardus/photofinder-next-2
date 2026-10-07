// IPC 类型定义（与 `apps/desktop/src-tauri/src/ipc_types.rs` 一一对应）。
//
// 命名规则：snake_case，与 Rust DTO 字段完全对齐。
// 前端用这些类型做 IPC + state 强类型。

export interface ScanResult {
  total_found: number;
  new_images: number;
  skipped: number;
  /** 被 ImageTypeClassifier 拒掉的（非 Photo：截图 / 二次元 / 海报） */
  filtered: number;
  /** 被 bytes_per_pixel 过滤的（vector / icon / 低色图） */
  filtered_bpp: number;
  queued_face_tasks: number;
  queued_object_tasks: number;
  path: string;
}

export interface ScanFolderArgs {
  path: string;
}

export interface SearchHit {
  image_id: number;
  target_id: number;
  score: number;
  rank: number;
}

export interface ListByClassArgs {
  class_id: number;
  limit: number;
}

export interface Person {
  id: number;
  name: string | null;
  face_count: number;
  /** 后端 PersonStatus 字符串（如 "active"）。Phase 1 引入,前端未读取。 */
  status?: string;
  created_at: string | null;
  updated_at?: string | null;
}

export interface Face {
  id: number;
  image_id: number;
  bbox: [number, number, number, number];
  quality: number;
  detector_score: number;
  image_path: string | null;
  thumbnail: string | null;
}

export interface RenamePersonArgs {
  person_id: number;
  name: string | null;
}

/** Phase 5: 按 person 列出该 person 出现过的所有 image。 */
export interface SearchByPersonArgs {
  person_id: number;
  top_k: number;
}

export interface LibraryImage {
  id: number;
  path: string;
  thumbnail: string | null;
  scan_status: string;
  width: number | null;
  height: number | null;
  /** 检测到的人脸数（索引完成后才有值） */
  face_count: number;
  /** 检测到的物品数（索引完成后才有值） */
  object_count: number;
}

export interface ListLibraryArgs {
  limit: number;
  offset: number;
}

export interface GetThumbnailArgs {
  image_id: number;
  max_size: number;
}

export interface Statistics {
  image_count: number;
  face_count: number;
  object_count: number;
  person_count: number;
  pending_task_count: number;
  completed_task_count: number;
  failed_task_count: number;
  face_index_size: number;
  object_index_size: number | null;
  db_path: string;
  models_dir: string | null;
  patch_count: number;
  index_size_bytes: number;
  vector_store_size_bytes: number;
  database_size_bytes: number;
  thumbnail_count: number;
  thumbnail_size_bytes: number;
}

export interface ScanStatus {
  is_scanning: boolean;
  total_images: number;
  processed_images: number;
  pending_tasks: number;
  current_file: string;
  current_faces: number;
  current_objects: number;
  current_patches: number;
}

export interface ProcessingStatus {
  current_image: string;
  current_faces: number;
  current_patches: number;
  current_objects: number;
  log_message: string;
  last_completion_message: string;
}

export interface ObjectSearchHit {
  image_id: number;
  image_path: string;
  image_name: string;
  thumbnail_path: string | null;
  confidence: number;
  similarity: number;
  embedding_score: number;
  inlier_ratio: number;
  inlier_count: number;
  match_count: number;
  bbox_overlap: number;
}

export interface RebuildProgress {
  total: number;
  generated: number;
  failed: number;
  speed: number;
}

export interface CopyFileResult {
  src: string;
  dst: string;
  success: boolean;
  error: string | null;
}

export interface CropArgs {
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface WriteImageArgs {
  data: number[];
  mime: string;
  crop?: CropArgs | null;
}

export interface SearchObjectsArgs {
  query_image: string;
  top_k: number;
}

export interface CopyFilesArgs {
  source_paths?: string[];
  image_ids?: number[];
  dest_dir: string;
}

export interface GetImageThumbnailArgs {
  image_path: string;
  max_size: number;
}

export type TaskStatusString =
  | 'pending'
  | 'running'
  | 'completed'
  | 'failed'
  | 'cancelled';

export interface Task {
  id: number;
  kind: string;
  status: TaskStatusString;
  priority: number;
  retry_count: number;
  max_retries: number;
  error: string | null;
  created_at: string;
  started_at: string | null;
  completed_at: string | null;
}

export interface ListTasksArgs {
  status: TaskStatusString | null;
  limit: number;
}

export interface ModelStatus {
  id: string;
  loaded: boolean;
  path: string | null;
  size_bytes: number;
  verified: boolean;
}

export interface AppInfo {
  name: string;
  version: string;
  data_dir: string;
  db_path: string;
  models_dir: string | null;
  platform: string;
}

export interface ClearDatabaseResult {
  success: boolean;
  cleared_images: boolean;
  cleared_faces: boolean;
  cleared_objects: boolean;
  cleared_persons: boolean;
  cleared_tasks: boolean;
  cleared_vectors: boolean;
  errors: string[];
}

/** 聚类结果（rebuild_person_clusters 返回值）。 */
export interface ClusterSummary {
  assigned: number;
  created: number;
}
