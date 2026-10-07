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
// ----------------------------------------------------------------------------
// Scan
// ----------------------------------------------------------------------------
export function scanFolder(path) {
    return invoke('scan_folder', { args: { path } });
}
export function stopScan() {
    return invoke('stop_scan');
}
// ----------------------------------------------------------------------------
// Search
// ----------------------------------------------------------------------------
/** 用图片 bytes 检索相似人脸。bytes 是 ArrayBuffer / Uint8Array。 */
export function searchByFaceImage(bytes, topK) {
    return invoke('search_by_face_image', {
        bytes: Array.from(bytes),
        topK,
    });
}
/** 用 512 维 embedding 检索相似人脸。 */
export function searchByFaceEmbedding(embedding, topK) {
    return invoke('search_by_face_embedding', {
        embedding,
        topK,
    });
}
/** 按对象类别 ID 列出图片（不走向量搜索）。 */
export function listObjectsByClass(classId, limit) {
    return invoke('list_objects_by_class', {
        args: { class_id: classId, limit },
    });
}
// ----------------------------------------------------------------------------
// Persons
// ----------------------------------------------------------------------------
export function listPersons() {
    return invoke('list_persons');
}
export function renamePerson(personId, name) {
    return invoke('rename_person', {
        args: { person_id: personId, name },
    });
}
export function facesOfPerson(personId) {
    return invoke('faces_of_person', { personId });
}
/**
 * Phase 5: 查找 person 出现过的所有 image。
 *
 * 后端用 multi-prototype HNSW 聚合(frontal/left/right/high_quality/general),
 * 比 `searchByFaceImage` 召回更全 — 即使目标 person 在新图里是侧脸,
 * 也能用 left_profile prototype 召回。
 */
export function searchByPerson(personId, topK) {
    return invoke('search_by_person', {
        args: { person_id: personId, top_k: topK },
    });
}
/** 触发全量聚类（异步，返回新任务的 ID）。 */
export function clusterFaces() {
    return invoke('cluster_faces');
}
/** 重新分析人物聚类（同步执行，直接返回聚类结果）。 */
export function rebuildPersonClusters() {
    return invoke('rebuild_person_clusters');
}
// ----------------------------------------------------------------------------
// Library
// ----------------------------------------------------------------------------
export function listLibrary(limit, offset) {
    return invoke('list_library', {
        args: { limit, offset },
    });
}
/** 取缩略图（data URL：`data:image/jpeg;base64,...`）。 */
export function getThumbnail(imageId, maxSize = 256) {
    return invoke('get_thumbnail', {
        args: { image_id: imageId, max_size: maxSize },
    });
}
// ----------------------------------------------------------------------------
// Statistics
// ----------------------------------------------------------------------------
export function getStatistics() {
    return invoke('get_statistics');
}
// ----------------------------------------------------------------------------
// Tasks
// ----------------------------------------------------------------------------
export function listTasks(status, limit = 200) {
    return invoke('list_tasks', {
        args: { status, limit },
    });
}
export function cancelTask(taskId) {
    return invoke('cancel_task', { taskId });
}
export function indexPending() {
    return invoke('index_pending');
}
// ----------------------------------------------------------------------------
// Models
// ----------------------------------------------------------------------------
export function getModelsStatus() {
    return invoke('get_models_status');
}
// ----------------------------------------------------------------------------
// App info
// ----------------------------------------------------------------------------
export function getDataDir() {
    return invoke('get_data_dir');
}
export function getAppInfo() {
    return invoke('get_app_info');
}
export function clearDatabase() {
    return invoke('clear_database');
}
// ----------------------------------------------------------------------------
// Phase 4 — 新增命令
// ----------------------------------------------------------------------------
export function rebuildThumbnails() {
    return invoke('rebuild_thumbnails');
}
export function rebuildFaceIndex() {
    return invoke('rebuild_face_index');
}
export function getScanStatus() {
    return invoke('get_scan_status');
}
export function getProcessingStatus() {
    return invoke('get_processing_status');
}
export function copyFiles(args) {
    return invoke('copy_files', { args });
}
export function initObjectSearch() {
    return invoke('init_object_search');
}
export function searchObjects(args) {
    return invoke('search_objects', { args });
}
export function indexImagesForObject() {
    return invoke('index_images_for_object');
}
export function writeQueryImage(args) {
    return invoke('write_query_image', { args });
}
export function writeCroppedImage(args) {
    return invoke('write_cropped_image', { args });
}
export function getImageThumbnail(args) {
    return invoke('get_image_thumbnail', { args });
}
// ----------------------------------------------------------------------------
// License / Account
// ----------------------------------------------------------------------------
export function registerAccount(email, password) {
    return invoke('register_account', { email, password });
}
export function loginAccount(email, password) {
    return invoke('login_account', { email, password });
}
export function logoutAccount() {
    return invoke('logout_account');
}
export function getAccount() {
    return invoke('get_account');
}
export function redeemCode(code) {
    return invoke('redeem_code', { code });
}
export function getLicenseStatus() {
    return invoke('get_license_status');
}
export function checkLicense() {
    return invoke('check_license');
}
