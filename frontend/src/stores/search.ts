// Search store — 人脸 / 对象 检索结果 + 分页 + 选择。
//
// 状态：query bytes / embedding / class_id
// 结果：SearchHit[] + 缩略图缓存
// 分页：每页 20 个；`page` 1-based
// 选择：Set<image_id>；select-all 在 store 层实现，跨页生效

import { defineStore } from 'pinia';
import { computed, ref } from 'vue';
import {
  copyFiles,
  getThumbnail,
  listObjectsByClass,
  searchByFaceEmbedding,
  searchByFaceImage,
  searchByPerson,
  searchObjects,
  writeQueryImage,
  type CopyFileResult,
  type ObjectSearchHit,
  type SearchHit,
} from '@/api';
import { useSettingsStore } from '@/stores/settings';

export type SearchMode = 'face-image' | 'face-embedding' | 'object-class';

const DEFAULT_PAGE_SIZE = 20;

export const useSearchStore = defineStore('search', () => {
  const mode = ref<SearchMode>('face-image');
  const isSearching = ref(false);
  const lastError = ref<string | null>(null);

  const hits = ref<SearchHit[]>([]);
  // 缩略图缓存：image_id -> data URL
  const thumbnailCache = ref<Map<number, string>>(new Map());
  const classIdInput = ref(0);

  // ---- Phase 6: 分页 + 选择 ----
  const page = ref(1);
  const selectedIds = ref<Set<number>>(new Set());

  const pageSize = computed(() => {
    try {
      return useSettingsStore().settings.resultsPerPage;
    } catch {
      return DEFAULT_PAGE_SIZE;
    }
  });
  const totalPages = computed(() =>
    Math.max(1, Math.ceil(hits.value.length / pageSize.value)),
  );
  const pageHits = computed<SearchHit[]>(() => {
    const start = (page.value - 1) * pageSize.value;
    return hits.value.slice(start, start + pageSize.value);
  });

  async function loadThumbnail(imageId: number, maxSize = 256): Promise<string | null> {
    if (thumbnailCache.value.has(imageId)) {
      return thumbnailCache.value.get(imageId) ?? null;
    }
    try {
      const url = await getThumbnail(imageId, maxSize);
      thumbnailCache.value.set(imageId, url);
      return url;
    } catch {
      return null;
    }
  }

  function clear() {
    hits.value = [];
    page.value = 1;
    selectedIds.value = new Set();
    lastError.value = null;
  }

  function goToPage(p: number) {
    if (p < 1 || p > totalPages.value) return;
    page.value = p;
  }
  function nextPage() {
    goToPage(page.value + 1);
  }
  function prevPage() {
    goToPage(page.value - 1);
  }

  function toggleSelected(imageId: number) {
    const s = new Set(selectedIds.value);
    if (s.has(imageId)) s.delete(imageId);
    else s.add(imageId);
    selectedIds.value = s;
  }
  function clearSelection() {
    selectedIds.value = new Set();
  }
  function selectAll() {
    selectedIds.value = new Set(hits.value.map((h) => h.image_id));
  }
  function isSelected(imageId: number): boolean {
    return selectedIds.value.has(imageId);
  }
  function selectPage() {
    const s = new Set(selectedIds.value);
    for (const h of pageHits.value) s.add(h.image_id);
    selectedIds.value = s;
  }
  function unselectPage() {
    const s = new Set(selectedIds.value);
    for (const h of pageHits.value) s.delete(h.image_id);
    selectedIds.value = s;
  }

  async function searchByImage(bytes: Uint8Array, topK = 30) {
    isSearching.value = true;
    lastError.value = null;
    try {
      hits.value = await searchByFaceImage(bytes, topK);
      page.value = 1;
      clearSelection();
    } catch (e) {
      lastError.value = String(e);
      throw e;
    } finally {
      isSearching.value = false;
    }
  }

  async function searchByEmbedding(embedding: number[], topK = 30) {
    isSearching.value = true;
    lastError.value = null;
    try {
      hits.value = await searchByFaceEmbedding(embedding, topK);
      page.value = 1;
      clearSelection();
    } catch (e) {
      lastError.value = String(e);
      throw e;
    } finally {
      isSearching.value = false;
    }
  }

  async function listByClass(classId: number, limit = 60) {
    isSearching.value = true;
    lastError.value = null;
    classIdInput.value = classId;
    try {
      hits.value = await listObjectsByClass(classId, limit);
      page.value = 1;
      clearSelection();
    } catch (e) {
      lastError.value = String(e);
      throw e;
    } finally {
      isSearching.value = false;
    }
  }

  /**
   * Phase 5: 用 person_id 列出该 person 出现过的所有 image。
   *
   * 后端走 multi-prototype HNSW 聚合 — frontal/left/right/high_quality/general
   * 各做一次粗筛,然后按 image_id 取 max cosine。比 searchByImage 更适合
   * "找这个人的所有照片" 这种 person-centric 用例。
   */
  async function searchByPersonId(personId: number, topK = 100) {
    isSearching.value = true;
    lastError.value = null;
    try {
      hits.value = await searchByPerson(personId, topK);
      page.value = 1;
      clearSelection();
    } catch (e) {
      lastError.value = String(e);
      throw e;
    } finally {
      isSearching.value = false;
    }
  }

  /**
   * 用图片 bytes 做对象向量检索（MobileCLIP + 几何 rerank）。
   * 流程：write_query_image（持久化）→ search_objects（查 HNSW）→ 映射到 SearchHit。
   */
  async function searchByObjectImage(bytes: Uint8Array, mime: string, topK = 30) {
    isSearching.value = true;
    lastError.value = null;
    try {
      const path = await writeQueryImage({
        data: Array.from(bytes),
        mime,
        crop: null,
      });
      const objHits = await searchObjects({ query_image: path, top_k: topK });
      hits.value = objHits.map((h: ObjectSearchHit, idx: number) => ({
        image_id: h.image_id,
        target_id: h.image_id,
        score: h.similarity,
        rank: idx,
      }));
      page.value = 1;
      clearSelection();
    } catch (e) {
      lastError.value = String(e);
      throw e;
    } finally {
      isSearching.value = false;
    }
  }

  /**
   * 导出选中的图片到目标目录（image_id → DB lookup → file copy）。
   * Rust 端 `copy_files` 支持 `image_ids`，不需要前端知道绝对路径。
   */
  async function exportSelected(destDir: string): Promise<CopyFileResult[]> {
    const ids = Array.from(selectedIds.value);
    if (ids.length === 0) return [];
    return await copyFiles({ image_ids: ids, dest_dir: destDir });
  }

  return {
    mode,
    isSearching,
    lastError,
    hits,
    thumbnailCache,
    classIdInput,
    loadThumbnail,
    clear,
    searchByImage,
    searchByEmbedding,
    listByClass,
    searchByPersonId,
    searchByObjectImage,
    page,
    pageSize,
    totalPages,
    pageHits,
    goToPage,
    nextPage,
    prevPage,
    selectedIds,
    toggleSelected,
    clearSelection,
    selectAll,
    selectPage,
    unselectPage,
    isSelected,
    exportSelected,
  };
});