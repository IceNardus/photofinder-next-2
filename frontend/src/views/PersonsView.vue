<script setup lang="ts">
// PersonsView — 人物列表 + 重命名 + 触发聚类 + 详情（faces of person）。
//
// 左侧 list，右侧 detail panel。

import { onMounted, ref } from 'vue';
import { useRouter } from 'vue-router';
import { usePersonStore } from '@/stores/person';
import { useSearchStore } from '@/stores/search';
import { useToastStore } from '@/stores/toast';
import EmptyState from '@/components/EmptyState.vue';
import { getThumbnail } from '@/api';

const person = usePersonStore();
const search = useSearchStore();
const toast = useToastStore();
const router = useRouter();

const editingId = ref<number | null>(null);
const editingName = ref('');

async function reload() {
  await person.refresh();
}

async function startRename(id: number, current: string | null) {
  editingId.value = id;
  editingName.value = current ?? '';
}

async function commitRename(id: number) {
  const name = editingName.value.trim();
  try {
    await person.rename(id, name.length > 0 ? name : null);
    editingId.value = null;
    toast.success('已重命名');
  } catch (e) {
    toast.danger('重命名失败: ' + String(e));
  }
}

function cancelRename() {
  editingId.value = null;
  editingName.value = '';
}

async function triggerCluster() {
  try {
    const id = await person.cluster();
    toast.success(`聚类任务已入队 #${id}`);
  } catch (e) {
    toast.danger('聚类失败: ' + String(e));
  }
}

async function triggerRebuild() {
  try {
    const result = await person.rebuildClusters();
    toast.success(`重新分析完成: 分配 ${result.assigned} 人脸, 新建 ${result.created} 人物`);
    await reload();
  } catch (e) {
    toast.danger('重新分析失败: ' + String(e));
  }
}

/**
 * Phase 5: 查找此 person 出现过的所有 image,跳转到检索结果 view。
 *
 * 多 prototype 召回:frontal/left/right/high_quality/general 各自做 HNSW 粗筛,
 * 聚合 per-image max cosine。比 `facesOfPerson` (DB 直查) 召回更广 —
 * 库中已索引但未聚类到此 person 的 face 也能找回(只要 cosine 够高)。
 */
async function findPhotosOfPerson(id: number) {
  try {
    await search.searchByPersonId(id, 100);
    if (search.hits.length === 0) {
      toast.info('未找到该人物的更多照片');
      return;
    }
    toast.success(`找到 ${search.hits.length} 张包含该人物的照片`);
    await router.push({ name: 'faces' });
  } catch (e) {
    toast.danger('搜索失败: ' + String(e));
  }
}

async function selectPerson(id: number) {
  await person.selectPerson(id);
}

// 缩略图懒加载（person.faces）
const faceUrls = ref<Map<number, string>>(new Map());
async function loadFaceThumb(faceId: number, imageId: number) {
  if (faceUrls.value.has(faceId)) return;
  try {
    const url = await getThumbnail(imageId, 160);
    faceUrls.value.set(faceId, url);
  } catch {
    /* ignore */
  }
}

onMounted(reload);
</script>

<template>
  <div class="view">
    <header class="view-header">
      <h1>人物</h1>
      <p class="muted text-sm">查看聚类结果、给人物命名、查看其所有人脸。</p>
    </header>

    <div class="layout">
      <!-- ============ Left: list ============ -->
      <section class="card list">
        <div class="card-header">
          <div>
            <div class="card-title">人物列表</div>
            <div class="card-subtitle">{{ person.persons.length }} 个人物</div>
          </div>
          <div class="row gap-2">
            <button class="btn sm ghost" :disabled="person.isLoading" @click="reload">刷新</button>
            <button class="btn sm primary" @click="triggerCluster">聚类</button>
            <button class="btn sm" :disabled="person.isLoading" @click="triggerRebuild">重新分析人物</button>
          </div>
        </div>

        <EmptyState
          v-if="!person.isLoading && person.persons.length === 0"
          title="还没有人物"
          hint="先扫描目录，再点上方「聚类」按钮"
        />
        <ul v-else class="person-list">
          <li
            v-for="p in person.persons"
            :key="p.id"
            :class="['person-item', { active: person.selectedPersonId === p.id }]"
            @click="selectPerson(p.id)"
          >
            <div class="person-meta">
              <div v-if="editingId === p.id" class="row gap-1" @click.stop>
                <input
                  v-model="editingName"
                  class="input sm"
                  placeholder="名字"
                  @keyup.enter="commitRename(p.id)"
                  @keyup.escape="cancelRename"
                />
                <button class="btn sm primary" @click="commitRename(p.id)">保存</button>
                <button class="btn sm ghost" @click="cancelRename">取消</button>
              </div>
              <div v-else class="row between flex-1">
                <div>
                  <div class="person-name">{{ p.name ?? `Person #${p.id}` }}</div>
                  <div class="muted text-sm">#{{ p.id }} · {{ p.face_count }} 张脸</div>
                </div>
                <button class="btn sm ghost" @click.stop="startRename(p.id, p.name)">重命名</button>
              </div>
            </div>
          </li>
        </ul>
      </section>

      <!-- ============ Right: detail ============ -->
      <section class="card detail">
        <div class="card-header">
          <div>
            <div class="card-title">人脸详情</div>
            <div v-if="person.selectedPersonId != null" class="card-subtitle">
              {{ person.selectedFaces.length }} 张脸 · #{{ person.selectedPersonId }}
            </div>
            <div v-else class="card-subtitle">选择左侧人物</div>
          </div>
          <div v-if="person.selectedPersonId != null" class="row gap-2">
            <button
              class="btn sm primary"
              @click="findPhotosOfPerson(person.selectedPersonId)"
            >
              查找所有照片
            </button>
          </div>
        </div>
        <div v-if="person.isLoadingFaces" class="muted">加载中…</div>
        <EmptyState
          v-else-if="person.selectedFaces.length === 0"
          title="无内容"
          hint="选择左侧一个人物查看其人脸"
        />
        <div v-else class="face-grid">
          <div
            v-for="f in person.selectedFaces"
            :key="f.id"
            class="face-cell"
          >
            <img v-if="faceUrls.get(f.id)" :src="faceUrls.get(f.id)" alt="" />
            <div v-else class="face-placeholder muted center" @mouseenter="loadFaceThumb(f.id, f.image_id)">…</div>
            <div class="image-cell-overlay">
              <span>score {{ f.detector_score.toFixed(2) }}</span>
              <span style="float: right;">q {{ f.quality.toFixed(2) }}</span>
            </div>
          </div>
        </div>
      </section>
    </div>
  </div>
</template>

<style scoped>
.view {
  display: flex;
  flex-direction: column;
  gap: var(--space-5);
  max-width: var(--content-max);
  margin: 0 auto;
  padding: var(--space-5);
}
.view-header h1 {
  font-size: var(--text-2xl);
  font-weight: 600;
  margin-bottom: var(--space-1);
}
.layout {
  display: grid;
  grid-template-columns: 360px 1fr;
  gap: var(--space-4);
}
@media (max-width: 800px) {
  .layout { grid-template-columns: 1fr; }
}
.person-list {
  list-style: none;
  max-height: 60vh;
  overflow-y: auto;
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
}
.person-item {
  padding: var(--space-3) var(--space-4);
  border-bottom: 1px solid var(--color-border);
  cursor: pointer;
  transition: background 120ms;
}
.person-item:hover { background: var(--color-bg-2); }
.person-item.active { background: var(--color-accent-soft); }
.person-meta { width: 100%; }
.person-name {
  font-size: var(--text-md);
  font-weight: 500;
}
.input.sm { height: 26px; padding: 0 var(--space-2); font-size: var(--text-sm); width: 100%; }
.face-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
  gap: var(--space-2);
}
.face-cell {
  position: relative;
  aspect-ratio: 1;
  background: var(--color-bg-2);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  overflow: hidden;
}
.face-cell img { width: 100%; height: 100%; object-fit: cover; }
.face-placeholder { width: 100%; height: 100%; }
</style>
