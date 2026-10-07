<script setup lang="ts">
// PersonSearchView — 上传一张照片 → 用其人脸 embedding 检索。
//
// UX：选文件（accept=image/*）→ 调 search.searchByImage(bytes)
// 同一 view 也支持拖拽。

import { onMounted, ref } from 'vue';
import { useSearchStore } from '@/stores/search';
import { useSettingsStore } from '@/stores/settings';
import { useToastStore } from '@/stores/toast';
import ImageGrid from '@/components/ImageGrid.vue';
import EmptyState from '@/components/EmptyState.vue';
import ExportBar from '@/components/ExportBar.vue';

const search = useSearchStore();
const settings = useSettingsStore();
const toast = useToastStore();
const fileInput = ref<HTMLInputElement | null>(null);
const previewUrl = ref<string | null>(null);
const topK = ref(settings.settings.topK);

onMounted(() => search.clear());

async function pickFile() {
  fileInput.value?.click();
}

async function onFile(e: Event) {
  const input = e.target as HTMLInputElement;
  const file = input.files?.[0];
  if (!file) return;
  await runWith(file);
  // reset so same file can be picked again
  input.value = '';
}

async function runWith(file: File) {
  try {
    const buf = new Uint8Array(await file.arrayBuffer());
    previewUrl.value = URL.createObjectURL(file);
    await search.searchByImage(buf, topK.value);
    if (search.hits.length === 0) {
      toast.info('未找到相似人脸（库中可能还没有人脸数据）');
    } else {
      toast.success(`找到 ${search.hits.length} 个相似结果`);
    }
  } catch (e) {
    toast.danger('检索失败: ' + String(e));
  }
}

// Drag & drop
const dragOver = ref(false);
function onDragOver(e: DragEvent) {
  e.preventDefault();
  dragOver.value = true;
}
function onDragLeave() {
  dragOver.value = false;
}
async function onDrop(e: DragEvent) {
  e.preventDefault();
  dragOver.value = false;
  const file = e.dataTransfer?.files?.[0];
  if (file && file.type.startsWith('image/')) {
    await runWith(file);
  } else {
    toast.warning('请拖入图片文件');
  }
}
</script>

<template>
  <div class="view">
    <header class="view-header">
      <h1>人脸检索</h1>
      <p class="muted text-sm">上传一张人脸照片，从图库中找出最相似的 N 张。</p>
    </header>

    <section class="card">
      <div class="card-header">
        <div>
          <div class="card-title">查询</div>
          <div class="card-subtitle">图片会先检测人脸，再用 ArcFace embedding 检索</div>
        </div>
        <div class="row gap-2">
          <label class="muted text-sm">Top K</label>
          <input v-model.number="topK" type="number" class="input" min="1" max="200" style="width: 80px;" />
        </div>
      </div>

      <div
        :class="['dropzone', { over: dragOver }]"
        @click="pickFile"
        @dragover="onDragOver"
        @dragleave="onDragLeave"
        @drop="onDrop"
      >
        <input
          ref="fileInput"
          type="file"
          accept="image/*"
          style="display: none;"
          @change="onFile"
        />
        <img v-if="previewUrl" :src="previewUrl" class="preview" />
        <div v-else class="dropzone-hint">
          <div class="text-lg">点击或拖入图片</div>
          <div class="muted text-sm">支持 JPG / PNG / WebP</div>
        </div>
      </div>
    </section>

    <section class="card">
      <div class="card-header">
        <div>
          <div class="card-title">结果</div>
          <div class="card-subtitle">{{ search.isSearching ? '检索中…' : `${search.hits.length} 个结果` }}</div>
        </div>
      </div>
      <ExportBar v-if="search.hits.length > 0" />
      <ImageGrid v-if="search.hits.length > 0" :hits="search.hits" />
      <EmptyState v-else title="还没有结果" hint="上传一张图片开始检索" />
    </section>
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
.dropzone {
  border: 2px dashed var(--color-border-strong);
  border-radius: var(--radius-lg);
  min-height: 200px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  background: var(--color-bg-2);
  transition: border-color 120ms, background 120ms;
  overflow: hidden;
}
.dropzone:hover,
.dropzone.over {
  border-color: var(--color-accent);
  background: var(--color-accent-soft);
}
.dropzone-hint {
  text-align: center;
  display: flex;
  flex-direction: column;
  gap: var(--space-1);
}
.preview {
  max-height: 320px;
  object-fit: contain;
}
</style>
