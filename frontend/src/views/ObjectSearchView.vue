<script setup lang="ts">
// ObjectSearchView — 图片查询：
//   drop → CropModal → write_query_image → search_objects 向量检索 → 去重结果
//
// 结果展示复用分页 + 颜色编码 + 导出。

import { onMounted, ref } from 'vue';
import { useSearchStore } from '@/stores/search';
import { useSettingsStore } from '@/stores/settings';
import { useToastStore } from '@/stores/toast';
import ImageGrid from '@/components/ImageGrid.vue';
import EmptyState from '@/components/EmptyState.vue';
import ExportBar from '@/components/ExportBar.vue';
import CropModal from '@/components/CropModal.vue';

const search = useSearchStore();
const settings = useSettingsStore();
const toast = useToastStore();

onMounted(() => search.clear());

const limit = ref<number>(settings.settings.topK);

// ===== 图片查询（crop modal）=====
const fileInput = ref<HTMLInputElement | null>(null);
const dragOver = ref(false);
const cropSrc = ref<{
  url: string;
  mime: string;
  width: number;
  height: number;
} | null>(null);
const croppedPreview = ref<string | null>(null);

async function pickFile() {
  fileInput.value?.click();
}

async function onFile(e: Event) {
  const input = e.target as HTMLInputElement;
  const file = input.files?.[0];
  if (file) await openCrop(file);
  input.value = '';
}

async function onDrop(e: DragEvent) {
  e.preventDefault();
  dragOver.value = false;
  const file = e.dataTransfer?.files?.[0];
  if (file && file.type.startsWith('image/')) {
    await openCrop(file);
  } else {
    toast.warning('请拖入图片文件');
  }
}
function onDragOver(e: DragEvent) {
  e.preventDefault();
  dragOver.value = true;
}
function onDragLeave() {
  dragOver.value = false;
}

async function openCrop(file: File) {
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.src = url;
  await new Promise<void>((res, rej) => {
    img.onload = () => res();
    img.onerror = () => rej(new Error('load failed'));
  });
  cropSrc.value = {
    url,
    mime: file.type || 'image/png',
    width: img.naturalWidth,
    height: img.naturalHeight,
  };
}

function cancelCrop() {
  if (cropSrc.value) {
    URL.revokeObjectURL(cropSrc.value.url);
  }
  cropSrc.value = null;
}

async function confirmCrop(payload: {
  x: number;
  y: number;
  w: number;
  h: number;
  imageBlob: Blob;
  mime: string;
  previewUrl: string;
}) {
  // 先显示预览，再关 modal + 搜索
  croppedPreview.value = payload.previewUrl;
  cancelCrop();
  try {
    const buf = new Uint8Array(await payload.imageBlob.arrayBuffer());
    await search.searchByObjectImage(buf, payload.mime, limit.value);
    if (search.hits.length === 0) {
      toast.info('未找到相似对象');
    } else {
      toast.success(`对象检索: ${search.hits.length} 个相似结果`);
    }
  } catch (e) {
    toast.danger('对象检索失败: ' + String(e));
  }
}
</script>

<template>
  <div class="view">
    <header class="view-header">
      <h1>对象检索</h1>
      <p class="muted text-sm">拖入图片并裁剪目标区域，用 MobileCLIP 向量检索相似对象。</p>
    </header>

    <section class="card">
      <div class="card-header">
        <div>
          <div class="card-title">图片查询（向量检索）</div>
          <div class="card-subtitle">拖入或选择图片，裁剪后用 MobileCLIP 检索，重复照片只返回一张</div>
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
        <template v-if="croppedPreview">
          <img :src="croppedPreview" alt="裁剪预览" class="crop-thumb" />
          <div class="muted text-sm">点击重新选择图片</div>
        </template>
        <template v-else>
          <div class="dropzone-hint">
            <div class="text-lg">点击或拖入图片</div>
            <div class="muted text-sm">会弹出裁剪框，裁剪后用于对象向量检索</div>
          </div>
        </template>
      </div>
    </section>

    <section class="card">
      <div class="card-header">
        <div>
          <div class="card-title">结果</div>
          <div class="card-subtitle">
            {{ search.isSearching ? '加载中…' : `${search.hits.length} 张` }}
          </div>
        </div>
      </div>
      <ExportBar v-if="search.hits.length > 0" />
      <ImageGrid v-if="search.hits.length > 0" :hits="search.hits" />
      <EmptyState v-else title="还没有结果" hint="拖入图片开始检索" />
    </section>

    <CropModal
      v-if="cropSrc"
      :image-src="cropSrc.url"
      :mime="cropSrc.mime"
      :image-width="cropSrc.width"
      :image-height="cropSrc.height"
      @cancel="cancelCrop"
      @confirm="confirmCrop"
    />
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
  min-height: 160px;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  background: var(--color-bg-2);
  transition: border-color 120ms, background 120ms;
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
.crop-thumb {
  max-width: 120px;
  max-height: 120px;
  border-radius: var(--radius-md);
  border: 2px solid var(--color-border);
  box-shadow: var(--shadow-sm);
  object-fit: cover;
}
</style>