<script setup lang="ts">
// 导出工具条 — 选择状态 + 全选/清空 + 目录选择 + 导出。
//
// 流程：
// 1) 用户在 ImageGrid 中点选 ResultCard → selectedIds 更新
// 2) 用户点「导出」 → 选目录（dialog） → 调 search.exportSelected(dir)
// 3) 显示 toast 报告成功/失败计数

import { computed, ref } from 'vue';
import { open as openDialog } from '@tauri-apps/plugin-dialog';
import { useSearchStore } from '@/stores/search';
import { useToastStore } from '@/stores/toast';
import type { CopyFileResult } from '@/api';

const search = useSearchStore();
const toast = useToastStore();
const exporting = ref(false);

const selectedCount = computed(() => search.selectedIds.size);
const totalCount = computed(() => search.hits.length);
const pageSelectedCount = computed(() =>
  search.pageHits.filter((h: { image_id: number }) => search.isSelected(h.image_id)).length,
);

function selectAll() {
  search.selectAll();
}
function unselectAll() {
  search.clearSelection();
}
function selectPage() {
  search.selectPage();
}
function unselectPage() {
  search.unselectPage();
}

async function pickAndExport() {
  if (selectedCount.value === 0) {
    toast.warning('请先选中要导出的图片');
    return;
  }
  let dir: string | null = null;
  try {
    const r = await openDialog({ directory: true, multiple: false, title: '选择导出目录' });
    if (typeof r === 'string') dir = r;
  } catch (e) {
    toast.danger('打开目录选择失败: ' + String(e));
    return;
  }
  if (!dir) return;
  exporting.value = true;
  try {
    const results = await search.exportSelected(dir);
    const ok = results.filter((r: CopyFileResult) => r.success).length;
    const fail = results.length - ok;
    if (fail === 0) {
      toast.success(`已导出 ${ok} 个文件到 ${dir}`);
    } else {
      toast.warning(`导出 ${ok} 个，失败 ${fail} 个`);
    }
  } catch (e) {
    toast.danger('导出失败: ' + String(e));
  } finally {
    exporting.value = false;
  }
}
</script>

<template>
  <div class="export-bar">
    <div class="export-summary">
      <span class="text-sm muted">
        已选 <span class="font-mono count">{{ selectedCount }}</span> / {{ totalCount }}
      </span>
    </div>

    <div class="export-actions">
      <button
        v-if="pageSelectedCount > 0"
        class="btn sm ghost"
        @click="unselectPage"
      >取消当前页</button>
      <button
        v-else
        class="btn sm ghost"
        :disabled="search.pageHits.length === 0"
        @click="selectPage"
      >选择当前页</button>

      <button
        v-if="selectedCount > 0"
        class="btn sm ghost"
        @click="unselectAll"
      >清空选择</button>
      <button
        v-else
        class="btn sm ghost"
        :disabled="totalCount === 0"
        @click="selectAll"
      >全选</button>

      <button
        class="btn sm primary"
        :disabled="selectedCount === 0 || exporting"
        @click="pickAndExport"
      >{{ exporting ? '导出中…' : `导出 (${selectedCount})` }}</button>
    </div>
  </div>
</template>

<style scoped>
.export-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: var(--space-3);
  padding: var(--space-2) var(--space-3);
  background: var(--color-bg-1);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-md);
  flex-wrap: wrap;
}
.export-summary {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}
.count {
  font-weight: 600;
  color: var(--color-accent);
}
.export-actions {
  display: flex;
  align-items: center;
  gap: var(--space-2);
  flex-wrap: wrap;
}
</style>