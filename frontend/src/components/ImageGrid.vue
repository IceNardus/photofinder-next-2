<script setup lang="ts">
// 结果网格 — 包裹 ResultCard，自带空态 + 分页。
import ResultCard from './ResultCard.vue';
import { useSearchStore } from '@/stores/search';

const search = useSearchStore();
const pageHits = search.pageHits;
const page = search.page;
const totalPages = search.totalPages;
const hits = search.hits;
const total = hits.length;
</script>

<template>
  <div v-if="total === 0" class="empty">
    <div class="empty-title">暂无结果</div>
  </div>
  <div v-else class="grid-with-pages">
    <div class="image-grid">
      <ResultCard v-for="h in pageHits" :key="`${h.target_id}-${h.rank}-${page}`" :hit="h" />
    </div>

    <!-- 分页栏（只在 >1 页时显示） -->
    <div v-if="totalPages > 1" class="pager">
      <button
        class="btn sm ghost"
        :disabled="page <= 1"
        @click="search.prevPage()"
      >上一页</button>
      <span class="pager-info font-mono">
        {{ page }} / {{ totalPages }} · 共 {{ total }}
      </span>
      <button
        class="btn sm ghost"
        :disabled="page >= totalPages"
        @click="search.nextPage()"
      >下一页</button>
    </div>
  </div>
</template>

<style scoped>
.grid-with-pages {
  display: flex;
  flex-direction: column;
  gap: var(--space-3);
}
.pager {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: var(--space-3);
  padding-top: var(--space-2);
}
.pager-info {
  font-size: var(--text-sm);
  color: var(--color-fg-2);
  font-variant-numeric: tabular-nums;
  min-width: 100px;
  text-align: center;
}
</style>