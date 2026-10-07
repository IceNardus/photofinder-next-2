<script setup lang="ts">
// 单个结果卡片 — 显示缩略图 + 分数 + image_id + 选择框。
//
// 颜色编码（按相似度 score ∈ [0,1]）：
//   ≥ 0.7 → 高 (green)
//   ≥ 0.5 → 中 (yellow)
//   < 0.5 → 低 (red)
//
// 缩略图懒加载（IntersectionObserver）。

import { computed, onMounted, onUnmounted, ref, watch } from 'vue';
import { useSearchStore } from '@/stores/search';
import { useSettingsStore } from '@/stores/settings';
import type { SearchHit } from '@/api';

const props = defineProps<{ hit: SearchHit }>();
const search = useSearchStore();
const settings = useSettingsStore();
const url = ref<string | null>(null);
const root = ref<HTMLElement | null>(null);
let observer: IntersectionObserver | null = null;

const selected = computed(() => search.isSelected(props.hit.image_id));
const colorTier = computed<'high' | 'mid' | 'low'>(() => {
  const high = settings.settings.similarityHigh;
  const mid = settings.settings.similarityMid;
  if (props.hit.score >= high) return 'high';
  if (props.hit.score >= mid) return 'mid';
  return 'low';
});

function startObserve() {
  if (!root.value || observer) return;
  observer = new IntersectionObserver(
    (entries) => {
      for (const e of entries) {
        if (e.isIntersecting) {
          observer?.disconnect();
          observer = null;
          void (async () => {
            url.value = await search.loadThumbnail(props.hit.image_id);
          })();
          break;
        }
      }
    },
    { rootMargin: '200px' },
  );
  observer.observe(root.value);
}

function onSelect(e: Event) {
  e.stopPropagation();
  search.toggleSelected(props.hit.image_id);
}

onMounted(startObserve);
onUnmounted(() => observer?.disconnect());

watch(() => props.hit.image_id, startObserve);
</script>

<template>
  <div
    ref="root"
    :class="['image-cell', `tier-${colorTier}`, { selected }]"
    @click="onSelect"
  >
    <img v-if="url" :src="url" :alt="`img-${hit.image_id}`" />
    <div v-else class="placeholder muted center">…</div>

    <!-- 选择 checkbox（左上角） -->
    <button
      :class="['select-toggle', { checked: selected }]"
      :aria-label="selected ? '取消选择' : '选择'"
      @click.stop="onSelect"
    >
      <span v-if="selected">✓</span>
    </button>

    <div class="image-cell-overlay">
      <span>#{{ hit.image_id }}</span>
      <span class="score font-mono">{{ (hit.score * 100).toFixed(1) }}%</span>
    </div>
  </div>
</template>

<style scoped>
.placeholder {
  width: 100%;
  height: 100%;
  background: var(--color-bg-3);
}

/* ===== 颜色编码 ===== */
.tier-high  { border: 2px solid var(--color-success); }
.tier-mid   { border: 2px solid var(--color-warning); }
.tier-low   { border: 2px solid var(--color-danger); }

.tier-high  .image-cell-overlay { background: rgba(52, 199, 89, 0.65); }
.tier-mid   .image-cell-overlay { background: rgba(255, 159, 10, 0.65); }
.tier-low   .image-cell-overlay { background: rgba(255, 69, 58, 0.65); }

.tier-high .score { color: #fff; font-weight: 700; }
.tier-mid  .score { color: #fff; font-weight: 700; }
.tier-low  .score { color: #fff; font-weight: 700; }

/* ===== 选择状态 ===== */
.image-cell {
  position: relative;
  cursor: pointer;
  border-radius: var(--radius-md);
  overflow: hidden;
  transition: outline 80ms;
}
.image-cell.selected {
  outline: 3px solid var(--color-accent);
  outline-offset: -3px;
}
.select-toggle {
  position: absolute;
  top: 6px;
  left: 6px;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  border: 2px solid rgba(255, 255, 255, 0.85);
  background: rgba(0, 0, 0, 0.4);
  color: white;
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 12px;
  cursor: pointer;
  padding: 0;
  z-index: 2;
  transition: background 120ms, border-color 120ms;
}
.select-toggle.checked {
  background: var(--color-accent);
  border-color: var(--color-accent);
}
</style>