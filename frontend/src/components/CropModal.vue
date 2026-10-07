<script setup lang="ts">
// CropModal — 8-handle resizable crop box overlay on an image.
//
// 用法：drop image → open CropModal → 用户拖拽调整 → confirm → 生成预览 → 用户确认搜索

import { computed, onMounted, onUnmounted, ref } from 'vue';

interface CropBox {
  x: number;
  y: number;
  w: number;
  h: number;
}

interface ConfirmPayload {
  x: number;
  y: number;
  w: number;
  h: number;
  imageBlob: Blob;
  mime: string;
  previewUrl: string;
}

const props = defineProps<{
  imageSrc: string;
  mime: string;
  imageWidth: number;
  imageHeight: number;
}>();

const emit = defineEmits<{
  (e: 'cancel'): void;
  (e: 'confirm', payload: ConfirmPayload): void;
}>();

const HANDLE_SIZE = 14;
const MIN_DRAG = 20;
const MIN_RUST = 32;

const stage = ref<HTMLElement | null>(null);
const stageSize = ref({ w: 0, h: 0 });

const displayed = computed(() => {
  const sw = stageSize.value.w;
  const sh = stageSize.value.h;
  if (sw === 0 || sh === 0 || props.imageWidth === 0 || props.imageHeight === 0) {
    return { w: 0, h: 0, offsetX: 0, offsetY: 0 };
  }
  const s = Math.min(sw / props.imageWidth, sh / props.imageHeight);
  const dw = Math.round(props.imageWidth * s);
  const dh = Math.round(props.imageHeight * s);
  return { w: dw, h: dh, offsetX: (sw - dw) / 2, offsetY: (sh - dh) / 2 };
});

const scale = computed(() => {
  if (props.imageWidth === 0) return 1;
  return displayed.value.w / props.imageWidth;
});

const cropImageSize = computed(() => {
  const sx = 1 / scale.value;
  return {
    w: Math.round(crop.value.w * sx),
    h: Math.round(crop.value.h * sx),
  };
});

const crop = ref<CropBox>({ x: 0, y: 0, w: 0, h: 0 });

type DragMode = 'move' | 'nw' | 'ne' | 'sw' | 'se' | 'n' | 's' | 'e' | 'w' | null;
const drag = ref<{
  mode: DragMode;
  startX: number;
  startY: number;
  startBox: CropBox;
} | null>(null);

const cropStyle = computed<Record<string, string>>(() => ({
  left: `${crop.value.x}px`,
  top: `${crop.value.y}px`,
  width: `${crop.value.w}px`,
  height: `${crop.value.h}px`,
}));

// 遮罩覆盖整个 stage
function shadeStyle(_side: 'top' | 'bottom' | 'left' | 'right'): Record<string, string> {
  return { inset: '0' };
}

function onResize() {
  const el = stage.value;
  if (!el) return;
  stageSize.value = { w: el.clientWidth, h: el.clientHeight };
}

onMounted(() => {
  onResize();
  window.addEventListener('resize', onResize);
  const d = displayed.value;
  crop.value = {
    x: d.offsetX + d.w * 0.2,
    y: d.offsetY + d.h * 0.2,
    w: d.w * 0.6,
    h: d.h * 0.6,
  };
});
onUnmounted(() => window.removeEventListener('resize', onResize));

function clampBox(b: CropBox): CropBox {
  const d = displayed.value;
  const minX = d.offsetX;
  const minY = d.offsetY;
  const maxX = d.offsetX + d.w;
  const maxY = d.offsetY + d.h;
  const minSide = Math.max(HANDLE_SIZE, MIN_DRAG);
  let { x, y, w, h } = b;
  w = Math.max(minSide, Math.min(d.w, w));
  h = Math.max(minSide, Math.min(d.h, h));
  x = Math.max(minX, Math.min(maxX - w, x));
  y = Math.max(minY, Math.min(maxY - h, y));
  return { x, y, w, h };
}

function startDrag(mode: Exclude<DragMode, null>, e: PointerEvent) {
  e.preventDefault();
  e.stopPropagation();
  drag.value = { mode, startX: e.clientX, startY: e.clientY, startBox: { ...crop.value } };
  (e.currentTarget as HTMLElement).setPointerCapture?.(e.pointerId);
}

function onMove(e: PointerEvent) {
  if (!drag.value) return;
  const dx = e.clientX - drag.value.startX;
  const dy = e.clientY - drag.value.startY;
  const sb = drag.value.startBox;
  let next: CropBox;
  switch (drag.value.mode) {
    case 'move': next = { ...sb, x: sb.x + dx, y: sb.y + dy }; break;
    case 'nw':   next = { x: sb.x + dx, y: sb.y + dy, w: sb.w - dx, h: sb.h - dy }; break;
    case 'ne':   next = { y: sb.y + dy, w: sb.w + dx, h: sb.h - dy, x: sb.x }; break;
    case 'sw':   next = { x: sb.x + dx, w: sb.w - dx, h: sb.h + dy, y: sb.y }; break;
    case 'se':   next = { w: sb.w + dx, h: sb.h + dy, x: sb.x, y: sb.y }; break;
    case 'n':    next = { ...sb, y: sb.y + dy, h: sb.h - dy }; break;
    case 's':    next = { ...sb, h: sb.h + dy }; break;
    case 'e':    next = { ...sb, w: sb.w + dx }; break;
    case 'w':    next = { ...sb, x: sb.x + dx, w: sb.w - dx }; break;
    default: return;
  }
  crop.value = clampBox(next);
}

function endDrag() { drag.value = null; }

async function onConfirm() {
  const bx = crop.value;
  const minBox = clampBox({ ...bx, w: Math.max(bx.w, MIN_RUST), h: Math.max(bx.h, MIN_RUST) });
  crop.value = minBox;

  const sx = 1 / scale.value;
  const d = displayed.value;
  const ix = Math.round((bx.x - d.offsetX) * sx);
  const iy = Math.round((bx.y - d.offsetY) * sx);
  const iw = Math.round(bx.w * sx);
  const ih = Math.round(bx.h * sx);

  const img = new Image();
  img.src = props.imageSrc;
  await new Promise<void>((res, rej) => {
    img.onload = () => res();
    img.onerror = () => rej(new Error('image load failed'));
  });

  const canvas = document.createElement('canvas');
  canvas.width = iw;
  canvas.height = ih;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(img, ix, iy, iw, ih, 0, 0, iw, ih);

  const blob: Blob = await new Promise((res, rej) =>
    canvas.toBlob((b) => b ? res(b) : rej(new Error('toBlob null')), props.mime, 0.92),
  );

  const previewUrl = await new Promise<string>((res) => {
    const reader = new FileReader();
    reader.onloadend = () => res(reader.result as string);
    reader.readAsDataURL(blob);
  });

  emit('confirm', { x: ix, y: iy, w: iw, h: ih, imageBlob: blob, mime: props.mime, previewUrl });
}
</script>

<template>
  <div class="crop-modal-backdrop" @click.self="emit('cancel')">
    <div class="crop-modal">
      <header class="crop-modal-head">
        <div class="crop-modal-title">裁剪查询区域</div>
        <div class="crop-info font-mono text-sm">
          <span class="crop-dim">{{ cropImageSize.w }} × {{ cropImageSize.h }}</span>
          <span v-if="cropImageSize.w < MIN_RUST || cropImageSize.h < MIN_RUST" class="crop-warn">
            （最小 {{ MIN_RUST }}px）
          </span>
          <span class="crop-hint muted">拖拽手柄调整区域</span>
        </div>
      </header>

      <div ref="stage" class="crop-stage">
        <img :src="imageSrc" class="crop-image" alt="crop source" draggable="false" />
        <div class="crop-box" :style="cropStyle">
          <div class="crop-shade" :style="shadeStyle('top')" />
          <div class="crop-shade" :style="shadeStyle('bottom')" />
          <div class="crop-shade" :style="shadeStyle('left')" />
          <div class="crop-shade" :style="shadeStyle('right')" />
          <div class="thirds-lines">
            <div class="thirds-v" :style="{ left: `${(1/3)*100}%` }" />
            <div class="thirds-v" :style="{ left: `${(2/3)*100}%` }" />
            <div class="thirds-h" :style="{ top: `${(1/3)*100}%` }" />
            <div class="thirds-h" :style="{ top: `${(2/3)*100}%` }" />
          </div>
          <div
            class="crop-handle crop-handle-move"
            @pointerdown="startDrag('move', $event)"
            @pointermove="onMove"
            @pointerup="endDrag"
            @pointercancel="endDrag"
          />
          <div
            v-for="h in (['nw','n','ne','e','se','s','sw','w'] as const)"
            :key="h"
            :class="['crop-handle', `crop-handle-${h}`]"
            @pointerdown="startDrag(h, $event)"
            @pointermove="onMove"
            @pointerup="endDrag"
            @pointercancel="endDrag"
          />
        </div>
      </div>

      <footer class="crop-modal-foot">
        <button class="btn ghost" @click="emit('cancel')">取消</button>
        <button class="btn primary" @click="onConfirm">使用此区域搜索</button>
      </footer>
    </div>
  </div>
</template>

<style scoped>
.crop-modal-backdrop {
  position: fixed;
  inset: 0;
  background: rgba(0, 0, 0, 0.8);
  display: flex;
  align-items: center;
  justify-content: center;
  z-index: 100;
}
.crop-modal {
  display: flex;
  flex-direction: column;
  background: var(--color-bg-1);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  width: min(92vw, 900px);
  max-height: 92vh;
  overflow: hidden;
  box-shadow: var(--shadow-lg);
}
.crop-modal-head,
.crop-modal-foot {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: var(--space-3) var(--space-4);
  border-bottom: 1px solid var(--color-border);
  gap: var(--space-3);
}
.crop-modal-foot {
  border-bottom: none;
  border-top: 1px solid var(--color-border);
  justify-content: flex-end;
}
.crop-modal-title {
  font-size: var(--text-md);
  font-weight: 600;
}
.crop-info {
  display: flex;
  align-items: center;
  gap: var(--space-2);
}
.crop-dim { color: var(--color-fg-1); }
.crop-warn { color: var(--color-warning); }
.crop-hint { font-size: var(--text-xs); }
.crop-stage {
  position: relative;
  width: 100%;
  max-height: 75vh;
  background: #000;
  display: flex;
  align-items: center;
  justify-content: center;
  overflow: hidden;
  user-select: none;
}
.crop-image {
  max-width: 100%;
  max-height: 75vh;
  display: block;
  pointer-events: none;
}
.crop-box {
  position: absolute;
  border: 2px solid var(--color-accent);
  box-shadow: 0 0 0 9999px rgba(0, 0, 0, 0.2);
  box-sizing: border-box;
}
.crop-shade {
  position: absolute;
  background: rgba(0, 0, 0, 0.5);
  pointer-events: none;
}
/* 三等分辅助线 */
.thirds-lines { position: absolute; inset: 0; pointer-events: none; }
.thirds-v {
  position: absolute; top: 0; bottom: 0; width: 1px;
  background: rgba(255,255,255,0.4);
}
.thirds-h {
  position: absolute; left: 0; right: 0; height: 1px;
  background: rgba(255,255,255,0.4);
}
/* 手柄 */
.crop-handle {
  position: absolute;
  background: var(--color-accent);
  border: 2px solid white;
  border-radius: 3px;
  box-shadow: 0 1px 4px rgba(0,0,0,0.5);
  box-sizing: border-box;
}
.crop-handle-move {
  position: absolute; inset: 0; background: transparent; cursor: move;
}
.crop-handle-move::before {
  /* 中心十字提示 */
  content: ''; position: absolute;
  left: 50%; top: 50%; width: 20px; height: 20px;
  transform: translate(-50%, -50%);
  background: radial-gradient(circle, rgba(255,255,255,0.6) 0%, transparent 70%);
}
.crop-handle-nw { top: -7px;    left: -7px;   width: 14px; height: 14px; cursor: nwse-resize; }
.crop-handle-n  { top: -7px;    left: 50%;    width: 14px; height: 14px; margin-left: -7px; cursor: ns-resize; }
.crop-handle-ne { top: -7px;    right: -7px;  width: 14px; height: 14px; cursor: nesw-resize; }
.crop-handle-e  { top: 50%;     right: -7px;  width: 14px; height: 14px; margin-top: -7px; cursor: ew-resize; }
.crop-handle-se { bottom: -7px; right: -7px;  width: 14px; height: 14px; cursor: nwse-resize; }
.crop-handle-s  { bottom: -7px; left: 50%;    width: 14px; height: 14px; margin-left: -7px; cursor: ns-resize; }
.crop-handle-sw { bottom: -7px; left: -7px;   width: 14px; height: 14px; cursor: nesw-resize; }
.crop-handle-w  { top: 50%;     left: -7px;   width: 14px; height: 14px; margin-top: -7px; cursor: ew-resize; }
</style>