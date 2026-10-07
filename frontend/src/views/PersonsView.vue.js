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
const editingId = ref(null);
const editingName = ref('');
async function reload() {
    await person.refresh();
}
async function startRename(id, current) {
    editingId.value = id;
    editingName.value = current ?? '';
}
async function commitRename(id) {
    const name = editingName.value.trim();
    try {
        await person.rename(id, name.length > 0 ? name : null);
        editingId.value = null;
        toast.success('已重命名');
    }
    catch (e) {
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
    }
    catch (e) {
        toast.danger('聚类失败: ' + String(e));
    }
}
async function triggerRebuild() {
    try {
        const result = await person.rebuildClusters();
        toast.success(`重新分析完成: 分配 ${result.assigned} 人脸, 新建 ${result.created} 人物`);
        await reload();
    }
    catch (e) {
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
async function findPhotosOfPerson(id) {
    try {
        await search.searchByPersonId(id, 100);
        if (search.hits.length === 0) {
            toast.info('未找到该人物的更多照片');
            return;
        }
        toast.success(`找到 ${search.hits.length} 张包含该人物的照片`);
        await router.push({ name: 'faces' });
    }
    catch (e) {
        toast.danger('搜索失败: ' + String(e));
    }
}
async function selectPerson(id) {
    await person.selectPerson(id);
}
// 缩略图懒加载（person.faces）
const faceUrls = ref(new Map());
async function loadFaceThumb(faceId, imageId) {
    if (faceUrls.value.has(faceId))
        return;
    try {
        const url = await getThumbnail(imageId, 160);
        faceUrls.value.set(faceId, url);
    }
    catch {
        /* ignore */
    }
}
onMounted(reload);
debugger; /* PartiallyEnd: #3632/scriptSetup.vue */
const __VLS_ctx = {};
let __VLS_components;
let __VLS_directives;
/** @type {__VLS_StyleScopedClasses['layout']} */ ;
/** @type {__VLS_StyleScopedClasses['person-item']} */ ;
/** @type {__VLS_StyleScopedClasses['person-item']} */ ;
/** @type {__VLS_StyleScopedClasses['face-cell']} */ ;
// CSS variable injection 
// CSS variable injection end 
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "view" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.header, __VLS_intrinsicElements.header)({
    ...{ class: "view-header" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.h1, __VLS_intrinsicElements.h1)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.p, __VLS_intrinsicElements.p)({
    ...{ class: "muted text-sm" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "layout" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.section, __VLS_intrinsicElements.section)({
    ...{ class: "card list" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "card-header" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "card-title" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "card-subtitle" },
});
(__VLS_ctx.person.persons.length);
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "row gap-2" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
    ...{ onClick: (__VLS_ctx.reload) },
    ...{ class: "btn sm ghost" },
    disabled: (__VLS_ctx.person.isLoading),
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
    ...{ onClick: (__VLS_ctx.triggerCluster) },
    ...{ class: "btn sm primary" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
    ...{ onClick: (__VLS_ctx.triggerRebuild) },
    ...{ class: "btn sm" },
    disabled: (__VLS_ctx.person.isLoading),
});
if (!__VLS_ctx.person.isLoading && __VLS_ctx.person.persons.length === 0) {
    /** @type {[typeof EmptyState, ]} */ ;
    // @ts-ignore
    const __VLS_0 = __VLS_asFunctionalComponent(EmptyState, new EmptyState({
        title: "还没有人物",
        hint: "先扫描目录，再点上方「聚类」按钮",
    }));
    const __VLS_1 = __VLS_0({
        title: "还没有人物",
        hint: "先扫描目录，再点上方「聚类」按钮",
    }, ...__VLS_functionalComponentArgsRest(__VLS_0));
}
else {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.ul, __VLS_intrinsicElements.ul)({
        ...{ class: "person-list" },
    });
    for (const [p] of __VLS_getVForSourceType((__VLS_ctx.person.persons))) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.li, __VLS_intrinsicElements.li)({
            ...{ onClick: (...[$event]) => {
                    if (!!(!__VLS_ctx.person.isLoading && __VLS_ctx.person.persons.length === 0))
                        return;
                    __VLS_ctx.selectPerson(p.id);
                } },
            key: (p.id),
            ...{ class: (['person-item', { active: __VLS_ctx.person.selectedPersonId === p.id }]) },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "person-meta" },
        });
        if (__VLS_ctx.editingId === p.id) {
            __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
                ...{ onClick: () => { } },
                ...{ class: "row gap-1" },
            });
            __VLS_asFunctionalElement(__VLS_intrinsicElements.input)({
                ...{ onKeyup: (...[$event]) => {
                        if (!!(!__VLS_ctx.person.isLoading && __VLS_ctx.person.persons.length === 0))
                            return;
                        if (!(__VLS_ctx.editingId === p.id))
                            return;
                        __VLS_ctx.commitRename(p.id);
                    } },
                ...{ onKeyup: (__VLS_ctx.cancelRename) },
                ...{ class: "input sm" },
                placeholder: "名字",
            });
            (__VLS_ctx.editingName);
            __VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
                ...{ onClick: (...[$event]) => {
                        if (!!(!__VLS_ctx.person.isLoading && __VLS_ctx.person.persons.length === 0))
                            return;
                        if (!(__VLS_ctx.editingId === p.id))
                            return;
                        __VLS_ctx.commitRename(p.id);
                    } },
                ...{ class: "btn sm primary" },
            });
            __VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
                ...{ onClick: (__VLS_ctx.cancelRename) },
                ...{ class: "btn sm ghost" },
            });
        }
        else {
            __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
                ...{ class: "row between flex-1" },
            });
            __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({});
            __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
                ...{ class: "person-name" },
            });
            (p.name ?? `Person #${p.id}`);
            __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
                ...{ class: "muted text-sm" },
            });
            (p.id);
            (p.face_count);
            __VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
                ...{ onClick: (...[$event]) => {
                        if (!!(!__VLS_ctx.person.isLoading && __VLS_ctx.person.persons.length === 0))
                            return;
                        if (!!(__VLS_ctx.editingId === p.id))
                            return;
                        __VLS_ctx.startRename(p.id, p.name);
                    } },
                ...{ class: "btn sm ghost" },
            });
        }
    }
}
__VLS_asFunctionalElement(__VLS_intrinsicElements.section, __VLS_intrinsicElements.section)({
    ...{ class: "card detail" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "card-header" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "card-title" },
});
if (__VLS_ctx.person.selectedPersonId != null) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "card-subtitle" },
    });
    (__VLS_ctx.person.selectedFaces.length);
    (__VLS_ctx.person.selectedPersonId);
}
else {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "card-subtitle" },
    });
}
if (__VLS_ctx.person.selectedPersonId != null) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "row gap-2" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.button, __VLS_intrinsicElements.button)({
        ...{ onClick: (...[$event]) => {
                if (!(__VLS_ctx.person.selectedPersonId != null))
                    return;
                __VLS_ctx.findPhotosOfPerson(__VLS_ctx.person.selectedPersonId);
            } },
        ...{ class: "btn sm primary" },
    });
}
if (__VLS_ctx.person.isLoadingFaces) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "muted" },
    });
}
else if (__VLS_ctx.person.selectedFaces.length === 0) {
    /** @type {[typeof EmptyState, ]} */ ;
    // @ts-ignore
    const __VLS_3 = __VLS_asFunctionalComponent(EmptyState, new EmptyState({
        title: "无内容",
        hint: "选择左侧一个人物查看其人脸",
    }));
    const __VLS_4 = __VLS_3({
        title: "无内容",
        hint: "选择左侧一个人物查看其人脸",
    }, ...__VLS_functionalComponentArgsRest(__VLS_3));
}
else {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "face-grid" },
    });
    for (const [f] of __VLS_getVForSourceType((__VLS_ctx.person.selectedFaces))) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            key: (f.id),
            ...{ class: "face-cell" },
        });
        if (__VLS_ctx.faceUrls.get(f.id)) {
            __VLS_asFunctionalElement(__VLS_intrinsicElements.img)({
                src: (__VLS_ctx.faceUrls.get(f.id)),
                alt: "",
            });
        }
        else {
            __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
                ...{ onMouseenter: (...[$event]) => {
                        if (!!(__VLS_ctx.person.isLoadingFaces))
                            return;
                        if (!!(__VLS_ctx.person.selectedFaces.length === 0))
                            return;
                        if (!!(__VLS_ctx.faceUrls.get(f.id)))
                            return;
                        __VLS_ctx.loadFaceThumb(f.id, f.image_id);
                    } },
                ...{ class: "face-placeholder muted center" },
            });
        }
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "image-cell-overlay" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({});
        (f.detector_score.toFixed(2));
        __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
            ...{ style: {} },
        });
        (f.quality.toFixed(2));
    }
}
/** @type {__VLS_StyleScopedClasses['view']} */ ;
/** @type {__VLS_StyleScopedClasses['view-header']} */ ;
/** @type {__VLS_StyleScopedClasses['muted']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['layout']} */ ;
/** @type {__VLS_StyleScopedClasses['card']} */ ;
/** @type {__VLS_StyleScopedClasses['list']} */ ;
/** @type {__VLS_StyleScopedClasses['card-header']} */ ;
/** @type {__VLS_StyleScopedClasses['card-title']} */ ;
/** @type {__VLS_StyleScopedClasses['card-subtitle']} */ ;
/** @type {__VLS_StyleScopedClasses['row']} */ ;
/** @type {__VLS_StyleScopedClasses['gap-2']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['sm']} */ ;
/** @type {__VLS_StyleScopedClasses['ghost']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['sm']} */ ;
/** @type {__VLS_StyleScopedClasses['primary']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['sm']} */ ;
/** @type {__VLS_StyleScopedClasses['person-list']} */ ;
/** @type {__VLS_StyleScopedClasses['person-meta']} */ ;
/** @type {__VLS_StyleScopedClasses['row']} */ ;
/** @type {__VLS_StyleScopedClasses['gap-1']} */ ;
/** @type {__VLS_StyleScopedClasses['input']} */ ;
/** @type {__VLS_StyleScopedClasses['sm']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['sm']} */ ;
/** @type {__VLS_StyleScopedClasses['primary']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['sm']} */ ;
/** @type {__VLS_StyleScopedClasses['ghost']} */ ;
/** @type {__VLS_StyleScopedClasses['row']} */ ;
/** @type {__VLS_StyleScopedClasses['between']} */ ;
/** @type {__VLS_StyleScopedClasses['flex-1']} */ ;
/** @type {__VLS_StyleScopedClasses['person-name']} */ ;
/** @type {__VLS_StyleScopedClasses['muted']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['sm']} */ ;
/** @type {__VLS_StyleScopedClasses['ghost']} */ ;
/** @type {__VLS_StyleScopedClasses['card']} */ ;
/** @type {__VLS_StyleScopedClasses['detail']} */ ;
/** @type {__VLS_StyleScopedClasses['card-header']} */ ;
/** @type {__VLS_StyleScopedClasses['card-title']} */ ;
/** @type {__VLS_StyleScopedClasses['card-subtitle']} */ ;
/** @type {__VLS_StyleScopedClasses['card-subtitle']} */ ;
/** @type {__VLS_StyleScopedClasses['row']} */ ;
/** @type {__VLS_StyleScopedClasses['gap-2']} */ ;
/** @type {__VLS_StyleScopedClasses['btn']} */ ;
/** @type {__VLS_StyleScopedClasses['sm']} */ ;
/** @type {__VLS_StyleScopedClasses['primary']} */ ;
/** @type {__VLS_StyleScopedClasses['muted']} */ ;
/** @type {__VLS_StyleScopedClasses['face-grid']} */ ;
/** @type {__VLS_StyleScopedClasses['face-cell']} */ ;
/** @type {__VLS_StyleScopedClasses['face-placeholder']} */ ;
/** @type {__VLS_StyleScopedClasses['muted']} */ ;
/** @type {__VLS_StyleScopedClasses['center']} */ ;
/** @type {__VLS_StyleScopedClasses['image-cell-overlay']} */ ;
var __VLS_dollars;
const __VLS_self = (await import('vue')).defineComponent({
    setup() {
        return {
            EmptyState: EmptyState,
            person: person,
            editingId: editingId,
            editingName: editingName,
            reload: reload,
            startRename: startRename,
            commitRename: commitRename,
            cancelRename: cancelRename,
            triggerCluster: triggerCluster,
            triggerRebuild: triggerRebuild,
            findPhotosOfPerson: findPhotosOfPerson,
            selectPerson: selectPerson,
            faceUrls: faceUrls,
            loadFaceThumb: loadFaceThumb,
        };
    },
});
export default (await import('vue')).defineComponent({
    setup() {
        return {};
    },
});
; /* PartiallyEnd: #4569/main.vue */
