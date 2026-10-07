// Person store — 人物列表 + 重命名 + 详情。
import { defineStore } from 'pinia';
import { ref } from 'vue';
import { clusterFaces, facesOfPerson, listPersons, rebuildPersonClusters, renamePerson, } from '@/api';
export const usePersonStore = defineStore('person', () => {
    const persons = ref([]);
    const isLoading = ref(false);
    const lastError = ref(null);
    // 当前打开的人物详情
    const selectedPersonId = ref(null);
    const selectedFaces = ref([]);
    const isLoadingFaces = ref(false);
    async function refresh() {
        isLoading.value = true;
        lastError.value = null;
        try {
            persons.value = await listPersons();
        }
        catch (e) {
            lastError.value = String(e);
        }
        finally {
            isLoading.value = false;
        }
    }
    async function rename(id, name) {
        await renamePerson(id, name);
        const idx = persons.value.findIndex((p) => p.id === id);
        if (idx >= 0) {
            const p = persons.value[idx];
            if (p)
                persons.value[idx] = { ...p, name };
        }
    }
    async function selectPerson(id) {
        selectedPersonId.value = id;
        isLoadingFaces.value = true;
        try {
            selectedFaces.value = await facesOfPerson(id);
        }
        catch (e) {
            lastError.value = String(e);
        }
        finally {
            isLoadingFaces.value = false;
        }
    }
    async function cluster() {
        return await clusterFaces();
    }
    async function rebuildClusters() {
        return await rebuildPersonClusters();
    }
    return {
        persons,
        isLoading,
        lastError,
        selectedPersonId,
        selectedFaces,
        isLoadingFaces,
        refresh,
        rename,
        selectPerson,
        cluster,
        rebuildClusters,
    };
});
