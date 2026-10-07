// Person store — 人物列表 + 重命名 + 详情。

import { defineStore } from 'pinia';
import { ref } from 'vue';
import {
  clusterFaces,
  facesOfPerson,
  listPersons,
  rebuildPersonClusters,
  renamePerson,
  type Face,
  type Person,
  type ClusterSummary,
} from '@/api';

export const usePersonStore = defineStore('person', () => {
  const persons = ref<Person[]>([]);
  const isLoading = ref(false);
  const lastError = ref<string | null>(null);

  // 当前打开的人物详情
  const selectedPersonId = ref<number | null>(null);
  const selectedFaces = ref<Face[]>([]);
  const isLoadingFaces = ref(false);

  async function refresh() {
    isLoading.value = true;
    lastError.value = null;
    try {
      persons.value = await listPersons();
    } catch (e) {
      lastError.value = String(e);
    } finally {
      isLoading.value = false;
    }
  }

  async function rename(id: number, name: string | null) {
    await renamePerson(id, name);
    const idx = persons.value.findIndex((p) => p.id === id);
    if (idx >= 0) {
      const p = persons.value[idx];
      if (p) persons.value[idx] = { ...p, name };
    }
  }

  async function selectPerson(id: number) {
    selectedPersonId.value = id;
    isLoadingFaces.value = true;
    try {
      selectedFaces.value = await facesOfPerson(id);
    } catch (e) {
      lastError.value = String(e);
    } finally {
      isLoadingFaces.value = false;
    }
  }

  async function cluster(): Promise<number> {
    return await clusterFaces();
  }

  async function rebuildClusters(): Promise<ClusterSummary> {
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
