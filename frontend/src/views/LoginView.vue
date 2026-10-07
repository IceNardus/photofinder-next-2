<script setup lang="ts">
// LoginView — 会员登录

import { ref } from 'vue';
import { useRouter } from 'vue-router';
import { useLicenseStore } from '@/stores/license';
import { useToastStore } from '@/stores/toast';

const router = useRouter();
const license = useLicenseStore();
const toast = useToastStore();

const email = ref('');
const password = ref('');
const isLoading = ref(false);

async function handleLogin() {
  if (!email.value || !password.value) {
    toast.warning('请填写邮箱和密码');
    return;
  }
  isLoading.value = true;
  try {
    await license.login(email.value, password.value);
    toast.success('登录成功');
    router.push('/');
  } catch (e) {
    toast.danger('登录失败: ' + String(e));
  } finally {
    isLoading.value = false;
  }
}
</script>

<template>
  <div class="view">
    <div class="login-card">
      <h1>会员登录</h1>
      <p class="muted text-sm">登录您的 PhotoFinder 账户</p>

      <form @submit.prevent="handleLogin">
        <div class="field">
          <label>邮箱</label>
          <input v-model="email" type="email" class="input" placeholder="your@email.com" :disabled="isLoading" />
        </div>
        <div class="field">
          <label>密码</label>
          <input v-model="password" type="password" class="input" placeholder="••••••••" :disabled="isLoading" />
        </div>
        <button type="submit" class="btn primary full" :disabled="isLoading">
          {{ isLoading ? '登录中...' : '登录' }}
        </button>
      </form>

      <div class="footer">
        还没有账户？<router-link to="/register">注册会员</router-link>
      </div>
    </div>
  </div>
</template>

<style scoped>
.view {
  display: flex;
  align-items: center;
  justify-content: center;
  min-height: 100vh;
  padding: var(--space-5);
}
.login-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: var(--space-8);
  width: 100%;
  max-width: 400px;
}
.login-card h1 {
  font-size: var(--text-xl);
  font-weight: 600;
  margin-bottom: var(--space-1);
}
.login-card form {
  margin-top: var(--space-6);
  display: flex;
  flex-direction: column;
  gap: var(--space-4);
}
.field {
  display: flex;
  flex-direction: column;
  gap: var(--space-2);
}
.field label {
  font-size: var(--text-sm);
  font-weight: 500;
}
.footer {
  margin-top: var(--space-6);
  text-align: center;
  font-size: var(--text-sm);
  color: var(--color-fg-2);
}
.footer a {
  color: var(--color-primary);
  text-decoration: none;
}
.footer a:hover {
  text-decoration: underline;
}
</style>
