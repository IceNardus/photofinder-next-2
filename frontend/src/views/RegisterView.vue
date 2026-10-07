<script setup lang="ts">
// RegisterView — 会员注册

import { ref } from 'vue';
import { useRouter } from 'vue-router';
import { useLicenseStore } from '@/stores/license';
import { useToastStore } from '@/stores/toast';

const router = useRouter();
const license = useLicenseStore();
const toast = useToastStore();

const email = ref('');
const password = ref('');
const confirmPassword = ref('');
const isLoading = ref(false);

async function handleRegister() {
  if (!email.value || !password.value || !confirmPassword.value) {
    toast.warning('请填写所有字段');
    return;
  }
  if (password.value !== confirmPassword.value) {
    toast.warning('两次密码输入不一致');
    return;
  }
  if (password.value.length < 6) {
    toast.warning('密码至少 6 个字符');
    return;
  }
  isLoading.value = true;
  try {
    await license.register(email.value, password.value);
    toast.success('注册成功');
    router.push('/');
  } catch (e) {
    toast.danger('注册失败: ' + String(e));
  } finally {
    isLoading.value = false;
  }
}
</script>

<template>
  <div class="view">
    <div class="register-card">
      <h1>注册会员</h1>
      <p class="muted text-sm">创建 PhotoFinder 账户</p>

      <form @submit.prevent="handleRegister">
        <div class="field">
          <label>邮箱</label>
          <input v-model="email" type="email" class="input" placeholder="your@email.com" :disabled="isLoading" />
        </div>
        <div class="field">
          <label>密码</label>
          <input v-model="password" type="password" class="input" placeholder="至少 6 个字符" :disabled="isLoading" />
        </div>
        <div class="field">
          <label>确认密码</label>
          <input v-model="confirmPassword" type="password" class="input" placeholder="再次输入密码" :disabled="isLoading" />
        </div>
        <button type="submit" class="btn primary full" :disabled="isLoading">
          {{ isLoading ? '注册中...' : '注册' }}
        </button>
      </form>

      <div class="footer">
        已有账户？<router-link to="/login">登录</router-link>
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
.register-card {
  background: var(--color-surface);
  border: 1px solid var(--color-border);
  border-radius: var(--radius-lg);
  padding: var(--space-8);
  width: 100%;
  max-width: 400px;
}
.register-card h1 {
  font-size: var(--text-xl);
  font-weight: 600;
  margin-bottom: var(--space-1);
}
.register-card form {
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
