<script setup lang="ts">
// MembershipView — 会员中心

import { onMounted, ref } from 'vue';
import { useRouter } from 'vue-router';
import { useLicenseStore } from '@/stores/license';
import { useToastStore } from '@/stores/toast';

const router = useRouter();
const license = useLicenseStore();
const toast = useToastStore();

const redeemCode = ref('');
const isRedeeming = ref(false);

onMounted(async () => {
  await license.refresh();
});

async function handleRedeem() {
  if (!redeemCode.value.trim()) {
    toast.warning('请输入兑换码');
    return;
  }
  isRedeeming.value = true;
  try {
    await license.redeem(redeemCode.value.trim());
    toast.success('兑换成功！会员已激活');
    redeemCode.value = '';
  } catch (e) {
    toast.danger('兑换失败: ' + String(e));
  } finally {
    isRedeeming.value = false;
  }
}

async function handleLogout() {
  try {
    await license.logout();
    toast.success('已退出登录');
    router.push('/login');
  } catch (e) {
    toast.danger('退出失败: ' + String(e));
  }
}

function formatDate(timestamp: number): string {
  const d = new Date(timestamp * 1000);
  return d.toLocaleDateString('zh-CN', { year: 'numeric', month: '2-digit', day: '2-digit' });
}
</script>

<template>
  <div class="view">
    <header class="view-header">
      <h1>会员中心</h1>
      <p class="muted text-sm">管理您的 PhotoFinder 会员</p>
    </header>

    <!-- Account info -->
    <section class="card">
      <div class="card-header">
        <div class="card-title">账户信息</div>
      </div>
      <div v-if="license.account" class="kv">
        <div class="kv-row"><span class="muted">账号</span><span>{{ license.account.email }}</span></div>
        <div class="kv-row"><span class="muted">注册时间</span><span>{{ formatDate(license.account.created_at) }}</span></div>
      </div>
      <div v-else class="empty-hint">未登录</div>
    </section>

    <!-- License status -->
    <section class="card">
      <div class="card-header">
        <div class="card-title">会员状态</div>
      </div>
      <div v-if="license.license">
        <div class="status-badge" :class="license.license.status">
          <span class="dot"></span>
          <span v-if="license.license.status === 'active'">会员有效</span>
          <span v-else-if="license.license.status === 'expired'">会员已过期</span>
          <span v-else-if="license.license.status === 'not_activated'">未激活</span>
          <span v-else>验证失败</span>
        </div>
        <div v-if="license.license.status === 'active'" class="kv">
          <div class="kv-row"><span class="muted">剩余</span><span class="highlight">{{ license.remainingDays }} 天</span></div>
          <div class="kv-row"><span class="muted">到期时间</span><span>{{ license.license.expires_at ? formatDate(license.license.expires_at) : 'N/A' }}</span></div>
        </div>
        <div v-else-if="license.license.status === 'expired'" class="kv">
          <div class="kv-row"><span class="muted">到期时间</span><span>{{ license.license.expires_at ? formatDate(license.license.expires_at) : 'N/A' }}</span></div>
        </div>
      </div>
      <div v-else class="empty-hint">暂无会员信息</div>
    </section>

    <!-- Redeem code -->
    <section class="card">
      <div class="card-header">
        <div class="card-title">兑换会员</div>
      </div>
      <div class="redeem-form">
        <input
          v-model="redeemCode"
          type="text"
          class="input"
          placeholder="输入兑换码"
          :disabled="isRedeeming"
        />
        <button class="btn primary" :disabled="isRedeeming" @click="handleRedeem">
          {{ isRedeeming ? '兑换中...' : '兑换' }}
        </button>
      </div>
      <p class="hint text-sm muted">输入任意非空兑换码即可激活 30 天会员</p>
    </section>

    <!-- Actions -->
    <section class="card">
      <div class="card-header">
        <div class="card-title">账户操作</div>
      </div>
      <div class="actions">
        <button class="btn danger" @click="handleLogout">退出登录</button>
      </div>
    </section>
  </div>
</template>

<style scoped>
.view {
  display: flex;
  flex-direction: column;
  gap: var(--space-5);
  max-width: 600px;
  margin: 0 auto;
  padding: var(--space-5);
}
.view-header h1 {
  font-size: var(--text-2xl);
  font-weight: 600;
  margin-bottom: var(--space-1);
}
.kv { display: flex; flex-direction: column; gap: var(--space-2); margin-top: var(--space-4); }
.kv-row {
  display: grid;
  grid-template-columns: 80px 1fr;
  gap: var(--space-3);
  font-size: var(--text-sm);
}
.kv-row .muted { font-size: var(--text-sm); }
.status-badge {
  display: inline-flex;
  align-items: center;
  gap: var(--space-2);
  padding: var(--space-2) var(--space-3);
  border-radius: var(--radius-md);
  font-size: var(--text-sm);
  font-weight: 500;
}
.status-badge .dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}
.status-badge.active { background: rgba(34, 197, 94, 0.1); color: #22c55e; }
.status-badge.active .dot { background: #22c55e; }
.status-badge.expired { background: rgba(255, 69, 58, 0.1); color: #ff3b30; }
.status-badge.expired .dot { background: #ff3b30; }
.status-badge.not_activated { background: rgba(234, 179, 8, 0.1); color: #eab308; }
.status-badge.not_activated .dot { background: #eab308; }
.status-badge.verification_failed { background: rgba(255, 69, 58, 0.1); color: #ff3b30; }
.status-badge.verification_failed .dot { background: #ff3b30; }
.highlight { color: var(--color-primary); font-weight: 600; font-size: var(--text-lg); }
.redeem-form {
  display: flex;
  gap: var(--space-3);
  margin-top: var(--space-4);
}
.redeem-form .input { flex: 1; }
.hint { margin-top: var(--space-2); }
.actions { margin-top: var(--space-4); }
.empty-hint { color: var(--color-fg-2); font-size: var(--text-sm); margin-top: var(--space-2); }
</style>
