// License store — 会员账户管理
//
// 管理登录/注册/兑换码状态
import { defineStore } from 'pinia';
import { ref, computed } from 'vue';
import { checkLicense, getAccount, getLicenseStatus, loginAccount, logoutAccount as apiLogout, redeemCode, registerAccount, } from '@/api';
export const useLicenseStore = defineStore('license', () => {
    // State
    const account = ref(null);
    const license = ref(null);
    const isLoading = ref(false);
    const lastError = ref(null);
    // Computed
    const isLoggedIn = computed(() => account.value !== null);
    const isActive = computed(() => license.value?.status === 'active');
    const remainingDays = computed(() => license.value?.remaining_days ?? 0);
    // Actions
    async function refresh() {
        isLoading.value = true;
        lastError.value = null;
        try {
            const [acc, lic] = await Promise.all([getAccount(), getLicenseStatus()]);
            account.value = acc;
            license.value = lic;
        }
        catch (e) {
            lastError.value = String(e);
            // On error, clear account to ensure we're treated as logged out
            account.value = null;
            license.value = null;
        }
        finally {
            isLoading.value = false;
        }
    }
    async function register(email, password) {
        isLoading.value = true;
        lastError.value = null;
        try {
            account.value = await registerAccount(email, password);
            // After register, check license status
            license.value = await getLicenseStatus();
        }
        catch (e) {
            lastError.value = String(e);
            throw e;
        }
        finally {
            isLoading.value = false;
        }
    }
    async function login(email, password) {
        isLoading.value = true;
        lastError.value = null;
        try {
            account.value = await loginAccount(email, password);
            license.value = await getLicenseStatus();
        }
        catch (e) {
            lastError.value = String(e);
            throw e;
        }
        finally {
            isLoading.value = false;
        }
    }
    async function logout() {
        isLoading.value = true;
        lastError.value = null;
        try {
            await apiLogout();
            account.value = null;
            license.value = null;
        }
        catch (e) {
            lastError.value = String(e);
            throw e;
        }
        finally {
            isLoading.value = false;
        }
    }
    async function redeem(code) {
        isLoading.value = true;
        lastError.value = null;
        try {
            license.value = await redeemCode(code);
        }
        catch (e) {
            lastError.value = String(e);
            throw e;
        }
        finally {
            isLoading.value = false;
        }
    }
    async function check() {
        try {
            return await checkLicense();
        }
        catch (e) {
            lastError.value = String(e);
            return false;
        }
    }
    return {
        // State
        account,
        license,
        isLoading,
        lastError,
        // Computed
        isLoggedIn,
        isActive,
        remainingDays,
        // Actions
        refresh,
        register,
        login,
        logout,
        redeem,
        check,
    };
});
