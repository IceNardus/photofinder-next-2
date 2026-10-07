import { onMounted, onUnmounted, watch } from 'vue';
import { RouterLink, RouterView, useRoute, useRouter } from 'vue-router';
import { useAppStore } from '@/stores/app';
import { useSettingsStore } from '@/stores/settings';
import { useLicenseStore } from '@/stores/license';
import { navItems } from '@/router';
import ToastStack from '@/components/ToastStack.vue';
const app = useAppStore();
const settings = useSettingsStore();
const license = useLicenseStore();
const route = useRoute();
const router = useRouter();
// Public routes that don't require login
const publicRoutes = ['/login', '/register'];
// 路由守卫：每次导航前检查会员状态
router.beforeEach(async (to) => {
    // 先确保会员状态已刷新
    await license.refresh();
    const isLoggedIn = license.isLoggedIn;
    const isActive = license.isActive;
    // 未登录：只能访问 public routes
    if (!isLoggedIn) {
        if (publicRoutes.includes(to.path)) {
            return true; // 允许访问登录/注册页
        }
        return '/login'; // 重定向到登录页
    }
    // 已登录：禁止访问登录/注册页
    if (publicRoutes.includes(to.path)) {
        return '/'; // 重定向到首页
    }
    // 已登录但会员未激活：只能访问会员页
    if (!isActive && to.path !== '/membership') {
        return '/membership';
    }
    return true; // 允许访问
});
let statsTimer = null;
function startStatsPolling(intervalMs) {
    stopStatsPolling();
    statsTimer = setInterval(() => {
        void app.refreshStats();
    }, Math.max(500, intervalMs));
}
function stopStatsPolling() {
    if (statsTimer) {
        clearInterval(statsTimer);
        statsTimer = null;
    }
}
onMounted(async () => {
    await license.refresh();
    void app.refresh();
    startStatsPolling(settings.settings.sidebarPollMs);
});
watch(() => settings.settings.sidebarPollMs, (ms) => startStatsPolling(ms));
onUnmounted(() => {
    stopStatsPolling();
});
debugger; /* PartiallyEnd: #3632/scriptSetup.vue */
const __VLS_ctx = {};
let __VLS_components;
let __VLS_directives;
/** @type {__VLS_StyleScopedClasses['layout']} */ ;
/** @type {__VLS_StyleScopedClasses['nav-item']} */ ;
/** @type {__VLS_StyleScopedClasses['sidebar']} */ ;
/** @type {__VLS_StyleScopedClasses['brand']} */ ;
/** @type {__VLS_StyleScopedClasses['nav']} */ ;
/** @type {__VLS_StyleScopedClasses['nav-item']} */ ;
/** @type {__VLS_StyleScopedClasses['nav-item']} */ ;
/** @type {__VLS_StyleScopedClasses['nav-item']} */ ;
/** @type {__VLS_StyleScopedClasses['active']} */ ;
/** @type {__VLS_StyleScopedClasses['nav-icon']} */ ;
/** @type {__VLS_StyleScopedClasses['nav-icon']} */ ;
/** @type {__VLS_StyleScopedClasses['nav-icon']} */ ;
/** @type {__VLS_StyleScopedClasses['nav-icon']} */ ;
/** @type {__VLS_StyleScopedClasses['nav-icon']} */ ;
/** @type {__VLS_StyleScopedClasses['nav-icon']} */ ;
/** @type {__VLS_StyleScopedClasses['sidebar-stats']} */ ;
/** @type {__VLS_StyleScopedClasses['sidebar-stat-value']} */ ;
/** @type {__VLS_StyleScopedClasses['sidebar-foot']} */ ;
/** @type {__VLS_StyleScopedClasses['main']} */ ;
// CSS variable injection 
// CSS variable injection end 
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "layout" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.aside, __VLS_intrinsicElements.aside)({
    ...{ class: "sidebar" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "brand" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "brand-logo" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "brand-text" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "brand-name" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "brand-sub muted text-sm" },
});
__VLS_asFunctionalElement(__VLS_intrinsicElements.nav, __VLS_intrinsicElements.nav)({
    ...{ class: "nav" },
});
for (const [item] of __VLS_getVForSourceType((__VLS_ctx.navItems))) {
    const __VLS_0 = {}.RouterLink;
    /** @type {[typeof __VLS_components.RouterLink, typeof __VLS_components.RouterLink, ]} */ ;
    // @ts-ignore
    const __VLS_1 = __VLS_asFunctionalComponent(__VLS_0, new __VLS_0({
        key: (item.name),
        to: (item.path),
        ...{ class: (['nav-item', { active: __VLS_ctx.route.path === item.path }]) },
    }));
    const __VLS_2 = __VLS_1({
        key: (item.name),
        to: (item.path),
        ...{ class: (['nav-item', { active: __VLS_ctx.route.path === item.path }]) },
    }, ...__VLS_functionalComponentArgsRest(__VLS_1));
    __VLS_3.slots.default;
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span)({
        ...{ class: "nav-icon" },
        'data-icon': (item.icon),
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
        ...{ class: "nav-label" },
    });
    (item.title);
    var __VLS_3;
}
if (__VLS_ctx.app.stats) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "sidebar-stats" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "sidebar-stats-title muted text-sm" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "sidebar-stat" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
        ...{ class: "sidebar-stat-label" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
        ...{ class: "sidebar-stat-value font-mono" },
    });
    (__VLS_ctx.app.stats.image_count);
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "sidebar-stat" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
        ...{ class: "sidebar-stat-label" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
        ...{ class: "sidebar-stat-value font-mono" },
    });
    (__VLS_ctx.app.stats.face_count);
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "sidebar-stat" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
        ...{ class: "sidebar-stat-label" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
        ...{ class: "sidebar-stat-value font-mono" },
    });
    (__VLS_ctx.app.stats.visual_object_v2_count);
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "sidebar-stat" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
        ...{ class: "sidebar-stat-label" },
    });
    __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
        ...{ class: "sidebar-stat-value font-mono" },
    });
    (__VLS_ctx.app.stats.person_count);
    if (__VLS_ctx.app.stats.pending_task_count > 0) {
        __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
            ...{ class: "sidebar-stat sidebar-stat-busy" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
            ...{ class: "sidebar-stat-label" },
        });
        __VLS_asFunctionalElement(__VLS_intrinsicElements.span, __VLS_intrinsicElements.span)({
            ...{ class: "sidebar-stat-value font-mono" },
        });
        (__VLS_ctx.app.stats.pending_task_count);
    }
}
__VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
    ...{ class: "sidebar-foot" },
});
if (__VLS_ctx.app.info) {
    __VLS_asFunctionalElement(__VLS_intrinsicElements.div, __VLS_intrinsicElements.div)({
        ...{ class: "text-sm muted font-mono" },
    });
    (__VLS_ctx.app.info.version);
    (__VLS_ctx.app.info.platform);
}
__VLS_asFunctionalElement(__VLS_intrinsicElements.main, __VLS_intrinsicElements.main)({
    ...{ class: "main" },
});
const __VLS_4 = {}.RouterView;
/** @type {[typeof __VLS_components.RouterView, ]} */ ;
// @ts-ignore
const __VLS_5 = __VLS_asFunctionalComponent(__VLS_4, new __VLS_4({}));
const __VLS_6 = __VLS_5({}, ...__VLS_functionalComponentArgsRest(__VLS_5));
/** @type {[typeof ToastStack, ]} */ ;
// @ts-ignore
const __VLS_8 = __VLS_asFunctionalComponent(ToastStack, new ToastStack({}));
const __VLS_9 = __VLS_8({}, ...__VLS_functionalComponentArgsRest(__VLS_8));
/** @type {__VLS_StyleScopedClasses['layout']} */ ;
/** @type {__VLS_StyleScopedClasses['sidebar']} */ ;
/** @type {__VLS_StyleScopedClasses['brand']} */ ;
/** @type {__VLS_StyleScopedClasses['brand-logo']} */ ;
/** @type {__VLS_StyleScopedClasses['brand-text']} */ ;
/** @type {__VLS_StyleScopedClasses['brand-name']} */ ;
/** @type {__VLS_StyleScopedClasses['brand-sub']} */ ;
/** @type {__VLS_StyleScopedClasses['muted']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['nav']} */ ;
/** @type {__VLS_StyleScopedClasses['nav-icon']} */ ;
/** @type {__VLS_StyleScopedClasses['nav-label']} */ ;
/** @type {__VLS_StyleScopedClasses['sidebar-stats']} */ ;
/** @type {__VLS_StyleScopedClasses['sidebar-stats-title']} */ ;
/** @type {__VLS_StyleScopedClasses['muted']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['sidebar-stat']} */ ;
/** @type {__VLS_StyleScopedClasses['sidebar-stat-label']} */ ;
/** @type {__VLS_StyleScopedClasses['sidebar-stat-value']} */ ;
/** @type {__VLS_StyleScopedClasses['font-mono']} */ ;
/** @type {__VLS_StyleScopedClasses['sidebar-stat']} */ ;
/** @type {__VLS_StyleScopedClasses['sidebar-stat-label']} */ ;
/** @type {__VLS_StyleScopedClasses['sidebar-stat-value']} */ ;
/** @type {__VLS_StyleScopedClasses['font-mono']} */ ;
/** @type {__VLS_StyleScopedClasses['sidebar-stat']} */ ;
/** @type {__VLS_StyleScopedClasses['sidebar-stat-label']} */ ;
/** @type {__VLS_StyleScopedClasses['sidebar-stat-value']} */ ;
/** @type {__VLS_StyleScopedClasses['font-mono']} */ ;
/** @type {__VLS_StyleScopedClasses['sidebar-stat']} */ ;
/** @type {__VLS_StyleScopedClasses['sidebar-stat-label']} */ ;
/** @type {__VLS_StyleScopedClasses['sidebar-stat-value']} */ ;
/** @type {__VLS_StyleScopedClasses['font-mono']} */ ;
/** @type {__VLS_StyleScopedClasses['sidebar-stat']} */ ;
/** @type {__VLS_StyleScopedClasses['sidebar-stat-busy']} */ ;
/** @type {__VLS_StyleScopedClasses['sidebar-stat-label']} */ ;
/** @type {__VLS_StyleScopedClasses['sidebar-stat-value']} */ ;
/** @type {__VLS_StyleScopedClasses['font-mono']} */ ;
/** @type {__VLS_StyleScopedClasses['sidebar-foot']} */ ;
/** @type {__VLS_StyleScopedClasses['text-sm']} */ ;
/** @type {__VLS_StyleScopedClasses['muted']} */ ;
/** @type {__VLS_StyleScopedClasses['font-mono']} */ ;
/** @type {__VLS_StyleScopedClasses['main']} */ ;
var __VLS_dollars;
const __VLS_self = (await import('vue')).defineComponent({
    setup() {
        return {
            RouterLink: RouterLink,
            RouterView: RouterView,
            navItems: navItems,
            ToastStack: ToastStack,
            app: app,
            route: route,
        };
    },
});
export default (await import('vue')).defineComponent({
    setup() {
        return {};
    },
});
; /* PartiallyEnd: #4569/main.vue */
