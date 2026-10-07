// Vue Router 配置 — 5 个 view。
//
// - /            ScanView   （默认）
// - /persons     PersonsView
// - /faces       PersonSearchView （人脸检索）
// - /objects     ObjectSearchView  （对象类别检索）
// - /settings    SettingsView
// - /membership  MembershipView （会员中心）
// - /login       LoginView （登录）
// - /register    RegisterView （注册）
import { createRouter, createWebHashHistory } from 'vue-router';
const routes = [
    {
        path: '/',
        name: 'scan',
        component: () => import('@/views/ScanView.vue'),
        meta: { title: '扫描', icon: 'scan' },
    },
    {
        path: '/persons',
        name: 'persons',
        component: () => import('@/views/PersonsView.vue'),
        meta: { title: '人物', icon: 'persons' },
    },
    {
        path: '/faces',
        name: 'faces',
        component: () => import('@/views/PersonSearchView.vue'),
        meta: { title: '人脸检索', icon: 'face' },
    },
    {
        path: '/objects',
        name: 'objects',
        component: () => import('@/views/ObjectSearchView.vue'),
        meta: { title: '对象检索', icon: 'object' },
    },
    {
        path: '/settings',
        name: 'settings',
        component: () => import('@/views/SettingsView.vue'),
        meta: { title: '设置', icon: 'settings' },
    },
    {
        path: '/membership',
        name: 'membership',
        component: () => import('@/views/MembershipView.vue'),
        meta: { title: '会员', icon: 'membership' },
    },
    {
        path: '/login',
        name: 'login',
        component: () => import('@/views/LoginView.vue'),
        meta: { title: '登录', icon: 'login' },
    },
    {
        path: '/register',
        name: 'register',
        component: () => import('@/views/RegisterView.vue'),
        meta: { title: '注册', icon: 'register' },
    },
];
export const router = createRouter({
    history: createWebHashHistory(),
    routes,
});
export const navItems = routes
    .filter((r) => r.meta?.title && !['persons', 'settings', 'login', 'register'].includes(String(r.name)))
    .map((r) => ({
    name: String(r.name),
    path: r.path,
    title: String(r.meta?.title ?? ''),
    icon: String(r.meta?.icon ?? ''),
}));
