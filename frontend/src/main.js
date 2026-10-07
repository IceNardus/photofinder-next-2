// PhotoFinder Next 2 — Frontend 入口
//
// 启动：createApp → pinia → router → mount。
// 不在此处做 IPC 调用（由各 view / store 按需触发）。
import { createApp } from 'vue';
import { createPinia } from 'pinia';
import App from './App.vue';
import { router } from './router';
import './design/tokens.css';
import './design/components.css';
const app = createApp(App);
app.use(createPinia());
app.use(router);
app.mount('#app');
