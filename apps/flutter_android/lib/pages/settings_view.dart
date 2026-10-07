import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../license_store.dart';
import 'membership_view.dart';
import 'login_view.dart';
import 'register_view.dart';

// Wrapper for register view with navigation
class RegisterViewWrapper extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return RegisterView(
      onBackToLogin: () {
        Navigator.of(context).pop();
      },
    );
  }
}

class SettingsView extends StatelessWidget {
  const SettingsView({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Consumer<LicenseStore>(
          builder: (context, store, _) {
            debugPrint('[SettingsView] build: isLoggedIn=${store.isLoggedIn}, account=${store.account}');
            if (!store.isLoggedIn) {
              return _buildNotLoggedInView(context);
            }
            return _buildLoggedInView(context, store);
          },
        ),
      ),
    );
  }

  Widget _buildNotLoggedInView(BuildContext context) {
    return Padding(
      padding: const EdgeInsets.all(20),
      child: Column(
        children: [
          const SizedBox(height: 40),
          Center(
            child: Column(
              children: [
                Container(
                  width: 80,
                  height: 80,
                  decoration: BoxDecoration(
                    color: const Color(0xFFF5F0E8),
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(
                    Icons.person_outline,
                    size: 40,
                    color: Color(0xFFC67B5C),
                  ),
                ),
                const SizedBox(height: 16),
                const Text(
                  '未登录',
                  style: TextStyle(fontSize: 16, color: Colors.grey),
                ),
                const SizedBox(height: 24),
                ElevatedButton(
                  onPressed: () {
                    Navigator.of(context).push(
                      MaterialPageRoute(builder: (_) => LoginView(
                        onRegisterTap: () {
                          Navigator.of(context).pop();
                          // Show register page
                          Navigator.of(context).push(
                            MaterialPageRoute(builder: (_) => RegisterViewWrapper()),
                          );
                        },
                      )),
                    );
                  },
                  style: ElevatedButton.styleFrom(
                    backgroundColor: const Color(0xFFC67B5C),
                    foregroundColor: Colors.white,
                    padding: const EdgeInsets.symmetric(horizontal: 32, vertical: 12),
                  ),
                  child: const Text('登录'),
                ),
                const SizedBox(height: 12),
                TextButton(
                  onPressed: () {
                    Navigator.of(context).push(
                      MaterialPageRoute(builder: (_) => RegisterViewWrapper()),
                    );
                  },
                  child: const Text('没有账号？立即注册'),
                ),
              ],
            ),
          ),
          const Spacer(),
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(
              color: const Color(0xFFF5F0E8),
              borderRadius: BorderRadius.circular(16),
            ),
            child: const Text(
              'Android 端目前为只读预览壳。完整功能请使用 macOS / Windows 桌面版本。',
              style: TextStyle(fontSize: 14, color: Colors.grey),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLoggedInView(BuildContext context, LicenseStore store) {
    debugPrint('[SettingsView] _buildLoggedInView: email=${store.account?.email}, licenseActive=${store.isLicenseActive}');
    return SingleChildScrollView(
      padding: const EdgeInsets.all(20),
      child: Column(
        children: [
          const SizedBox(height: 24),
          // User avatar and email
          Center(
            child: Column(
              children: [
                Container(
                  width: 80,
                  height: 80,
                  decoration: const BoxDecoration(
                    color: Color(0xFFF5F0E8),
                    shape: BoxShape.circle,
                  ),
                  child: const Icon(
                    Icons.person,
                    size: 40,
                    color: Color(0xFFC67B5C),
                  ),
                ),
                const SizedBox(height: 12),
                Text(
                  store.account?.email ?? 'NO EMAIL',
                  style: const TextStyle(fontSize: 16, fontWeight: FontWeight.w500),
                ),
              ],
            ),
          ),
          const SizedBox(height: 24),
          // Membership card
          GestureDetector(
            onTap: () {
              Navigator.of(context).push(
                MaterialPageRoute(builder: (_) => MembershipView()),
              );
            },
            child: Container(
              padding: const EdgeInsets.all(20),
              decoration: BoxDecoration(
                gradient: const LinearGradient(
                  colors: [Color(0xFFC67B5C), Color(0xFFE8A87C)],
                  begin: Alignment.topLeft,
                  end: Alignment.bottomRight,
                ),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Row(
                children: [
                  const Icon(Icons.workspace_premium, color: Colors.white, size: 32),
                  const SizedBox(width: 16),
                  Expanded(
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        const Text(
                          '会员状态',
                          style: TextStyle(color: Colors.white70, fontSize: 12),
                        ),
                        const SizedBox(height: 4),
                        Text(
                          store.isLicenseActive
                              ? '已激活 (剩余 ${store.remainingDays} 天)'
                              : '未激活',
                          style: const TextStyle(
                            color: Colors.white,
                            fontSize: 18,
                            fontWeight: FontWeight.bold,
                          ),
                        ),
                      ],
                    ),
                  ),
                  const Icon(Icons.chevron_right, color: Colors.white70),
                ],
              ),
            ),
          ),
          const SizedBox(height: 24),
          // Info section
          const Text(
            '账号信息',
            style: TextStyle(fontSize: 14, color: Colors.grey, fontWeight: FontWeight.w500),
          ),
          const SizedBox(height: 12),
          _buildInfoCard(
            icon: Icons.email_outlined,
            title: '邮箱',
            value: store.account?.email ?? '',
          ),
          _buildInfoCard(
            icon: Icons.calendar_today_outlined,
            title: '会员有效期',
            value: store.isLicenseActive
                ? '剩余 ${store.remainingDays} 天'
                : '未激活',
          ),
          _buildInfoCard(
            icon: Icons.photo_library_outlined,
            title: '照片访问',
            value: '完整访问',
          ),
          _buildInfoCard(
            icon: Icons.info_outline,
            title: '版本',
            value: '1.0.0',
          ),
          const SizedBox(height: 24),
          // Logout button
          GestureDetector(
            onTap: () async {
              final confirmed = await showDialog<bool>(
                context: context,
                builder: (context) => AlertDialog(
                  title: const Text('退出登录'),
                  content: const Text('确定要退出当前账号吗？'),
                  actions: [
                    TextButton(
                      onPressed: () => Navigator.pop(context, false),
                      child: const Text('取消'),
                    ),
                    TextButton(
                      onPressed: () => Navigator.pop(context, true),
                      child: const Text('确定'),
                    ),
                  ],
                ),
              );
              if (confirmed == true) {
                await store.logout();
              }
            },
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.red.shade50,
                borderRadius: BorderRadius.circular(12),
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  Icon(Icons.logout, color: Colors.red.shade400),
                  const SizedBox(width: 8),
                  Text(
                    '退出登录',
                    style: TextStyle(color: Colors.red.shade400, fontWeight: FontWeight.w500),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 32),
        ],
      ),
    );
  }

  static Widget _buildInfoCard({
    required IconData icon,
    required String title,
    required String value,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 14),
      margin: const EdgeInsets.only(bottom: 8),
      decoration: BoxDecoration(
        color: const Color(0xFFF5F0E8),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Row(
        children: [
          Icon(icon, color: const Color(0xFFC67B5C), size: 20),
          const SizedBox(width: 12),
          Text(title, style: const TextStyle(fontSize: 14)),
          const Spacer(),
          Text(value, style: const TextStyle(color: Colors.grey, fontSize: 14)),
        ],
      ),
    );
  }
}
