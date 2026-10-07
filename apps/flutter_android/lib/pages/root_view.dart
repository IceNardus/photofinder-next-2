import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'scan_view.dart';
import 'search_view.dart';
import 'settings_view.dart';
import '../license_store.dart';

class RootView extends StatefulWidget {
  const RootView({super.key});

  @override
  State<RootView> createState() => _RootViewState();
}

class _RootViewState extends State<RootView> {
  int _selectedIndex = 0;

  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<LicenseStore>().init();
    });
  }

  static const List<Widget> _pages = [
    ScanView(),
    SearchView(),
    SettingsView(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _pages[_selectedIndex],
      bottomNavigationBar: NavigationBar(
        selectedIndex: _selectedIndex,
        onDestinationSelected: (index) {
          setState(() {
            _selectedIndex = index;
          });
        },
        destinations: const [
          NavigationDestination(
            icon: Icon(Icons.scanner_outlined),
            selectedIcon: Icon(Icons.scanner),
            label: '扫描',
          ),
          NavigationDestination(
            icon: Icon(Icons.search),
            label: '搜索',
          ),
          NavigationDestination(
            icon: Icon(Icons.person_outlined),
            selectedIcon: Icon(Icons.person),
            label: '用户',
          ),
        ],
      ),
    );
  }
}
