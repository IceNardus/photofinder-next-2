import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'pages/root_view.dart';
import 'license_store.dart';

void main() {
  runApp(const PhotoFinderApp());
}

class PhotoFinderApp extends StatelessWidget {
  const PhotoFinderApp({super.key});

  @override
  Widget build(BuildContext context) {
    return ChangeNotifierProvider(
      create: (_) => LicenseStore(),
      child: MaterialApp(
        title: 'PhotoFinder',
        theme: ThemeData(
          colorScheme: ColorScheme.fromSeed(seedColor: const Color(0xFFE8D5C4)),
          useMaterial3: true,
        ),
        home: const RootView(),
      ),
    );
  }
}
