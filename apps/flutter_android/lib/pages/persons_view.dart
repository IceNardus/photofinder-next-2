import 'dart:convert';
import 'package:flutter/material.dart';
import '../ffi/rust_ffi.dart';

class Person {
  final int id;
  final String name;
  final int faceCount;

  Person({required this.id, required this.name, required this.faceCount});

  factory Person.fromJson(Map<String, dynamic> json) {
    return Person(
      id: json['id'] as int,
      name: json['name'] as String? ?? 'Unknown',
      faceCount: json['face_count'] as int? ?? 0,
    );
  }
}

class ClusterResult {
  final int assigned;
  final int created;
  final int failed;

  ClusterResult({required this.assigned, required this.created, required this.failed});

  factory ClusterResult.fromJson(Map<String, dynamic> json) {
    return ClusterResult(
      assigned: json['assigned'] as int? ?? 0,
      created: json['created'] as int? ?? 0,
      failed: json['failed'] as int? ?? 0,
    );
  }
}

/// Dialog that shows loading indicator and triggers clustering
class _ClusteringDialog extends StatefulWidget {
  final Future<void> Function() onCluster;

  const _ClusteringDialog({required this.onCluster});

  @override
  State<_ClusteringDialog> createState() => _ClusteringDialogState();
}

class _ClusteringDialogState extends State<_ClusteringDialog> {
  @override
  void initState() {
    super.initState();
    widget.onCluster();
  }

  @override
  Widget build(BuildContext context) {
    return const Center(child: CircularProgressIndicator());
  }
}

class PersonsView extends StatefulWidget {
  const PersonsView({super.key});

  @override
  State<PersonsView> createState() => _PersonsViewState();
}

class _PersonsViewState extends State<PersonsView> {
  List<Person> _persons = [];
  bool _loading = true;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadPersons();
  }

  Future<void> _loadPersons() async {
    debugPrint('[PERSONS] _loadPersons: 开始加载');
    setState(() {
      _loading = true;
      _error = null;
    });

    try {
      // Check if Rust FFI is initialized
      debugPrint('[PERSONS] _loadPersons: 检查Rust状态');
      final status = RustFFI.getStatus();
      debugPrint('[PERSONS] _loadPersons: status=$status');
      if (status.contains('"error"')) {
        setState(() {
          _error = 'Rust未初始化: $status';
          _loading = false;
        });
        return;
      }

      debugPrint('[PERSONS] _loadPersons: 调用 listPersons');
      final jsonStr = RustFFI.listPersons();
      debugPrint('[PERSONS] _loadPersons: listPersons returned ${jsonStr.length} chars');
      if (jsonStr.startsWith('{"error')) {
        setState(() {
          _error = jsonStr;
          _loading = false;
        });
        return;
      }

      final List<dynamic> jsonList = jsonDecode(jsonStr);
      final persons = jsonList.map((e) => Person.fromJson(e)).toList();

      setState(() {
        _persons = persons;
        _loading = false;
      });
    } catch (e) {
      setState(() {
        _error = 'Failed to load persons: $e';
        _loading = false;
      });
    }
  }

  Future<void> _clusterFaces() async {
    debugPrint('[PERSONS] _clusterFaces: 开始聚类');
    final result = await showDialog<ClusterResult>(
      context: context,
      barrierDismissible: false,
      builder: (dialogContext) => _ClusteringDialog(
        onCluster: () => _doCluster(dialogContext),
      ),
    );

    if (result != null) {
      if (!mounted) return;
      debugPrint('[PERSONS] _clusterFaces: 聚类完成 assigned=${result.assigned} created=${result.created} failed=${result.failed}');
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(
          content: Text('聚类完成: 分配 ${result.assigned} 张, 新建 ${result.created} 人, 失败 ${result.failed} 张'),
        ),
      );
      _loadPersons();
    }
  }

  Future<void> _doCluster(BuildContext dialogContext) async {
    debugPrint('[PERSONS] _doCluster: 开始执行聚类');
    try {
      debugPrint('[PERSONS] _doCluster: 调用 RustFFI.clusterAll()');
      final jsonStr = RustFFI.clusterAll();
      debugPrint('[PERSONS] _doCluster: clusterAll returned ${jsonStr.length} chars');
      final json = jsonDecode(jsonStr) as Map<String, dynamic>;

      if (!mounted) return;

      if (json.containsKey('error')) {
        debugPrint('[PERSONS] _doCluster: 聚类失败 error=${json['error']}');
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('聚类失败: ${json['error']}')),
        );
        Navigator.of(dialogContext).pop(null);
        return;
      }

      final result = ClusterResult.fromJson(json);
      debugPrint('[PERSONS] _doCluster: 聚类成功 ClusterResult=$result');
      Navigator.of(dialogContext).pop(result);
    } catch (e) {
      debugPrint('[PERSONS] _doCluster: 聚类异常 $e');
      if (!mounted) return;
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('聚类失败: $e')),
      );
      Navigator.of(dialogContext).pop(null);
    }
  }

  Future<void> _renamePerson(Person person) async {
    final controller = TextEditingController(text: person.name);

    final newName = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('重命名人物'),
        content: TextField(
          controller: controller,
          decoration: const InputDecoration(
            labelText: '姓名',
            hintText: '输入新名称',
          ),
          autofocus: true,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('取消'),
          ),
          TextButton(
            onPressed: () => Navigator.of(context).pop(controller.text),
            child: const Text('确定'),
          ),
        ],
      ),
    );

    if (newName != null && newName.isNotEmpty && newName != person.name) {
      final result = RustFFI.renamePerson(person.id, newName);
      if (result == 0) {
        _loadPersons();
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('已重命名为: $newName')),
        );
      } else {
        if (!mounted) return;
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('重命名失败')),
        );
      }
    }
  }

  void _showPersonFaces(Person person) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: Text(person.name),
        content: Text('${person.faceCount} 张脸'),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('关闭'),
          ),
        ],
      ),
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('人物'),
        actions: [
          IconButton(
            icon: const Icon(Icons.refresh),
            onPressed: _loadPersons,
            tooltip: '刷新',
          ),
        ],
      ),
      body: _loading
          ? const Center(child: CircularProgressIndicator())
          : _error != null
              ? Center(
                  child: Column(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Text('错误: $_error', style: const TextStyle(color: Colors.red)),
                      const SizedBox(height: 16),
                      ElevatedButton(
                        onPressed: _loadPersons,
                        child: const Text('重试'),
                      ),
                    ],
                  ),
                )
              : _persons.isEmpty
                  ? Center(
                      child: Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          const Icon(Icons.group, size: 64, color: Colors.grey),
                          const SizedBox(height: 16),
                          const Text('暂无人物'),
                          const SizedBox(height: 8),
                          const Text('扫描照片后点击聚类按钮'),
                        ],
                      ),
                    )
                  : ListView.builder(
                      itemCount: _persons.length,
                      itemBuilder: (context, index) {
                        final person = _persons[index];
                        return ListTile(
                          leading: CircleAvatar(
                            child: Text(person.name.isNotEmpty ? person.name[0] : '?'),
                          ),
                          title: Text(person.name),
                          subtitle: Text('${person.faceCount} 张脸'),
                          trailing: const Icon(Icons.chevron_right),
                          onTap: () => _showPersonFaces(person),
                          onLongPress: () => _renamePerson(person),
                        );
                      },
                    ),
      floatingActionButton: FloatingActionButton(
        onPressed: () async {
          final confirm = await showDialog<bool>(
            context: context,
            builder: (context) => AlertDialog(
              title: const Text('执行聚类'),
              content: const Text('将对所有未分配的人脸进行聚类。这可能需要一些时间。'),
              actions: [
                TextButton(
                  onPressed: () => Navigator.of(context).pop(false),
                  child: const Text('取消'),
                ),
                TextButton(
                  onPressed: () => Navigator.of(context).pop(true),
                  child: const Text('确定'),
                ),
              ],
            ),
          );

          if (confirm == true) {
            await _clusterFaces();
          }
        },
        tooltip: '执行聚类',
        child: const Icon(Icons.group_add),
      ),
    );
  }
}
