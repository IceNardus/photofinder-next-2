import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import 'package:photo_manager/photo_manager.dart';
import 'package:provider/provider.dart';
import '../ffi/rust_ffi.dart';
import '../license_store.dart';
import '../platform/model_path.dart';

class ScanView extends StatefulWidget {
  const ScanView({super.key});

  @override
  State<ScanView> createState() => _ScanViewState();
}

class _ScanViewState extends State<ScanView> {
  int _dbCount = 0;
  int _faceCount = 0;
  int _objectCount = 0;
  int _personCount = 0;
  int _totalPhotos = 0;
  int _processedPhotos = 0;
  bool _hasEverScanned = false; // 是否曾经扫描过
  String _currentFile = '';
  PermissionState _permissionState = PermissionState.notDetermined;
  bool get _isGranted => _permissionState == PermissionState.authorized || _permissionState == PermissionState.limited;

  // DEBUG: 只扫描一张图片测试速度
  static const bool _scanOnlyOne = false;

  @override
  void initState() {
    super.initState();
    _initRust();
    _checkPermission();
  }

  Future<void> _initRust() async {
    debugPrint('[SCAN] _initRust: 开始初始化...');
    try {
      // Only initialize if not already initialized
      final status = RustFFI.getStatus();
      debugPrint('[SCAN] _initRust: current status=$status');

      if (status.contains('"db_initialized":false')) {
        debugPrint('[SCAN] _initRust: need to initialize');
        final modelsPath = await ModelPath.getModelsDir();
        debugPrint('[SCAN] _initRust: modelsPath=$modelsPath');
        final setResult = RustFFI.setModelsDir(modelsPath);
        debugPrint('[SCAN] _initRust: setModelsDir result=$setResult');

        debugPrint('[SCAN] _initRust: 调用 RustFFI.init()...');
        await RustFFI.init();
        debugPrint('[SCAN] _initRust: init 完成');
      } else {
        debugPrint('[SCAN] _initRust: already initialized, skipping');
      }

      final newStatus = RustFFI.getStatus();
      debugPrint('[SCAN] _initRust: ONNX Status=$newStatus');

      debugPrint('[SCAN] _initRust: 调用 _loadDbCount()...');
      await _loadDbCount();
      debugPrint('[SCAN] _initRust: 完成');
    } catch (e, stack) {
      debugPrint('[SCAN] _initRust: ERROR $e');
      debugPrint('[SCAN] _initRust: stack=$stack');
      if (mounted) {
        setState(() {
          _currentFile = '初始化失败: $e';
        });
      }
    }
  }

  Future<void> _checkPermission() async {
    final state = await PhotoManager.requestPermissionExtend();
    setState(() {
      _permissionState = state;
    });
  }

  Future<void> _loadDbCount() async {
    debugPrint('[SCAN] _loadDbCount: 开始加载数据库统计');
    try {
      debugPrint('[SCAN] _loadDbCount: 调用 RustFFI.getStatistics()');
      final statsJson = RustFFI.getStatistics();
      debugPrint('[SCAN] _loadDbCount: getStatistics returned ${statsJson.length} chars');
      final stats = jsonDecode(statsJson) as Map<String, dynamic>;
      debugPrint('[SCAN] _loadDbCount: stats=$stats');
      setState(() {
        _dbCount = stats['image_count'] as int? ?? 0;
        _faceCount = stats['face_count'] as int? ?? 0;
        _objectCount = stats['object_count'] as int? ?? 0;
      });

      // Load person count
      try {
        debugPrint('[SCAN] _loadDbCount: 调用 RustFFI.listPersons()');
        final personsJson = RustFFI.listPersons();
        debugPrint('[SCAN] _loadDbCount: listPersons returned ${personsJson.length} chars');
        final persons = jsonDecode(personsJson) as List;
        debugPrint('[SCAN] _loadDbCount: ${persons.length} persons');
        setState(() {
          _personCount = persons.length;
        });
      } catch (e) {
        debugPrint('[SCAN] _loadDbCount: Failed to load person count: $e');
      }
    } catch (e) {
      debugPrint('[SCAN] _loadDbCount: Failed to load DB count: $e');
    }
    debugPrint('[SCAN] _loadDbCount: 完成, _dbCount=$_dbCount, _faceCount=$_faceCount, _personCount=$_personCount');
  }

  Future<void> _startScan() async {
    debugPrint('[SCAN] _startScan: 开始扫描流程');

    // 清空旧数据，确保重新扫描
    debugPrint('[SCAN] _startScan: 清空旧数据库...');
    RustFFI.clearDatabase();
    debugPrint('[SCAN] _startScan: 数据库已清空');

    if (!_isGranted) {
      debugPrint('[SCAN] _startScan: 权限未授权，检查权限...');
      await _checkPermission();
      if (!_isGranted) {
        debugPrint('[SCAN] _startScan: 权限仍然未授权，显示对话框');
        _showPermissionDeniedDialog();
        return;
      }
    }

    setState(() {
      _processedPhotos = 0;
      _currentFile = '正在获取照片列表...';
    });

    // 设置全局扫描状态
    context.read<LicenseStore>().setScanning(true);

    // 启动 Rust AI worker
    debugPrint('[SCAN] _startScan: 启动 Rust AI worker...');
    final beginResult = RustFFI.scanBegin();
    debugPrint('[SCAN] _startScan: scanBegin result=$beginResult');
    if (beginResult != 0) {
      debugPrint('[SCAN] _startScan: scanBegin 失败');
      context.read<LicenseStore>().setScanning(false);
      setState(() {
        _currentFile = '启动扫描失败';
      });
      return;
    }

    try {
      debugPrint('[SCAN] _startScan: 获取照片album列表');
      // 获取所有照片album
      final List<AssetPathEntity> albums = await PhotoManager.getAssetPathList(
        type: RequestType.image,
        hasAll: true,
      );
      debugPrint('[SCAN] _startScan: 获取到 ${albums.length} 个album');

      if (albums.isEmpty) {
        debugPrint('[SCAN] _startScan: 没有找到照片album');
        RustFFI.scanEnd();
        // 入队空专辑不算扫描完成，保持状态让用户可以重试
        setState(() {
          _currentFile = '未找到照片';
        });
        return;
      }

      // 使用"全部"相册
      final allAlbums = albums.firstWhere(
        (album) => album.isAll,
        orElse: () => albums.first,
      );
      debugPrint('[SCAN] _startScan: 使用album: ${allAlbums.name}');

      // 获取照片总数
      final totalCount = await allAlbums.assetCountAsync;
      debugPrint('[SCAN] _startScan: 照片总数=$totalCount');
      setState(() {
        _totalPhotos = totalCount;
        _currentFile = '开始扫描 $totalCount 张照片...';
      });

      // 分页获取照片并入队
      const pageSize = 100;
      int page = 0;
      int attemptedCount = 0;

      while (context.read<LicenseStore>().isScanning) {
        debugPrint('[SCAN] _startScan: 获取第 ${page} 页, size=$pageSize');
        final assets = await allAlbums.getAssetListPaged(page: page, size: pageSize);
        if (assets.isEmpty) {
          debugPrint('[SCAN] _startScan: 第${page}页为空，结束扫描');
          break;
        }
        debugPrint('[SCAN] _startScan: 第${page}页获取到 ${assets.length} 张照片');

        for (final asset in assets) {
          if (!context.read<LicenseStore>().isScanning) break;

          // 获取照片文件路径
          try {
            final file = await asset.file;
            if (file == null) continue;
            final filePath = file.path;
            if (filePath == null || filePath.isEmpty) continue;

            debugPrint('[SCAN] _startScan: enqueue photo ${asset.id} path=$filePath');
            // 入队到 Rust worker，QUEUE_FULL时最多重试10次（2秒），之后跳过继续处理下一张
            int result;
            int retryCount = 0;
            bool enqueueSuccess = false;
            do {
              result = RustFFI.enqueuePhoto(int.parse(asset.id), filePath);
              if (result == 0) {
                enqueueSuccess = true;
                break; // 成功
              } else if (result == 2) {
                // INPUT_CLOSED - 扫描被取消
                debugPrint('[SCAN] _startScan: scan cancelled, stopping enqueue');
                break;
              } else if (result == 1) {
                // QUEUE_FULL - 重试10次（每次等待200ms = 2秒后跳过）
                retryCount++;
                if (retryCount >= 10) {
                  debugPrint('[SCAN] _startScan: photo ${asset.id} skip after $retryCount retries');
                  break;
                }
                await Future.delayed(const Duration(milliseconds: 200));
              } else {
                debugPrint('[SCAN] _startScan: enqueuePhoto failed result=$result');
                break;
              }
            } while (result != 0 && context.read<LicenseStore>().isScanning);

            attemptedCount++;
            debugPrint('[SCAN] enqueue ${asset.id} result=${enqueueSuccess ? "OK" : "SKIP"} attemptedCount=$attemptedCount');
            setState(() {
              _processedPhotos = attemptedCount;
              _currentFile = '已入队: $attemptedCount / $totalCount';
            });

            // DEBUG: 只扫描一张图片
            if (_scanOnlyOne) {
              debugPrint('[SCAN] _startScan: DEBUG模式，只扫描一张图片');
              break;
            }
          } catch (e) {
            debugPrint('[SCAN] _startScan: photo ${asset.id} enqueue exception: $e');
          }
        }

        // DEBUG: 只扫描一张图片时，提前退出
        if (_scanOnlyOne) {
          debugPrint('[SCAN] _startScan: DEBUG模式，入队完成');
          break;
        }

        page++;
        // 让UI有更新机会
        await Future.delayed(const Duration(milliseconds: 10));
      }

      debugPrint('[SCAN] _startScan: 入队完成, 共尝试入队 $attemptedCount 张照片');

      // 信号结束，等待 worker 处理完所有照片
      setState(() {
        _currentFile = '等待处理完成...';
      });

      RustFFI.scanEnd();
      debugPrint('[SCAN] _startScan: scanEnd 完成，轮询等待 worker...');

      // 轮询等待所有照片处理完成
      while (true) {
        await Future.delayed(const Duration(milliseconds: 500));
        final progressJson = RustFFI.getScanProgress();
        final progress = jsonDecode(progressJson) as Map<String, dynamic>;
        final processing = progress['processing'] as int? ?? 0;
        final completed = progress['completed'] as int? ?? 0;
        final visualCompleted = progress['visual_completed'] as int? ?? 0;
        final photoCompleted = progress['photo_completed'] as int? ?? 0;
        final queued = progress['queued'] as int? ?? 0;
        final finished = progress['finished'] as bool? ?? false;
        final inputClosed = progress['input_closed'] as bool? ?? false;

        debugPrint('[SCAN] _startScan: progress photo=$photoCompleted completed=$completed visual=$visualCompleted queued=$queued finished=$finished');
	        debugPrint('[SCAN] _startScan: _processedPhotos=$_processedPhotos attemptedCount=$attemptedCount');

        // Wait for finished=true (all workers have exited AND queues are drained)
        if (finished) {
          debugPrint('[SCAN] _startScan: 所有照片处理完成 (finished=true)');
          break;
        }

        setState(() {
          // Use photo_completed = min(completed, visual_completed) - only counts fully processed photos
          _processedPhotos = photoCompleted;
          _currentFile = '处理中: $photoCompleted / $attemptedCount';
        });
      }

      // 扫描完成后重新加载统计
      await _loadDbCount();

      // 扫描完成后自动触发聚类
      setState(() {
        _currentFile = '扫描完成，正在聚类...';
      });

      try {
        debugPrint('[SCAN] _startScan: 启动异步聚类');
        // 启动异步聚类
        final taskId = RustFFI.clusterAllAsync();
        if (taskId == -1) {
          debugPrint('[SCAN] _startScan: 聚类已在进行中，跳过');
        } else {
          debugPrint('[SCAN] _startScan: 聚类任务ID=$taskId, 等待结果...');
          // 轮询等待结果
          while (true) {
            await Future.delayed(const Duration(milliseconds: 500));
            final result = RustFFI.checkClusterResult(taskId);
            debugPrint('[SCAN] _startScan: 聚类检查结果=$result');
            final clusterJson = jsonDecode(result) as Map<String, dynamic>;

            if (clusterJson.containsKey('status')) {
              if (clusterJson['status'] == 'running') {
                debugPrint('[SCAN] _startScan: 聚类仍在运行...');
                continue;  // 仍在运行，继续等待
              }
              // not_found 或其他状态
              debugPrint('[SCAN] _startScan: 聚类任务状态异常: ${clusterJson['status']}');
              break;
            }

            // 有结果（可能是 error 或成功）
            if (clusterJson.containsKey('error')) {
              debugPrint('[SCAN] _startScan: 聚类失败: ${clusterJson['error']}');
            } else {
              final assigned = clusterJson['assigned'] ?? 0;
              final created = clusterJson['created'] ?? 0;
              final failed = clusterJson['failed'] ?? 0;
              debugPrint('[SCAN] _startScan: 聚类完成: 分配 $assigned 张, 新建 $created 人, 失败 $failed 张');
            }
            break;
          }
        }
      } catch (e) {
        debugPrint('[SCAN] _startScan: 聚类出错: $e');
      }

      // 聚类完成后重建 HNSW 索引（确保索引与 DB 同步）
      try {
        setState(() {
          _currentFile = '扫描完成，正在重建索引...';
        });
        debugPrint('[SCAN] _startScan: 重建HNSW索引');
        final rebuildResult = RustFFI.rebuildHnswIndices();
        if (rebuildResult == 0) {
          debugPrint('[SCAN] _startScan: HNSW 索引重建成功');
        } else {
          debugPrint('[SCAN] _startScan: HNSW 索引重建失败: $rebuildResult');
        }
      } catch (e) {
        debugPrint('[SCAN] _startScan: 重建索引出错: $e');
      }

      // 聚类完成后重新加载人物数量
      debugPrint('[SCAN] _startScan: 重新加载统计数据');
      await _loadDbCount();

      context.read<LicenseStore>().setScanning(false);
      setState(() {
        _hasEverScanned = true;
        _currentFile = '扫描完成！共处理 $_processedPhotos 张照片';
      });
      debugPrint('[SCAN] _startScan: 扫描流程全部完成');
    } catch (e, stack) {
      debugPrint('[SCAN] _startScan: 扫描异常: $e');
      debugPrint('[SCAN] _startScan: stack: $stack');
      context.read<LicenseStore>().setScanning(false);
      setState(() {
        _currentFile = '扫描失败: $e';
      });
    }
  }

  void _showPermissionDeniedDialog() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('需要照片访问权限'),
        content: const Text('请在设置中允许访问照片库，以便扫描您的照片。'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('取消'),
          ),
          TextButton(
            onPressed: () {
              Navigator.pop(context);
              PhotoManager.openSetting();
            },
            child: const Text('打开设置'),
          ),
        ],
      ),
    );
  }

  Future<void> _stopScan() async {
    // Signal cancel and close queue
    RustFFI.stopScan();
    RustFFI.scanEnd();
    debugPrint('[SCAN] _stopScan: waiting for worker to finish...');

    // 轮询等待 worker 退出
    while (true) {
      await Future.delayed(const Duration(milliseconds: 200));
      final progressJson = RustFFI.getScanProgress();
      final progress = jsonDecode(progressJson) as Map<String, dynamic>;
      final processing = progress['processing'] as int? ?? 0;
      if (processing == 0) {
        debugPrint('[SCAN] _stopScan: worker finished');
        break;
      }
    }

    await _loadDbCount();
    context.read<LicenseStore>().setScanning(false);
    setState(() {
      if (_processedPhotos > 0) _hasEverScanned = true;
      _currentFile = '已停止 (已处理 $_processedPhotos 张)';
    });
  }

  /// 从Asset读取照片并添加到数据库（使用字节方式）
  /// 返回 true 表示成功，false 表示失败
  Future<bool> _addPhotoFromBytes(dynamic asset) async {
    try {
      // asset.file 是 Future<File?>，需要先 await
      final file = await asset.file;
      if (file == null) {
        debugPrint('[SCAN] _addPhotoFromBytes: photo ${asset.id} file is null');
        return false;
      }
      final bytes = await file.readAsBytes();
      debugPrint('[SCAN] _addPhotoFromBytes: photo ${asset.id} 读取成功, ${bytes.length} bytes');
      // FFI calls go directly to Rust - no need for compute()
      final result = RustFFI.addPhotoFullFFI(bytes, asset.id);
      debugPrint('[SCAN] _addPhotoFromBytes: photo ${asset.id} addPhotoFullFFI result=$result');
      return result == 0;
    } catch (e) {
      debugPrint('[SCAN] _addPhotoFromBytes: photo ${asset.id} 异常: $e');
      return false;
    }
  }

  @override
  Widget build(BuildContext context) {
    return Consumer<LicenseStore>(
      builder: (context, store, _) {
        return Scaffold(
          body: SafeArea(
            child: Padding(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Text(
                        '扫描',
                        style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
                      ),
                      const SizedBox(width: 12),
                      _buildScanStatusChip(store.isScanning),
                    ],
                  ),
                  const SizedBox(height: 24),
                  _buildDbStatusCard(),
                  const SizedBox(height: 16),
                  _buildPermissionCard(),
                  const SizedBox(height: 16),
                  _buildActionCard(store.isScanning),
                  const SizedBox(height: 16),
                  if (store.isScanning) _buildProgressCard(),
                  const Spacer(),
                ],
              ),
            ),
          ),
        );
      },
    );
  }

  Widget _buildScanStatusChip(bool isScanning) {
    if (isScanning) {
      return Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
        decoration: BoxDecoration(
          color: Colors.orange.shade100,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            SizedBox(
              width: 12,
              height: 12,
              child: CircularProgressIndicator(
                strokeWidth: 2,
                color: Colors.orange.shade700,
              ),
            ),
            const SizedBox(width: 6),
            Text(
              '扫描中',
              style: TextStyle(
                fontSize: 12,
                color: Colors.orange.shade700,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      );
    } else if (_hasEverScanned && _dbCount > 0) {
      return Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
        decoration: BoxDecoration(
          color: Colors.green.shade100,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.check_circle, size: 14, color: Colors.green.shade700),
            const SizedBox(width: 4),
            Text(
              '已完成',
              style: TextStyle(
                fontSize: 12,
                color: Colors.green.shade700,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      );
    } else {
      return Container(
        padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
        decoration: BoxDecoration(
          color: Colors.grey.shade200,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(Icons.schedule, size: 14, color: Colors.grey.shade600),
            const SizedBox(width: 4),
            Text(
              '未扫描',
              style: TextStyle(
                fontSize: 12,
                color: Colors.grey.shade600,
                fontWeight: FontWeight.w500,
              ),
            ),
          ],
        ),
      );
    }
  }

  Widget _buildDbStatusCard() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFF5F0E8),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Text(
                '数据库状态',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
              ),
              const Spacer(),
              IconButton(
                onPressed: _loadDbCount,
                icon: const Icon(Icons.refresh, color: Color(0xFFC67B5C)),
                iconSize: 20,
                padding: EdgeInsets.zero,
                constraints: const BoxConstraints(),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Row(
            children: [
              _buildStatItem('照片', '$_dbCount', Icons.photo),
              const SizedBox(width: 16),
              _buildStatItem('人脸', '$_faceCount', Icons.face),
              const SizedBox(width: 16),
              _buildStatItem('人物', '$_personCount', Icons.person),
              const SizedBox(width: 16),
              _buildStatItem('物体', '$_objectCount', Icons.category),
            ],
          ),
          const SizedBox(height: 4),
          Text(
            _dbCount > 0
                ? '已准备好进行人脸和物体搜索'
                : '尚未扫描照片',
            style: TextStyle(
              fontSize: 11,
              color: Colors.grey.shade600,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatItem(String label, String value, IconData icon) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(icon, size: 14, color: const Color(0xFFC67B5C)),
            const SizedBox(width: 4),
            Text(
              label,
              style: TextStyle(
                fontSize: 11,
                color: Colors.grey.shade600,
              ),
            ),
          ],
        ),
        const SizedBox(height: 2),
        Text(
          value,
          style: const TextStyle(
            fontSize: 13,
            fontWeight: FontWeight.w600,
            color: Color(0xFFC67B5C),
          ),
        ),
      ],
    );
  }

  Widget _buildPermissionCard() {
    final isGranted = _isGranted;
    final isDenied = _permissionState == PermissionState.denied;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: isGranted
            ? const Color(0xFFF5F0E8)
            : const Color(0xFFFFF0F0),
        borderRadius: BorderRadius.circular(16),
        border: isDenied
            ? Border.all(color: Colors.red.shade200)
            : null,
      ),
      child: Row(
        children: [
          Icon(
            isGranted
                ? Icons.check_circle
                : (isDenied ? Icons.warning : Icons.photo_library),
            color: isGranted
                ? Colors.green
                : (isDenied ? Colors.red : Colors.grey),
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  '照片库权限',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                ),
                const SizedBox(height: 4),
                Text(
                  isGranted
                      ? '已授权'
                      : (isDenied ? '已拒绝 - 请在设置中开启' : '点击开始扫描申请权限'),
                  style: TextStyle(
                    fontSize: 14,
                    color: isGranted
                        ? Colors.green
                        : (isDenied ? Colors.red : Colors.grey),
                  ),
                ),
              ],
            ),
          ),
          if (isDenied)
            TextButton(
              onPressed: () => PhotoManager.openSetting(),
              child: const Text('设置'),
            ),
        ],
      ),
    );
  }

  Widget _buildActionCard(bool isScanning) {
    final canScan = _isGranted && !isScanning;
    final isLimited = _permissionState == PermissionState.limited;

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFFF5F0E8),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        children: [
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: canScan ? _startScan : (isScanning ? _stopScan : null),
              style: ElevatedButton.styleFrom(
                backgroundColor: const Color(0xFFC67B5C),
                foregroundColor: Colors.white,
                padding: const EdgeInsets.symmetric(vertical: 16),
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
              child: Text(isScanning ? '扫描中...' : '开始扫描'),
            ),
          ),
          const SizedBox(height: 12),
          Text(
            isScanning
                ? '正在后台扫描，请稍候...'
                : (isLimited
                    ? '部分照片权限 - 将在设置中授予完整访问权限'
                    : '点击开始扫描整个照片库'),
            style: const TextStyle(fontSize: 12, color: Colors.grey),
            textAlign: TextAlign.center,
          ),
        ],
      ),
    );
  }

  Widget _buildProgressCard() {
    // 显示进度: 处理完成数 / 入队总数
    final total = _totalPhotos > 0 ? _totalPhotos : 1;
    final percentage = (_processedPhotos / total * 100).clamp(0.0, 100.0);

    return Container(
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        color: const Color(0xFFF5F0E8),
        borderRadius: BorderRadius.circular(16),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              const Text(
                '扫描进度',
                style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
              ),
              Text(
                '${percentage.toStringAsFixed(0)}%',
                style: const TextStyle(
                  fontSize: 14,
                  color: Color(0xFFC67B5C),
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          LinearProgressIndicator(
            value: percentage / 100,
            backgroundColor: Colors.grey.shade300,
            valueColor: const AlwaysStoppedAnimation(Color(0xFFC67B5C)),
          ),
          if (_currentFile.isNotEmpty) ...[
            const SizedBox(height: 8),
            Text(
              _currentFile,
              style: const TextStyle(fontSize: 12, color: Colors.grey),
            ),
          ],
        ],
      ),
    );
  }
}
