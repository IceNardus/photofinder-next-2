import 'dart:convert';
import 'dart:io' show File;
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:image_cropper/image_cropper.dart';
import 'package:photo_manager/photo_manager.dart';
import 'package:share_plus/share_plus.dart';
import '../ffi/rust_ffi.dart';

class SearchResult {
  final int faceId;
  final int imageId;
  final double score;
  final int? personId;
  final bool isExpanded;
  final String? thumbnail;
  final String? assetId;

  SearchResult({
    required this.faceId,
    required this.imageId,
    required this.score,
    this.personId,
    this.isExpanded = false,
    this.thumbnail,
    this.assetId,
  });

  factory SearchResult.fromJson(Map<String, dynamic> json) {
    return SearchResult(
      faceId: json['face_id'] as int? ?? 0,
      imageId: json['image_id'] as int? ?? 0,
      score: (json['score'] as num?)?.toDouble() ?? 0.0,
      personId: json['person_id'] as int?,
      isExpanded: json['is_expanded'] as bool? ?? false,
      thumbnail: json['thumbnail'] as String?,
      assetId: json['asset_id'] as String?,
    );
  }
}

class SearchView extends StatefulWidget {
  const SearchView({super.key});

  @override
  State<SearchView> createState() => _SearchViewState();
}

class _SearchViewState extends State<SearchView> with SingleTickerProviderStateMixin {
  final ImagePicker _picker = ImagePicker();
  late TabController _tabController;
  final ScrollController _faceScrollController = ScrollController();
  final ScrollController _objectScrollController = ScrollController();

  Uint8List? _originalImage;    // 原始选择的图片
  Uint8List? _croppedImage;     // 裁剪后的图片
  List<SearchResult> _faceResults = [];
  List<SearchResult> _objectResults = [];
  bool _isSearching = false;
  String _statusText = '选择照片并裁剪区域开始搜索';
  String _searchMode = 'face';

  // 分页状态
  int _faceTotal = 0;
  int _objectTotal = 0;
  bool _isLoadingMore = false;
  static const int _pageSize = 20;

  // 多选状态
  final Set<int> _selectedIndices = {};
  bool _isSelectionMode = false;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 2, vsync: this);
    _tabController.addListener(() {
      setState(() {
        _searchMode = _tabController.index == 0 ? 'face' : 'object';
      });
    });

    // 监听滚动实现无限滚动
    _faceScrollController.addListener(_onFaceScroll);
    _objectScrollController.addListener(_onObjectScroll);
  }

  @override
  void dispose() {
    _tabController.dispose();
    _faceScrollController.dispose();
    _objectScrollController.dispose();
    super.dispose();
  }

  void _onFaceScroll() {
    if (_faceScrollController.position.pixels >=
        _faceScrollController.position.maxScrollExtent - 200) {
      _loadMore('face');
    }
  }

  void _onObjectScroll() {
    if (_objectScrollController.position.pixels >=
        _objectScrollController.position.maxScrollExtent - 200) {
      _loadMore('object');
    }
  }

  Future<void> _loadMore(String mode) async {
    final results = mode == 'face' ? _faceResults : _objectResults;
    final total = mode == 'face' ? _faceTotal : _objectTotal;

    if (_isLoadingMore || results.length >= total) return;

    setState(() => _isLoadingMore = true);

    try {
      final offset = results.length;
      final resultJson = mode == 'face'
          ? RustFFI.searchFace(_croppedImage!, _pageSize, offset: offset)
          : RustFFI.searchObject(_croppedImage!, _pageSize, offset: offset);

      final data = jsonDecode(resultJson) as Map<String, dynamic>;
      final newResults = (data['results'] as List)
          .map((item) => SearchResult.fromJson(item as Map<String, dynamic>))
          .toList();

      setState(() {
        if (mode == 'face') {
          _faceResults = [..._faceResults, ...newResults];
        } else {
          _objectResults = [..._objectResults, ...newResults];
        }
      });
    } catch (e) {
      debugPrint('Load more error: $e');
    } finally {
      setState(() => _isLoadingMore = false);
    }
  }

  void _toggleSelection(int index) {
    setState(() {
      if (_selectedIndices.contains(index)) {
        _selectedIndices.remove(index);
      } else {
        _selectedIndices.add(index);
      }
    });
  }

  void _clearSelection() {
    setState(() {
      _selectedIndices.clear();
      _isSelectionMode = false;
    });
  }

  Future<void> _shareSelected() async {
    final results = _currentResults;
    final selected = _selectedIndices
        .map((i) => results[i])
        .where((r) => r.assetId != null)
        .toList();
    if (selected.isEmpty) return;

    final files = <XFile>[];
    for (final result in selected) {
      final assetId = result.assetId!;
      if (assetId.startsWith('/')) {
        // File path - share directly
        files.add(XFile(assetId));
      } else {
        // Asset ID - get file path
        final asset = await AssetEntity.fromId(assetId);
        if (asset != null) {
          final file = await asset.file;
          if (file != null) {
            files.add(XFile(file.path));
          }
        }
      }
    }

    if (files.isNotEmpty) {
      // sharePositionOrigin is required on iOS when sharing from sheets/popovers
      await Share.shareXFiles(
        files,
        sharePositionOrigin: const Rect.fromLTWH(0, 0, 1, 1),
      );
    }

    if (mounted) _clearSelection();
  }

  Future<void> _showSingleImageInfo(SearchResult result) async {
    if (result.assetId == null) return;

    final info = StringBuffer();
    info.writeln('人脸ID: ${result.faceId}');
    info.writeln('相似度: ${(result.score * 100).toStringAsFixed(1)}%');
    if (result.personId != null) {
      info.writeln('人物ID: ${result.personId}');
    }
    if (result.isExpanded) {
      info.writeln('(扩展结果)');
    }
    info.writeln('');

    // Try to get asset info if it's an asset ID, otherwise show file info
    if (result.assetId!.startsWith('/')) {
      // File path
      info.writeln('--- 原图属性 (文件路径) ---');
      info.writeln('路径: ${result.assetId}');
      final file = File(result.assetId!);
      if (await file.exists()) {
        final stat = await file.stat();
        info.writeln('大小: ${(stat.size / 1024).toStringAsFixed(1)} KB');
        info.writeln('修改时间: $stat.modified');
      }
    } else {
      // Asset ID
      final asset = await AssetEntity.fromId(result.assetId!);
      if (asset != null) {
        info.writeln('--- 原图属性 ---');
        info.writeln('宽度: ${asset.width}');
        info.writeln('高度: ${asset.height}');
        final f = await asset.file;
        if (f != null) {
          final fileSize = await f.length();
          info.writeln('文件大小: ${(fileSize / 1024).toStringAsFixed(1)} KB');
        }
        info.writeln('创建时间: ${asset.createDateTime}');
        info.writeln('修改时间: ${asset.modifiedDateTime}');
        if (asset.title != null) {
          info.writeln('标题: ${asset.title}');
        }
        info.writeln('类型: ${asset.mimeType}');
      }
    }

    if (mounted) {
      showDialog(
        context: context,
        builder: (ctx) => AlertDialog(
          title: const Text('图片信息'),
          content: SingleChildScrollView(
            child: Text(info.toString()),
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(ctx),
              child: const Text('关闭'),
            ),
          ],
        ),
      );
    }
  }

  Future<void> _previewImages() async {
    final results = _currentResults;
    final selected = _selectedIndices.map((i) => results[i]).where((r) => r.assetId != null).toList();
    if (selected.isEmpty) return;

    if (!mounted) return;
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (ctx) => _PreviewPage(
          results: selected,
          initialIndex: 0,
          onDelete: (assetId) {
            setState(() {
              if (_searchMode == 'face') {
                _faceResults.removeWhere((r) => r.assetId == assetId);
              } else {
                _objectResults.removeWhere((r) => r.assetId == assetId);
              }
              _selectedIndices.clear();
              _isSelectionMode = false;
            });
          },
        ),
      ),
    );
  }

  Future<void> _openPreview(SearchResult result, int index) async {
    if (!mounted) return;
    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (ctx) => _PreviewPage(
          results: [result],
          initialIndex: 0,
          onDelete: (assetId) {
            setState(() {
              if (_searchMode == 'face') {
                _faceResults.removeWhere((r) => r.assetId == assetId);
              } else {
                _objectResults.removeWhere((r) => r.assetId == assetId);
              }
            });
          },
        ),
      ),
    );
  }

  Future<void> _deleteSelected() async {
    final results = _currentResults;
    final selected = _selectedIndices.map((i) => results[i]).where((r) => r.assetId != null).toList();
    if (selected.isEmpty) return;

    final confirmed = await showDialog<bool>(
      context: context,
      builder: (ctx) => AlertDialog(
        title: const Text('确认删除'),
        content: Text('确定要删除选中的 ${selected.length} 张图片吗？此操作不可撤销。'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(ctx, false),
            child: const Text('取消'),
          ),
          TextButton(
            onPressed: () => Navigator.pop(ctx, true),
            style: TextButton.styleFrom(foregroundColor: Colors.red),
            child: const Text('删除'),
          ),
        ],
      ),
    );

    if (confirmed != true) return;

    int deleted = 0;
    for (final result in selected) {
      try {
        final assetId = result.assetId!;
        if (assetId.startsWith('/')) {
          // File path - delete directly
          final file = File(assetId);
          if (await file.exists()) {
            await file.delete();
            deleted++;
          }
        } else {
          // Asset ID - use photo_manager
          await PhotoManager.editor.deleteWithIds([assetId]);
          deleted++;
        }
      } catch (e) {
        debugPrint('Delete failed: $e');
      }
    }

    if (mounted) {
      setState(() {
        for (final result in selected) {
          if (_searchMode == 'face') {
            _faceResults.removeWhere((r) => r.assetId == result.assetId);
          } else {
            _objectResults.removeWhere((r) => r.assetId == result.assetId);
          }
        }
        _selectedIndices.clear();
        _isSelectionMode = false;
        _statusText = '已删除 $deleted 张图片';
      });
    }
  }

  List<SearchResult> get _currentResults => _searchMode == 'face' ? _faceResults : _objectResults;

  void _performAction(void Function(List<SearchResult> results) action) {
    final results = _currentResults;
    final selected = _selectedIndices.map((i) => results[i]).toList();
    if (selected.isNotEmpty) {
      action(selected);
    }
  }

  Future<void> _pickAndCropImage() async {
    try {
      final XFile? picked = await _picker.pickImage(source: ImageSource.gallery);
      if (picked == null) return;

      final Uint8List imageBytes = await picked.readAsBytes();
      setState(() {
        _originalImage = imageBytes;
        _croppedImage = null;
        _faceResults = [];
        _objectResults = [];
        _statusText = '请裁剪要搜索的区域';
      });

      // 裁剪图片
      CroppedFile? cropped = await ImageCropper().cropImage(
        sourcePath: picked.path,
        uiSettings: [
          IOSUiSettings(
            title: '裁剪搜索区域',
            aspectRatioLockEnabled: false,
            resetAspectRatioEnabled: true,
            aspectRatioPickerButtonHidden: false,
            rotateButtonsHidden: false,
            rotateClockwiseButtonHidden: false,
          ),
        ],
      );

      if (cropped != null) {
        final Uint8List croppedBytes = await cropped.readAsBytes();
        setState(() {
          _croppedImage = croppedBytes;
          _statusText = '已裁剪完成，点击搜索按钮进行${_searchMode == 'face' ? '人脸' : '物体'}搜索';
        });
      }
    } catch (e) {
      debugPrint('Failed to pick/crop image: $e');
      setState(() {
        _statusText = '选择失败: $e';
      });
    }
  }

  Future<void> _search() async {
    if (_isSearching) return;

    // 如果没有裁剪图片，先走选择流程
    if (_croppedImage == null) {
      await _pickAndCropImage();
      // 等待 UI 有机会刷新
      await Future.delayed(const Duration(milliseconds: 100));
      if (_croppedImage == null) return;
    }

    // 先设置搜索状态，让 UI 有机会显示 loading
    setState(() {
      _isSearching = true;
      _statusText = _searchMode == 'face' ? '正在识别人脸...' : '正在识别物体...';
      _faceResults = [];
      _objectResults = [];
      _faceTotal = 0;
      _objectTotal = 0;
    });

    // 让 UI 刷新一次
    await Future.delayed(const Duration(milliseconds: 50));

    try {
      String resultJson;
      if (_searchMode == 'face') {
        // 传原始图片字节给 Rust，这样 bbox 坐标基准和入库时一致，vector_id 能正确匹配
        resultJson = RustFFI.searchFace(_originalImage!, _pageSize, offset: 0);
        debugPrint('[SEARCH] Raw result JSON: $resultJson');
        final data = jsonDecode(resultJson) as Map<String, dynamic>;
        final results = data['results'] as List;
        setState(() {
          _faceResults = results
              .map((item) => SearchResult.fromJson(item as Map<String, dynamic>))
              .toList();
          _faceTotal = data['total'] as int? ?? _faceResults.length;
          _statusText = _faceResults.isEmpty
              ? '未找到相似人脸'
              : '人脸搜索: 找到 ${_faceResults.length}/${_faceTotal} 个结果';
        });
      } else {
        resultJson = RustFFI.searchObject(_croppedImage!, _pageSize, offset: 0);
        debugPrint('[SEARCH] Raw result JSON: $resultJson');
        final data = jsonDecode(resultJson) as Map<String, dynamic>;
        final results = data['results'] as List;
        setState(() {
          _objectResults = results
              .map((item) => SearchResult.fromJson(item as Map<String, dynamic>))
              .toList();
          _objectTotal = data['total'] as int? ?? _objectResults.length;
          _statusText = _objectResults.isEmpty
              ? '未找到相似物体'
              : '物体搜索: 找到 ${_objectResults.length}/${_objectTotal} 个结果';
        });
      }
    } catch (e) {
      debugPrint('Search error: $e');
      setState(() {
        _statusText = '搜索失败: $e';
      });
    } finally {
      setState(() {
        _isSearching = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Padding(
                padding: EdgeInsets.fromLTRB(20, 20, 20, 0),
                child: Text(
                  '以图搜图',
                  style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold),
                ),
              ),
              const SizedBox(height: 16),
              _buildCroppedImageCard(),
              const SizedBox(height: 12),
              _buildTabBar(),
              const SizedBox(height: 12),
              SizedBox(
                height: MediaQuery.of(context).size.height * 0.45,
                child: TabBarView(
                  controller: _tabController,
                  children: [
                    _buildFaceSearchContent(),
                    _buildObjectSearchContent(),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildCroppedImageCard() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Container(
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
                  '裁剪区域',
                  style: TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
                ),
                const Spacer(),
                if (_originalImage != null)
                  TextButton.icon(
                    onPressed: _pickAndCropImage,
                    icon: const Icon(Icons.refresh, size: 18),
                    label: const Text('重新选择'),
                  ),
              ],
            ),
            const SizedBox(height: 12),
            GestureDetector(
              onTap: _pickAndCropImage,
              child: Container(
                width: double.infinity,
                height: 180,
                decoration: BoxDecoration(
                  color: Colors.grey.shade200,
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(color: Colors.grey.shade300),
                ),
                child: _croppedImage != null
                    ? ClipRRect(
                        borderRadius: BorderRadius.circular(12),
                        child: Image.memory(
                          _croppedImage!,
                          fit: BoxFit.contain,
                        ),
                      )
                    : Column(
                        mainAxisAlignment: MainAxisAlignment.center,
                        children: [
                          Icon(
                            Icons.crop,
                            size: 48,
                            color: Colors.grey.shade400,
                          ),
                          const SizedBox(height: 12),
                          Text(
                            _originalImage != null
                                ? '点击裁剪搜索区域'
                                : '点击选择图片',
                            style: TextStyle(
                              fontSize: 14,
                              color: Colors.grey.shade600,
                            ),
                          ),
                        ],
                      ),
              ),
            ),
            const SizedBox(height: 8),
            Text(
              '提示: 选择图片后可裁剪特定区域进行搜索',
              style: TextStyle(
                fontSize: 12,
                color: Colors.grey.shade500,
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildTabBar() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 20),
      child: Container(
        decoration: BoxDecoration(
          color: const Color(0xFFF5F0E8),
          borderRadius: BorderRadius.circular(12),
        ),
        child: TabBar(
          controller: _tabController,
          indicator: BoxDecoration(
            color: const Color(0xFFC67B5C),
            borderRadius: BorderRadius.circular(10),
          ),
          indicatorSize: TabBarIndicatorSize.tab,
          labelColor: Colors.white,
          unselectedLabelColor: const Color(0xFFC67B5C),
          labelStyle: const TextStyle(fontWeight: FontWeight.w600, fontSize: 14),
          dividerColor: Colors.transparent,
          tabs: const [
            Tab(text: '人脸搜索'),
            Tab(text: '物体搜索'),
          ],
        ),
      ),
    );
  }

  Widget _buildFaceSearchContent() {
    return Padding(
      padding: const EdgeInsets.all(20),
      child: Column(
        children: [
          _buildSearchButton('识别人脸', Icons.face),
          const SizedBox(height: 16),
          Expanded(child: _isSearching ? _buildSearchingIndicator('正在识别人脸...') : _buildResultsGrid(_faceResults, '人脸')),
        ],
      ),
    );
  }

  Widget _buildObjectSearchContent() {
    return Padding(
      padding: const EdgeInsets.all(20),
      child: Column(
        children: [
          _buildSearchButton('识别物体', Icons.category),
          const SizedBox(height: 16),
          Expanded(child: _isSearching ? _buildSearchingIndicator('正在识别物体...') : _buildResultsGrid(_objectResults, '物体')),
        ],
      ),
    );
  }

  Widget _buildSearchingIndicator(String text) {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          const SizedBox(
            width: 48,
            height: 48,
            child: CircularProgressIndicator(
              strokeWidth: 3,
              valueColor: AlwaysStoppedAnimation(Color(0xFFC67B5C)),
            ),
          ),
          const SizedBox(height: 16),
          Text(
            text,
            style: TextStyle(fontSize: 14, color: Colors.grey.shade600),
          ),
        ],
      ),
    );
  }

  Widget _buildSearchButton(String label, IconData icon) {
    return SizedBox(
      width: double.infinity,
      child: ElevatedButton.icon(
        onPressed: _isSearching ? null : _search,
        style: ElevatedButton.styleFrom(
          backgroundColor: const Color(0xFFC67B5C),
          foregroundColor: Colors.white,
          padding: const EdgeInsets.symmetric(vertical: 14),
          shape: RoundedRectangleBorder(
            borderRadius: BorderRadius.circular(12),
          ),
        ),
        icon: _isSearching
            ? const SizedBox(
                width: 20,
                height: 20,
                child: CircularProgressIndicator(
                  strokeWidth: 2,
                  valueColor: AlwaysStoppedAnimation(Colors.white),
                ),
              )
            : Icon(icon),
        label: Text(_isSearching ? '搜索中...' : label),
      ),
    );
  }

  Widget _buildResultsGrid(List<SearchResult> results, String type) {
    if (_isSearching) {
      // 搜索中状态
      return Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const SizedBox(
              width: 48,
              height: 48,
              child: CircularProgressIndicator(
                strokeWidth: 3,
                valueColor: AlwaysStoppedAnimation(const Color(0xFFC67B5C)),
              ),
            ),
            const SizedBox(height: 16),
            Text(
              '搜索中...',
              style: TextStyle(fontSize: 16, color: Colors.grey.shade600),
            ),
            const SizedBox(height: 8),
            Text(
              '请稍候',
              style: TextStyle(fontSize: 12, color: Colors.grey.shade400),
            ),
          ],
        ),
      );
    }

    if (results.isEmpty) {
      return Center(
        child: SingleChildScrollView(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                type == '人脸' ? Icons.face : Icons.category,
                size: 64,
                color: Colors.grey.shade300,
              ),
              const SizedBox(height: 16),
              Text(
                '暂无${type}搜索结果',
                style: TextStyle(fontSize: 16, color: Colors.grey.shade400),
              ),
              const SizedBox(height: 8),
              Text(
                '选择图片并裁剪区域后点击搜索',
                style: TextStyle(fontSize: 12, color: Colors.grey.shade400),
              ),
            ],
          ),
        ),
      );
    }

    final total = type == '人脸' ? _faceTotal : _objectTotal;
    final hasMore = results.length < total;

    return Column(
      children: [
        Expanded(
          child: GridView.builder(
            controller: type == '人脸' ? _faceScrollController : _objectScrollController,
            gridDelegate: const SliverGridDelegateWithFixedCrossAxisCount(
              crossAxisCount: 3,
              crossAxisSpacing: 8,
              mainAxisSpacing: 8,
              childAspectRatio: 1,
            ),
            itemCount: results.length,
            itemBuilder: (context, index) {
              final result = results[index];
              return _buildResultTile(result, index, type);
            },
          ),
        ),
        // 加载更多指示器
        if (hasMore || _isLoadingMore)
          Container(
            padding: const EdgeInsets.all(16),
            child: Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                if (_isLoadingMore)
                  const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(
                      strokeWidth: 2,
                      valueColor: AlwaysStoppedAnimation(Color(0xFFC67B5C)),
                    ),
                  )
                else
                  Text(
                    '上拉加载更多',
                    style: TextStyle(fontSize: 12, color: Colors.grey.shade500),
                  ),
              ],
            ),
          ),
      ],
    );
  }

  Widget _buildResultTile(SearchResult result, int index, String type) {
    final isSelected = _selectedIndices.contains(index);
    return GestureDetector(
      onTap: () {
        if (_isSelectionMode) {
          _toggleSelection(index);
          if (_selectedIndices.isEmpty) {
            setState(() => _isSelectionMode = false);
          } else {
            _showActionSheet();
          }
        } else {
          // 非选择模式：单图预览
          if (result.assetId != null) {
            _openPreview(result, index);
          }
        }
      },
      onLongPress: () {
        setState(() {
          _isSelectionMode = true;
          _selectedIndices.add(index);
        });
        _showActionSheet();
      },
      child: Container(
        decoration: BoxDecoration(
          color: Colors.grey.shade200,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: isSelected
                ? const Color(0xFFC67B5C)
                : (result.isExpanded ? Colors.purple : Colors.grey.shade300),
            width: isSelected ? 3 : (result.isExpanded ? 2 : 1),
          ),
        ),
        child: Stack(
          children: [
            Center(
              child: _buildResultImage(result),
            ),
            // 选择指示器
            if (_isSelectionMode)
              Positioned(
                top: 4,
                left: 4,
                child: Container(
                  width: 24,
                  height: 24,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: isSelected
                        ? const Color(0xFFC67B5C)
                        : Colors.white.withOpacity(0.8),
                    border: Border.all(
                      color: isSelected
                          ? const Color(0xFFC67B5C)
                          : Colors.grey.shade400,
                      width: 2,
                    ),
                  ),
                  child: isSelected
                      ? const Icon(Icons.check, size: 16, color: Colors.white)
                      : null,
                ),
              ),
            // 相似度
            Positioned(
              bottom: 4,
              right: 4,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                decoration: BoxDecoration(
                  color: _getScoreColor(result.score),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: Text(
                  '${(result.score * 100).toStringAsFixed(0)}%',
                  style: const TextStyle(
                    fontSize: 10,
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Color _getScoreColor(double score) {
    if (score >= 0.8) return Colors.green;
    if (score >= 0.6) return Colors.orange;
    return Colors.red;
  }

  void _showActionSheet() {
    final count = _selectedIndices.length;
    if (count == 0) return;

    final results = _currentResults;
    final selected = _selectedIndices.map((i) => results[i]).toList();

    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.transparent,
      builder: (ctx) => Container(
        decoration: const BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.vertical(top: Radius.circular(20)),
        ),
        child: SafeArea(
          top: false,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              // 拖动条
              Container(
                margin: const EdgeInsets.only(top: 12),
                width: 40,
                height: 4,
                decoration: BoxDecoration(
                  color: Colors.grey.shade300,
                  borderRadius: BorderRadius.circular(2),
                ),
              ),
              Padding(
                padding: const EdgeInsets.all(16),
                child: Row(
                  children: [
                    Text(
                      '已选择 $count 项',
                      style: const TextStyle(
                        fontSize: 16,
                        fontWeight: FontWeight.w600,
                      ),
                    ),
                    const Spacer(),
                    TextButton(
                      onPressed: () => Navigator.pop(ctx),
                      child: const Text('取消'),
                    ),
                  ],
                ),
              ),
              const Divider(height: 1),
              // 单图：分享、详情、删除
              // 多图：分享、删除
              if (count == 1) ...[
                _buildSheetAction(
                  ctx: ctx,
                  icon: Icons.info_outline,
                  label: '查看详情',
                  onTap: () {
                    Navigator.pop(ctx);
                    _showSingleImageInfo(selected.first);
                  },
                ),
              ],
              _buildSheetAction(
                ctx: ctx,
                icon: Icons.share,
                label: '分享',
                onTap: () {
                  Navigator.pop(ctx);
                  _shareSelected();
                },
              ),
              _buildSheetAction(
                ctx: ctx,
                icon: Icons.delete_outline,
                label: '删除',
                onTap: () {
                  Navigator.pop(ctx);
                  _deleteSelected();
                },
                isDestructive: true,
              ),
              const SizedBox(height: 8),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildSheetAction({
    required BuildContext ctx,
    required IconData icon,
    required String label,
    required VoidCallback onTap,
    bool isDestructive = false,
  }) {
    return ListTile(
      leading: Icon(
        icon,
        color: isDestructive ? Colors.red : const Color(0xFFC67B5C),
      ),
      title: Text(
        label,
        style: TextStyle(
          color: isDestructive ? Colors.red : null,
        ),
      ),
      onTap: onTap,
    );
  }

  Widget _buildStatusBar() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: const Color(0xFFF5F0E8),
        border: Border(top: BorderSide(color: Colors.grey.shade300)),
      ),
      child: SafeArea(
        top: false,
        child: Row(
          children: [
            Icon(
              _faceResults.isNotEmpty || _objectResults.isNotEmpty
                  ? Icons.check_circle
                  : Icons.info_outline,
              color: const Color(0xFFC67B5C),
              size: 20,
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Text(
                _statusText,
                style: const TextStyle(fontSize: 14),
              ),
            ),
            if (_faceResults.isNotEmpty)
              _buildResultBadge('人脸', _faceResults.length),
            if (_objectResults.isNotEmpty)
              _buildResultBadge('物体', _objectResults.length),
          ],
        ),
      ),
    );
  }

  Widget _buildResultBadge(String type, int count) {
    return Container(
      margin: const EdgeInsets.only(left: 8),
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
      decoration: BoxDecoration(
        color: const Color(0xFFC67B5C),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Text(
        '$type: $count',
        style: const TextStyle(fontSize: 12, color: Colors.white, fontWeight: FontWeight.w600),
      ),
    );
  }

  Widget _buildResultImage(SearchResult result) {
    if (result.assetId == null) {
      return Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.image_not_supported,
            size: 36,
            color: Colors.grey.shade400,
          ),
          const SizedBox(height: 4),
          Text(
            '无图片',
            style: TextStyle(fontSize: 10, color: Colors.grey.shade600),
          ),
        ],
      );
    }

    return FutureBuilder<Uint8List?>(
      future: _loadAssetThumbnail(result.assetId!),
      builder: (context, snapshot) {
        if (snapshot.connectionState == ConnectionState.waiting) {
          return const CircularProgressIndicator(strokeWidth: 2);
        }
        if (snapshot.hasError || snapshot.data == null) {
          return Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(
                Icons.broken_image,
                size: 36,
                color: Colors.grey.shade400,
              ),
              const SizedBox(height: 4),
              Text(
                '加载失败',
                style: TextStyle(fontSize: 10, color: Colors.grey.shade600),
              ),
            ],
          );
        }
        return ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: Image.memory(
            snapshot.data!,
            fit: BoxFit.cover,
            width: double.infinity,
            height: double.infinity,
          ),
        );
      },
    );
  }

  Future<Uint8List?> _loadAssetThumbnail(String assetId) async {
    try {
      final asset = await AssetEntity.fromId(assetId);
      if (asset == null) return null;
      return await asset.thumbnailDataWithSize(const ThumbnailSize(200, 200));
    } catch (e) {
      debugPrint('Failed to load asset $assetId: $e');
      return null;
    }
  }
}

class _PreviewPage extends StatefulWidget {
  final List<SearchResult> results;
  final Function(String assetId) onDelete;
  final int initialIndex;

  const _PreviewPage({
    required this.results,
    required this.onDelete,
    this.initialIndex = 0,
  });

  @override
  State<_PreviewPage> createState() => _PreviewPageState();
}

class _PreviewPageState extends State<_PreviewPage> {
  late PageController _pageController;
  late int _currentIndex;

  @override
  void initState() {
    super.initState();
    _currentIndex = widget.initialIndex;
    _pageController = PageController(initialPage: widget.initialIndex);
  }

  @override
  void dispose() {
    _pageController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      appBar: AppBar(
        backgroundColor: Colors.black,
        foregroundColor: Colors.white,
        title: Text('${_currentIndex + 1} / ${widget.results.length}'),
        actions: [
          IconButton(
            icon: const Icon(Icons.delete_outline),
            onPressed: () async {
              final result = widget.results[_currentIndex];
              final confirmed = await showDialog<bool>(
                context: context,
                builder: (ctx) => AlertDialog(
                  title: const Text('确认删除'),
                  content: const Text('确定要删除这张图片吗？'),
                  actions: [
                    TextButton(
                      onPressed: () => Navigator.pop(ctx, false),
                      child: const Text('取消'),
                    ),
                    TextButton(
                      onPressed: () => Navigator.pop(ctx, true),
                      style: TextButton.styleFrom(foregroundColor: Colors.red),
                      child: const Text('删除'),
                    ),
                  ],
                ),
              );
              if (confirmed == true) {
                widget.onDelete(result.assetId!);
                if (widget.results.length <= 1) {
                  if (mounted) Navigator.pop(context);
                } else {
                  if (_currentIndex >= widget.results.length - 1) {
                    _currentIndex = widget.results.length - 2;
                  }
                  setState(() {});
                }
              }
            },
          ),
        ],
      ),
      body: PageView.builder(
        controller: _pageController,
        itemCount: widget.results.length,
        onPageChanged: (index) {
          setState(() {
            _currentIndex = index;
          });
        },
        itemBuilder: (context, index) {
          final result = widget.results[index];
          return FutureBuilder<Uint8List?>(
            future: _loadFullAsset(result.assetId!),
            builder: (context, snapshot) {
              if (snapshot.connectionState == ConnectionState.waiting) {
                return const Center(
                  child: CircularProgressIndicator(color: Colors.white),
                );
              }
              if (snapshot.hasError || snapshot.data == null) {
                return const Center(
                  child: Text(
                    '加载失败',
                    style: TextStyle(color: Colors.white),
                  ),
                );
              }
              return InteractiveViewer(
                minScale: 0.5,
                maxScale: 4.0,
                child: Center(
                  child: Image.memory(
                    snapshot.data!,
                    fit: BoxFit.contain,
                  ),
                ),
              );
            },
          );
        },
      ),
      bottomNavigationBar: Container(
        height: 80,
        color: Colors.black,
        child: SafeArea(
          child: ListView.builder(
            scrollDirection: Axis.horizontal,
            padding: const EdgeInsets.symmetric(horizontal: 8),
            itemCount: widget.results.length,
            itemBuilder: (context, index) {
              return GestureDetector(
                onTap: () {
                  _pageController.animateToPage(
                    index,
                    duration: const Duration(milliseconds: 300),
                    curve: Curves.easeInOut,
                  );
                },
                child: Container(
                  width: 60,
                  height: 60,
                  margin: const EdgeInsets.symmetric(horizontal: 4, vertical: 10),
                  decoration: BoxDecoration(
                    border: Border.all(
                      color: index == _currentIndex
                          ? const Color(0xFFC67B5C)
                          : Colors.transparent,
                      width: 2,
                    ),
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: FutureBuilder<Uint8List?>(
                    future: _loadThumb(widget.results[index].assetId!),
                    builder: (context, snapshot) {
                      if (snapshot.hasData && snapshot.data != null) {
                        return ClipRRect(
                          borderRadius: BorderRadius.circular(6),
                          child: Image.memory(
                            snapshot.data!,
                            fit: BoxFit.cover,
                          ),
                        );
                      }
                      return Container(
                        color: Colors.grey.shade800,
                        child: const Icon(
                          Icons.image,
                          color: Colors.white54,
                          size: 24,
                        ),
                      );
                    },
                  ),
                ),
              );
            },
          ),
        ),
      ),
    );
  }

  Future<Uint8List?> _loadFullAsset(String assetId) async {
    try {
      // assetId might be a file path (from visual search) or an asset ID (from photo_manager)
      if (assetId.startsWith('/')) {
        // It's a file path - load directly
        final file = File(assetId);
        if (await file.exists()) {
          return await file.readAsBytes();
        }
        return null;
      } else {
        // It's an asset ID - use photo_manager
        final asset = await AssetEntity.fromId(assetId);
        if (asset == null) return null;
        return await asset.thumbnailDataWithSize(const ThumbnailSize(1200, 1200));
      }
    } catch (e) {
      return null;
    }
  }

  Future<Uint8List?> _loadThumb(String assetId) async {
    try {
      // assetId might be a file path (from visual search) or an asset ID (from photo_manager)
      if (assetId.startsWith('/')) {
        // It's a file path - load directly
        final file = File(assetId);
        if (await file.exists()) {
          return await file.readAsBytes();
        }
        return null;
      } else {
        // It's an asset ID - use photo_manager
        final asset = await AssetEntity.fromId(assetId);
        if (asset == null) return null;
        return await asset.thumbnailDataWithSize(const ThumbnailSize(150, 150));
      }
    } catch (e) {
      return null;
    }
  }
}
