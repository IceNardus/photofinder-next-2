import 'dart:convert';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:path_provider/path_provider.dart';
import '../ffi/rust_ffi.dart';
import '../platform/model_path.dart';

class Account {
  final int id;
  final String email;
  final DateTime? lastLoginAt;

  Account({required this.id, required this.email, this.lastLoginAt});

  Map<String, dynamic> toJson() => {
        'id': id,
        'email': email,
        'last_login_at': lastLoginAt?.millisecondsSinceEpoch,
      };

  factory Account.fromJson(Map<String, dynamic> json) {
    return Account(
      id: json['id'] ?? 0,
      email: json['email'] ?? '',
      lastLoginAt: json['last_login_at'] != null
          ? DateTime.fromMillisecondsSinceEpoch(json['last_login_at'])
          : null,
    );
  }
}

class License {
  final int id;
  final int? accountId;
  final String activationCode;
  final DateTime activatedAt;
  final DateTime expiresAt;
  final int durationDays;
  final String status;

  License({
    required this.id,
    this.accountId,
    required this.activationCode,
    required this.activatedAt,
    required this.expiresAt,
    required this.durationDays,
    required this.status,
  });

  factory License.fromJson(Map<String, dynamic> json) {
    return License(
      id: json['id'] ?? 0,
      accountId: json['account_id'],
      activationCode: json['activation_code'] ?? '',
      activatedAt: json['activated_at'] != null
          ? DateTime.fromMillisecondsSinceEpoch(json['activated_at'] * 1000)
          : DateTime.now(),
      expiresAt: json['expires_at'] != null
          ? DateTime.fromMillisecondsSinceEpoch(json['expires_at'] * 1000)
          : DateTime.now(),
      durationDays: json['duration_days'] ?? 0,
      status: json['status'] ?? 'unknown',
    );
  }

  bool get isActive => status.toLowerCase() == 'active';

  int get remainingDays {
    final now = DateTime.now();
    if (expiresAt.isBefore(now)) return 0;
    return expiresAt.difference(now).inDays;
  }
}

class LicenseStore extends ChangeNotifier {
  Account? _account;
  License? _license;
  bool _loading = false;
  String? _error;
  bool _isScanning = false; // 全局扫描状态

  Account? get account => _account;
  License? get license => _license;
  bool get loading => _loading;
  String? get error => _error;
  bool get isScanning => _isScanning;

  bool get isLoggedIn => _account != null;
  bool get isLicenseActive => _license != null && _license!.isActive;
  int get remainingDays => _license?.remainingDays ?? 0;

  void setScanning(bool value) {
    _isScanning = value;
    notifyListeners();
  }

  Future<void> init() async {
    // Extract models from assets to device filesystem before initializing Rust FFI
    final modelsDir = await ModelPath.getModelsDir();
    debugPrint('[LicenseStore] init: models extracted to $modelsDir');

    // Set the models directory in Rust
    RustFFI.setModelsDir(modelsDir);

    // RustFFI.init() already calls initBackend() internally
    await RustFFI.init();

    // Restore account from SharedPreferences
    await _restoreAccount();

    await checkStatus();
  }

  Future<void> _restoreAccount() async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final file = File('${dir.path}/account.json');
      if (await file.exists()) {
        final content = await file.readAsString();
        _account = Account.fromJson(jsonDecode(content));
        debugPrint('[LicenseStore] _restoreAccount: restored ${_account?.email}');
      } else {
        debugPrint('[LicenseStore] _restoreAccount: no saved account');
      }
    } catch (e) {
      debugPrint('[LicenseStore] _restoreAccount error: $e');
    }
  }

  Future<void> _saveAccount() async {
    try {
      final dir = await getApplicationDocumentsDirectory();
      final file = File('${dir.path}/account.json');
      if (_account != null) {
        await file.writeAsString(jsonEncode(_account!.toJson()));
        debugPrint('[LicenseStore] _saveAccount: saved ${_account!.email}');
      } else {
        if (await file.exists()) {
          await file.delete();
        }
        debugPrint('[LicenseStore] _saveAccount: cleared');
      }
    } catch (e) {
      debugPrint('[LicenseStore] _saveAccount error: $e');
    }
  }

  Future<void> checkStatus() async {
    try {
      // Only update license status, don't overwrite account
      final licenseJson = RustFFI.getLicenseStatus();
      if (licenseJson == 'null' || licenseJson.isEmpty) {
        _license = null;
      } else {
        final parsed = jsonDecode(licenseJson);
        _license = parsed == null ? null : License.fromJson(parsed);
      }

      _error = null;
      notifyListeners();
    } catch (e) {
      _error = e.toString();
      notifyListeners();
    }
  }

  Future<bool> register(String email, String password) async {
    debugPrint('[LicenseStore] register: starting');
    _loading = true;
    _error = null;
    notifyListeners();

    try {
      final result = RustFFI.registerAccount(email, password);
      debugPrint('[LicenseStore] register: FFI raw result: $result');
      final json = jsonDecode(result);

      // Check for error - Rust uses "code" and "message" keys for errors
      if (json.containsKey('error') || json.containsKey('code')) {
        _error = json['message'] ?? json['error'] ?? '注册失败';
        _loading = false;
        notifyListeners();
        return false;
      }

      _account = Account.fromJson(json);
      debugPrint('[LicenseStore] register: _account.email=${_account?.email}');
      await _saveAccount();
      await checkStatus();
      _loading = false;
      notifyListeners();
      return true;
    } catch (e) {
      debugPrint('[LicenseStore] register error: $e');
      _error = e.toString();
      _loading = false;
      notifyListeners();
      return false;
    }
  }

  Future<bool> login(String email, String password) async {
    debugPrint('[LicenseStore] login: starting');
    _loading = true;
    _error = null;
    notifyListeners();

    try {
      debugPrint('[LicenseStore] login: calling FFI');
      final result = RustFFI.loginAccount(email, password);
      debugPrint('[LicenseStore] login: FFI raw result: $result');
      final json = jsonDecode(result);

      // Check for error - Rust uses "code" and "message" keys for errors
      if (json.containsKey('error') || json.containsKey('code')) {
        _error = json['message'] ?? json['error'] ?? '登录失败';
        _loading = false;
        notifyListeners();
        return false;
      }

      _account = Account.fromJson(json);
      debugPrint('[LicenseStore] login: _account.email=${_account?.email}');
      await _saveAccount();
      await checkStatus();
      _loading = false;
      notifyListeners();
      return true;
    } catch (e) {
      debugPrint('[LicenseStore] login error: $e');
      _error = e.toString();
      _loading = false;
      notifyListeners();
      return false;
    }
  }

  Future<bool> logout() async {
    _loading = true;
    notifyListeners();

    try {
      RustFFI.logoutAccount();
      _account = null;
      _license = null;
      await _saveAccount(); // Clear saved account
      _loading = false;
      notifyListeners();
      return true;
    } catch (e) {
      _error = e.toString();
      _loading = false;
      notifyListeners();
      return false;
    }
  }

  Future<bool> redeemCode(String code) async {
    _loading = true;
    _error = null;
    notifyListeners();

    try {
      final result = RustFFI.redeemCode(code);
      final json = jsonDecode(result);

      if (json.containsKey('error')) {
        _error = json['message'] ?? json['error'];
        _loading = false;
        notifyListeners();
        return false;
      }

      await checkStatus();
      _loading = false;
      notifyListeners();
      return true;
    } catch (e) {
      _error = e.toString();
      _loading = false;
      notifyListeners();
      return false;
    }
  }

  void clearError() {
    _error = null;
    notifyListeners();
  }
}
