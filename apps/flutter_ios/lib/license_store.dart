import 'dart:convert';
import 'package:flutter/foundation.dart';
import '../ffi/rust_ffi.dart';

class Account {
  final int id;
  final String email;
  final DateTime? lastLoginAt;

  Account({required this.id, required this.email, this.lastLoginAt});

  factory Account.fromJson(Map<String, dynamic> json) {
    return Account(
      id: json['id'] ?? 0,
      email: json['email'] ?? '',
      lastLoginAt: json['last_login_at'] != null
          ? DateTime.fromMillisecondsSinceEpoch(json['last_login_at'] * 1000)
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

  Account? get account => _account;
  License? get license => _license;
  bool get loading => _loading;
  String? get error => _error;

  bool get isLoggedIn => _account != null;
  bool get isLicenseActive => _license != null && _license!.isActive;
  int get remainingDays => _license?.remainingDays ?? 0;

  Future<void> init() async {
    try {
      await RustFFI.init();
      RustFFI.initBackend();
      await checkStatus();
    } catch (e) {
      _error = '初始化失败: ${e.toString()}';
      _loading = false;
      notifyListeners();
    }
  }

  Future<void> checkStatus() async {
    try {
      final accountJson = RustFFI.getAccount();
      if (accountJson == 'null' || accountJson.isEmpty) {
        _account = null;
      } else {
        _account = Account.fromJson(jsonDecode(accountJson));
      }

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
    _loading = true;
    _error = null;
    notifyListeners();

    try {
      final result = RustFFI.registerAccount(email, password);
      final json = jsonDecode(result);

      if (json.containsKey('error')) {
        _error = json['message'] ?? json['error'];
        _loading = false;
        notifyListeners();
        return false;
      }

      _account = Account.fromJson(json);
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

  Future<bool> login(String email, String password) async {
    _loading = true;
    _error = null;
    notifyListeners();

    try {
      final result = RustFFI.loginAccount(email, password);
      final json = jsonDecode(result);

      if (json.containsKey('error')) {
        _error = json['message'] ?? json['error'];
        _loading = false;
        notifyListeners();
        return false;
      }

      _account = Account.fromJson(json);
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

  Future<bool> logout() async {
    _loading = true;
    notifyListeners();

    try {
      RustFFI.logoutAccount();
      _account = null;
      _license = null;
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
