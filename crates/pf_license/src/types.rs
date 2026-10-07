//! License types

use serde::{Deserialize, Serialize};

/// Local account stored in SQLite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Account {
    pub id: i64,
    pub email: String,
    pub password_hash: String,
    pub created_at: i64,
    pub last_login_at: Option<i64>,
    /// Session active flag - when false, user is logged out and shouldn't auto-login
    pub session_active: bool,
}

/// Account DTO for frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccountDto {
    pub id: i64,
    pub email: String,
    pub created_at: i64,
    pub last_login_at: Option<i64>,
}

impl From<&Account> for AccountDto {
    fn from(account: &Account) -> Self {
        Self {
            id: account.id,
            email: account.email.clone(),
            created_at: account.created_at,
            last_login_at: account.last_login_at,
        }
    }
}

/// License status enum
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LicenseStatus {
    NotActivated,
    Active,
    Expired,
    VerificationFailed,
}

impl std::fmt::Display for LicenseStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LicenseStatus::NotActivated => write!(f, "not_activated"),
            LicenseStatus::Active => write!(f, "active"),
            LicenseStatus::Expired => write!(f, "expired"),
            LicenseStatus::VerificationFailed => write!(f, "verification_failed"),
        }
    }
}

impl From<&str> for LicenseStatus {
    fn from(s: &str) -> Self {
        match s {
            "active" => LicenseStatus::Active,
            "expired" => LicenseStatus::Expired,
            "verification_failed" => LicenseStatus::VerificationFailed,
            _ => LicenseStatus::NotActivated,
        }
    }
}

/// Local license stored in SQLite
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalLicense {
    pub id: i64,
    pub account_id: Option<i64>,
    pub activation_code: String,
    pub activated_at: i64,
    pub expires_at: i64,
    pub duration_days: i64,
    pub status: LicenseStatus,
    pub last_network_check_at: Option<i64>,
    pub last_network_check_success: Option<bool>,
}

/// License DTO for frontend
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseDto {
    pub status: LicenseStatus,
    pub expires_at: i64,
    pub remaining_days: i64,
    pub activation_code: Option<String>,
    pub last_network_check_at: Option<i64>,
    pub last_network_check_success: Option<bool>,
}

impl From<&LocalLicense> for LicenseDto {
    fn from(license: &LocalLicense) -> Self {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs() as i64)
            .unwrap_or(0);

        let remaining_days = if license.expires_at > now {
            ((license.expires_at - now) / (24 * 60 * 60)).max(0)
        } else {
            0
        };

        Self {
            status: license.status,
            expires_at: license.expires_at,
            remaining_days,
            activation_code: Some(license.activation_code.clone()),
            last_network_check_at: license.last_network_check_at,
            last_network_check_success: license.last_network_check_success,
        }
    }
}
