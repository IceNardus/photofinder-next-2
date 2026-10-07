//! License error types

use serde::{Deserialize, Serialize};
use std::fmt;

use super::types::LicenseStatus;

/// License error codes
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum LicenseErrorCode {
    NotLoggedIn,
    NotActivated,
    Expired,
    VerificationFailed,
    InvalidCredentials,
    AccountNotFound,
    WrongPassword,
    EmailAlreadyExists,
    StorageError,
}

impl fmt::Display for LicenseErrorCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LicenseErrorCode::NotLoggedIn => write!(f, "not_logged_in"),
            LicenseErrorCode::NotActivated => write!(f, "not_activated"),
            LicenseErrorCode::Expired => write!(f, "expired"),
            LicenseErrorCode::VerificationFailed => write!(f, "verification_failed"),
            LicenseErrorCode::InvalidCredentials => write!(f, "invalid_credentials"),
            LicenseErrorCode::AccountNotFound => write!(f, "account_not_found"),
            LicenseErrorCode::WrongPassword => write!(f, "wrong_password"),
            LicenseErrorCode::EmailAlreadyExists => write!(f, "email_already_exists"),
            LicenseErrorCode::StorageError => write!(f, "storage_error"),
        }
    }
}

/// License error
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LicenseError {
    pub code: LicenseErrorCode,
    pub message: String,
}

impl LicenseError {
    pub fn new(code: LicenseErrorCode, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
        }
    }

    pub fn not_logged_in() -> Self {
        Self::new(LicenseErrorCode::NotLoggedIn, "Not logged in")
    }

    pub fn not_activated() -> Self {
        Self::new(LicenseErrorCode::NotActivated, "License not activated")
    }

    pub fn expired() -> Self {
        Self::new(LicenseErrorCode::Expired, "License expired")
    }

    pub fn verification_failed() -> Self {
        Self::new(
            LicenseErrorCode::VerificationFailed,
            "License verification failed",
        )
    }

    pub fn invalid_credentials() -> Self {
        Self::new(
            LicenseErrorCode::InvalidCredentials,
            "邮箱或密码错误",
        )
    }

    pub fn account_not_found() -> Self {
        Self::new(
            LicenseErrorCode::AccountNotFound,
            "账户不存在，请先注册",
        )
    }

    pub fn wrong_password() -> Self {
        Self::new(
            LicenseErrorCode::WrongPassword,
            "密码错误",
        )
    }

    pub fn email_already_exists() -> Self {
        Self::new(
            LicenseErrorCode::EmailAlreadyExists,
            "Email already registered",
        )
    }

    pub fn storage(msg: impl Into<String>) -> Self {
        Self::new(LicenseErrorCode::StorageError, msg)
    }

    pub fn from_status(status: LicenseStatus) -> Self {
        match status {
            LicenseStatus::NotActivated => Self::not_activated(),
            LicenseStatus::Expired => Self::expired(),
            LicenseStatus::VerificationFailed => Self::verification_failed(),
            LicenseStatus::Active => {
                Self::new(LicenseErrorCode::NotActivated, "Unexpected active state")
            }
        }
    }
}

impl fmt::Display for LicenseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:?}: {}", self.code, self.message)
    }
}

impl std::error::Error for LicenseError {}

impl From<r2d2::Error> for LicenseError {
    fn from(err: r2d2::Error) -> Self {
        Self::storage(format!("Database pool error: {}", err))
    }
}

impl From<rusqlite::Error> for LicenseError {
    fn from(err: rusqlite::Error) -> Self {
        Self::storage(format!("Database error: {}", err))
    }
}
