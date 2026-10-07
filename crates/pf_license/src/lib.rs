//! PhotoFinder License Module
//!
//! Local license/account management without server dependency.

pub mod error;
pub mod manager;
pub mod network;
pub mod storage;
pub mod types;

pub use error::{LicenseError, LicenseErrorCode};
pub use manager::LicenseManager;
pub use network::NetworkCheckResult;
pub use storage::LicenseStorage;
pub use types::{Account, AccountDto, LicenseDto, LicenseStatus, LocalLicense};

/// Default license duration in days
pub const DEFAULT_LICENSE_DURATION_DAYS: i64 = 30;

/// Network check interval in seconds (3 days)
pub const NETWORK_CHECK_INTERVAL_SECS: i64 = 3 * 24 * 60 * 60;
