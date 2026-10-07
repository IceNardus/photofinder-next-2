//! License manager — core business logic

use std::sync::Arc;
use std::time::SystemTime;

use sha2::{Digest, Sha256};

use crate::error::LicenseError;
use crate::network::{check_network_reachable, NetworkCheckResult};
use crate::storage::LicenseStorage;
use crate::types::{Account, AccountDto, LicenseDto, LicenseStatus, LocalLicense};
use crate::{DEFAULT_LICENSE_DURATION_DAYS, NETWORK_CHECK_INTERVAL_SECS};

/// Thread-safe license manager
pub struct LicenseManager {
    storage: Arc<LicenseStorage>,
    current_account: parking_lot::Mutex<Option<Account>>,
    /// Flag: user has logged out this session, don't auto-login on next startup
    session_logged_out: parking_lot::Mutex<bool>,
}

impl LicenseManager {
    /// Create a new license manager
    pub fn new(storage: Arc<LicenseStorage>) -> Self {
        Self {
            storage,
            current_account: parking_lot::Mutex::new(None),
            session_logged_out: parking_lot::Mutex::new(false),
        }
    }

    /// Initialize from existing storage (load current account)
    /// Only loads if user hasn't explicitly logged out this session
    pub fn initialize(&self) {
        // Don't auto-login if user logged out during this session
        {
            let logged_out = self.session_logged_out.lock();
            if *logged_out {
                return;
            }
        }

        if let Ok(Some(account)) = self.storage.get_account() {
            let mut curr = self.current_account.lock();
            *curr = Some(account);
        }
    }

    // === Account operations ===

    /// Register a new account
    pub fn register(&self, email: &str, password: &str) -> Result<AccountDto, LicenseError> {
        // Validate email
        if email.is_empty() || !email.contains('@') {
            return Err(LicenseError::new(
                crate::error::LicenseErrorCode::StorageError,
                "Invalid email format",
            ));
        }

        // Validate password
        if password.len() < 6 {
            return Err(LicenseError::new(
                crate::error::LicenseErrorCode::StorageError,
                "Password must be at least 6 characters",
            ));
        }

        // Hash password
        let password_hash = hash_password(password);

        // Create account
        let account = self.storage.create_account(email, &password_hash)?;

        // Store as current account
        {
            let mut curr = self.current_account.lock();
            *curr = Some(account.clone());
        }

        Ok(AccountDto::from(&account))
    }

    /// Login with email and password
    pub fn login(&self, email: &str, password: &str) -> Result<AccountDto, LicenseError> {
        // Find account by email (ignores session_active to allow re-login)
        let account = self
            .storage
            .get_account_by_email(email)?
            .ok_or(LicenseError::account_not_found())?;

        // Verify password
        let hash = hash_password(password);
        if hash != account.password_hash {
            return Err(LicenseError::wrong_password());
        }

        // Reactivate session (set session_active = 1)
        let _ = self.storage.activate_session(account.id);

        // Update last login time
        let _ = self.storage.update_account_last_login(account.id);

        // Store as current account
        let mut account_clone = account;
        account_clone.session_active = true;
        {
            let mut curr = self.current_account.lock();
            *curr = Some(account_clone.clone());
        }

        Ok(AccountDto::from(&account_clone))
    }

    /// Logout - deactivates session so won't auto-login on next startup
    pub fn logout(&self) {
        // Persist logout to database
        let _ = self.storage.deactivate_session(1);
        // Clear in-memory state
        let mut curr = self.current_account.lock();
        *curr = None;
    }

    /// Get current account
    pub fn get_account(&self) -> Option<AccountDto> {
        let curr = self.current_account.lock();
        curr.as_ref().map(AccountDto::from)
    }

    /// Check if logged in
    pub fn is_logged_in(&self) -> bool {
        let curr = self.current_account.lock();
        curr.is_some()
    }

    // === License operations ===

    /// Redeem a code (any non-empty code activates 30-day license)
    pub fn redeem_code(&self, code: &str) -> Result<LicenseDto, LicenseError> {
        // Must be logged in
        let account = {
            let curr = self.current_account.lock();
            curr.clone()
        }
        .ok_or(LicenseError::not_logged_in())?;

        // Code must be non-empty
        if code.trim().is_empty() {
            return Err(LicenseError::new(
                crate::error::LicenseErrorCode::StorageError,
                "Activation code cannot be empty",
            ));
        }

        let now = current_timestamp();
        let expires_at = now + (DEFAULT_LICENSE_DURATION_DAYS * 24 * 60 * 60);

        let license = LocalLicense {
            id: 1,
            account_id: Some(account.id),
            activation_code: code.trim().to_string(),
            activated_at: now,
            expires_at,
            duration_days: DEFAULT_LICENSE_DURATION_DAYS,
            status: LicenseStatus::Active,
            last_network_check_at: None,
            last_network_check_success: None,
        };

        self.storage.create_license(&license)?;

        Ok(LicenseDto::from(&license))
    }

    /// Require a valid license (for search operations)
    pub fn require_valid(&self) -> Result<(), LicenseError> {
        // Must be logged in
        if !self.is_logged_in() {
            return Err(LicenseError::not_logged_in());
        }

        // Check license status
        let license = self.storage.get_license()?.ok_or_else(|| {
            LicenseError::not_activated()
        })?;

        // Check if expired based on current time
        let now = current_timestamp();
        if license.status == LicenseStatus::Active && license.expires_at <= now {
            // Update status to expired
            let mut updated = license.clone();
            updated.status = LicenseStatus::Expired;
            self.storage.update_license(&updated)?;
            return Err(LicenseError::expired());
        }

        match license.status {
            LicenseStatus::Active => Ok(()),
            LicenseStatus::NotActivated => Err(LicenseError::not_activated()),
            LicenseStatus::Expired => Err(LicenseError::expired()),
            LicenseStatus::VerificationFailed => Err(LicenseError::verification_failed()),
        }
    }

    /// Get current license status
    pub fn get_license_status(&self) -> Result<Option<LicenseDto>, LicenseError> {
        let license = self.storage.get_license()?;

        // If license exists but is past expiry, treat as expired
        if let Some(ref lic) = license {
            let now = current_timestamp();
            if lic.status == LicenseStatus::Active && lic.expires_at <= now {
                let mut updated = lic.clone();
                updated.status = LicenseStatus::Expired;
                let _ = self.storage.update_license(&updated);
                let mut dto = LicenseDto::from(&updated);
                dto.status = LicenseStatus::Expired;
                return Ok(Some(dto));
            }
        }

        Ok(license.as_ref().map(LicenseDto::from))
    }

    /// Get remaining days
    pub fn get_remaining_days(&self) -> i64 {
        if let Ok(Some(license)) = self.storage.get_license() {
            let now = current_timestamp();
            if license.expires_at > now {
                return ((license.expires_at - now) / (24 * 60 * 60)).max(0);
            }
        }
        0
    }

    /// Check if license is active (for UI display)
    pub fn is_license_active(&self) -> bool {
        if let Ok(Some(license)) = self.storage.get_license() {
            let now = current_timestamp();
            return license.status == LicenseStatus::Active && license.expires_at > now;
        }
        false
    }

    // === Network check ===

    /// Check if network check is needed (more than 3 days since last check)
    pub fn needs_network_check(&self) -> bool {
        if let Ok(Some(license)) = self.storage.get_license() {
            if let Some(last_check) = license.last_network_check_at {
                let now = current_timestamp();
                return now - last_check >= NETWORK_CHECK_INTERVAL_SECS;
            }
        }
        // No check ever done, need to check
        true
    }

    /// Perform background network check (async)
    pub fn check_network_background(&self) {
        let storage = self.storage.clone();

        std::thread::spawn(move || {
            let result = check_network_reachable();
            let now = current_timestamp();

            if let Ok(Some(mut license)) = storage.get_license() {
                license.last_network_check_at = Some(now);
                license.last_network_check_success =
                    Some(result == NetworkCheckResult::Reachable);
                let _ = storage.update_license(&license);
            }
        });
    }

    /// Initialize network check (called on startup)
    pub fn init_network_check(&self) {
        if self.needs_network_check() {
            self.check_network_background();
        }
    }
}

/// Hash password with SHA-256 (simple, for local storage)
fn hash_password(password: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(password.as_bytes());
    // Use first 32 chars of hex encoding
    format!("{:x}", hasher.finalize())[..32].to_string()
}

/// Get current Unix timestamp
fn current_timestamp() -> i64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hash_password() {
        let hash = hash_password("test123");
        assert_eq!(hash.len(), 32);
    }

    #[test]
    fn test_timestamp() {
        let ts = current_timestamp();
        assert!(ts > 1700000000); // Should be a recent timestamp
    }
}
