//! SQLite storage for license and account data

use std::sync::Arc;

use crate::error::LicenseError;
use crate::types::{Account, LicenseStatus, LocalLicense};

/// Thread-safe license storage wrapper
pub struct LicenseStorage {
    db: Arc<pf_database::Database>,
}

impl LicenseStorage {
    /// Create a new license storage from database
    pub fn new(db: Arc<pf_database::Database>) -> Self {
        Self { db }
    }

    /// Create from existing database
    pub fn from_database(db: &Arc<pf_database::Database>) -> Result<Self, LicenseError> {
        Ok(Self::new(db.clone()))
    }

    /// Get a connection from the pool (uses interior mutability)
    fn conn(&self) -> Result<pf_database::PooledConnection, LicenseError> {
        // Database::connection takes &self, using pooled connections
        self.db
            .connection()
            .map_err(|e| LicenseError::storage(format!("Pool error: {}", e)))
    }

    // === Account operations ===

    /// Create a new account
    pub fn create_account(
        &self,
        email: &str,
        password_hash: &str,
    ) -> Result<Account, LicenseError> {
        let mut conn = self.conn()?;
        let now = current_timestamp();

        conn.execute(
            "INSERT INTO account (id, email, password_hash, created_at, session_active) VALUES (1, ?1, ?2, ?3, 1)",
            rusqlite::params![email, password_hash, now],
        )
        .map_err(|e| {
            if let rusqlite::Error::SqliteFailure(code, _) = e {
                if code.code == rusqlite::ErrorCode::ConstraintViolation {
                    return LicenseError::email_already_exists();
                }
            }
            LicenseError::storage(e.to_string())
        })?;

        Ok(Account {
            id: 1,
            email: email.to_string(),
            password_hash: password_hash.to_string(),
            created_at: now,
            last_login_at: None,
            session_active: true,
        })
    }

    /// Get account by email (checks password, ignores session_active)
    pub fn get_account_by_email(&self, email: &str) -> Result<Option<Account>, LicenseError> {
        let mut conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, email, password_hash, created_at, last_login_at, session_active FROM account WHERE email = ?1",
        )?;

        let account = stmt
            .query_row(rusqlite::params![email], |row| {
                Ok(Account {
                    id: row.get(0)?,
                    email: row.get(1)?,
                    password_hash: row.get(2)?,
                    created_at: row.get(3)?,
                    last_login_at: row.get(4)?,
                    session_active: row.get::<_, i64>(5)? == 1,
                })
            })
            .ok();

        Ok(account)
    }

    /// Get the single account (only if session_active = true)
    pub fn get_account(&self) -> Result<Option<Account>, LicenseError> {
        let mut conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, email, password_hash, created_at, last_login_at, session_active FROM account WHERE id = 1 AND session_active = 1",
        )?;

        let account = stmt.query_row([], |row| {
            Ok(Account {
                id: row.get(0)?,
                email: row.get(1)?,
                password_hash: row.get(2)?,
                created_at: row.get(3)?,
                last_login_at: row.get(4)?,
                session_active: row.get::<_, i64>(5)? == 1,
            })
        }).ok();

        Ok(account)
    }

    /// Deactivate session (logout - set session_active = 0)
    pub fn deactivate_session(&self, account_id: i64) -> Result<(), LicenseError> {
        let mut conn = self.conn()?;
        conn.execute(
            "UPDATE account SET session_active = 0 WHERE id = ?1",
            rusqlite::params![account_id],
        )?;
        Ok(())
    }

    /// Activate session (login - set session_active = 1)
    pub fn activate_session(&self, account_id: i64) -> Result<(), LicenseError> {
        let mut conn = self.conn()?;
        conn.execute(
            "UPDATE account SET session_active = 1 WHERE id = ?1",
            rusqlite::params![account_id],
        )?;
        Ok(())
    }

    /// Update last login time
    pub fn update_account_last_login(&self, account_id: i64) -> Result<(), LicenseError> {
        let mut conn = self.conn()?;
        let now = current_timestamp();
        conn.execute(
            "UPDATE account SET last_login_at = ?1 WHERE id = ?2",
            rusqlite::params![now, account_id],
        )?;
        Ok(())
    }

    /// Delete account
    pub fn delete_account(&self) -> Result<(), LicenseError> {
        let mut conn = self.conn()?;
        conn.execute("DELETE FROM account WHERE id = 1", [])?;
        Ok(())
    }

    // === License operations ===

    /// Create a new license
    pub fn create_license(&self, license: &LocalLicense) -> Result<(), LicenseError> {
        let mut conn = self.conn()?;
        conn.execute(
            "INSERT OR REPLACE INTO license (id, account_id, activation_code, activated_at, expires_at, duration_days, status, last_network_check_at, last_network_check_success) VALUES (1, ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![
                license.account_id,
                license.activation_code,
                license.activated_at,
                license.expires_at,
                license.duration_days,
                license.status.to_string(),
                license.last_network_check_at,
                license.last_network_check_success,
            ],
        )?;
        Ok(())
    }

    /// Get the single license (id = 1)
    pub fn get_license(&self) -> Result<Option<LocalLicense>, LicenseError> {
        let mut conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, account_id, activation_code, activated_at, expires_at, duration_days, status, last_network_check_at, last_network_check_success FROM license WHERE id = 1",
        )?;

        let license = stmt.query_row([], |row| {
            let status_str: String = row.get(6)?;
            Ok(LocalLicense {
                id: row.get(0)?,
                account_id: row.get(1)?,
                activation_code: row.get(2)?,
                activated_at: row.get(3)?,
                expires_at: row.get(4)?,
                duration_days: row.get(5)?,
                status: LicenseStatus::from(status_str.as_str()),
                last_network_check_at: row.get(7)?,
                last_network_check_success: row.get(8)?,
            })
        }).ok();

        Ok(license)
    }

    /// Update license
    pub fn update_license(&self, license: &LocalLicense) -> Result<(), LicenseError> {
        let mut conn = self.conn()?;
        conn.execute(
            "UPDATE license SET account_id = ?1, activation_code = ?2, activated_at = ?3, expires_at = ?4, duration_days = ?5, status = ?6, last_network_check_at = ?7, last_network_check_success = ?8 WHERE id = 1",
            rusqlite::params![
                license.account_id,
                license.activation_code,
                license.activated_at,
                license.expires_at,
                license.duration_days,
                license.status.to_string(),
                license.last_network_check_at,
                license.last_network_check_success,
            ],
        )?;
        Ok(())
    }

    /// Delete license
    pub fn delete_license(&self) -> Result<(), LicenseError> {
        let mut conn = self.conn()?;
        conn.execute("DELETE FROM license WHERE id = 1", [])?;
        Ok(())
    }
}

/// Get current Unix timestamp in seconds
fn current_timestamp() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
