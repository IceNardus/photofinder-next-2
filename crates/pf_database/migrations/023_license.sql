-- License and account tables for local membership system
-- Phase 1: Local-only, no server dependency

-- Account table (single local account, id = 1)
CREATE TABLE IF NOT EXISTS account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    email TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    last_login_at INTEGER,
    session_active INTEGER NOT NULL DEFAULT 1
);

-- License table (single license record, id = 1)
CREATE TABLE IF NOT EXISTS license (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    account_id INTEGER,
    activation_code TEXT NOT NULL,
    activated_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    duration_days INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('not_activated', 'active', 'expired', 'verification_failed')),
    last_network_check_at INTEGER,
    last_network_check_success INTEGER,
    FOREIGN KEY (account_id) REFERENCES account(id)
);

-- Index for account lookup
CREATE INDEX IF NOT EXISTS idx_account_email ON account(email);

-- Index for license lookup
CREATE INDEX IF NOT EXISTS idx_license_account ON license(account_id);
