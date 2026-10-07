-- Add session_active column to account table for logout persistence
ALTER TABLE account ADD COLUMN session_active INTEGER NOT NULL DEFAULT 1;
