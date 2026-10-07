-- 022_images_asset_id.sql — Add asset_id column to images table
--
-- Stores the photo_manager AssetEntity ID for each image.
-- This allows Flutter to load images via AssetEntity.fromId() instead of
-- relying on file paths which may not be accessible due to scoped storage.

-- Add asset_id column (nullable for existing rows)
ALTER TABLE images ADD COLUMN asset_id TEXT;

-- Create index for fast lookup by asset_id
CREATE INDEX IF NOT EXISTS idx_images_asset_id ON images(asset_id);
