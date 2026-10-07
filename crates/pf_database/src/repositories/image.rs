//! Image repository。

use chrono::{DateTime, Utc};
use rusqlite::params;
use serde::{Deserialize, Serialize};

use crate::error::DatabaseError;
use crate::Transaction;


/// 扫描状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScanStatus {
    /// 待扫描
    Pending,
    /// 已索引
    Indexed,
    /// 失败
    Failed,
}

impl ScanStatus {
    fn as_str(&self) -> &'static str {
        match self {
            ScanStatus::Pending => "pending",
            ScanStatus::Indexed => "indexed",
            ScanStatus::Failed => "failed",
        }
    }

    fn parse(s: &str) -> Result<Self, DatabaseError> {
        match s {
            "pending" => Ok(ScanStatus::Pending),
            "indexed" => Ok(ScanStatus::Indexed),
            "failed" => Ok(ScanStatus::Failed),
            other => Err(DatabaseError::Conversion(format!(
                "invalid ScanStatus: {other}"
            ))),
        }
    }
}

/// 缩略图状态。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThumbnailStatus {
    /// 待生成
    Pending,
    /// 已生成
    Generated,
    /// 失败
    Failed,
}

impl ThumbnailStatus {
    fn as_str(&self) -> &'static str {
        match self {
            ThumbnailStatus::Pending => "pending",
            ThumbnailStatus::Generated => "generated",
            ThumbnailStatus::Failed => "failed",
        }
    }

    fn parse(s: &str) -> Result<Self, DatabaseError> {
        match s {
            "pending" => Ok(ThumbnailStatus::Pending),
            "generated" => Ok(ThumbnailStatus::Generated),
            "failed" => Ok(ThumbnailStatus::Failed),
            other => Err(DatabaseError::Conversion(format!(
                "invalid ThumbnailStatus: {other}"
            ))),
        }
    }
}

/// 新插入的图片（不含 id）。
#[derive(Debug, Clone)]
pub struct NewImage {
    /// 文件路径
    pub path: String,
    /// BLAKE3 hash
    pub hash: String,
    /// 大小
    pub size: u64,
    /// 文件修改时间（Unix timestamp 秒）
    pub modified_time: i64,
    /// 宽
    pub width: u32,
    /// 高
    pub height: u32,
    /// 拍摄时间
    pub captured_at: Option<DateTime<Utc>>,
}

/// Image repository。
///
/// `<'tx, 'db>` 拆开两个生命周期：'tx 是 `&mut Transaction` 引用本身的生命周期，
/// 'db 是底层数据库事务的生命周期。两者独立，避免 invariance 阻止借用。
pub struct ImageRepository<'tx, 'db> {
    pub(crate) tx: &'tx mut Transaction<'db>,
}

impl<'tx, 'db> ImageRepository<'tx, 'db> {
    /// 插入新图片。返回 id。
    pub fn insert(&mut self, image: &NewImage) -> Result<i64, DatabaseError> {
        self.tx.execute(
            "INSERT INTO images
             (path, hash, size, modified_time, width, height, captured_at,
              thumbnail_status, scan_status, created_at, updated_at)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 'pending', 'pending',
              strftime('%Y-%m-%dT%H:%M:%fZ', 'now'),
              strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))",
            params![
                image.path,
                image.hash,
                image.size as i64,
                image.modified_time,
                image.width as i64,
                image.height as i64,
                image.captured_at.map(|dt| dt.to_rfc3339()),
            ],
        )?;
        Ok(self.tx.last_insert_rowid())
    }

    /// 按 id 列表批量查 path（仅返回存在的；保持入参顺序）。
    pub fn get_paths_for_ids(&self, ids: &[i64]) -> Result<Vec<(i64, String)>, DatabaseError> {
        if ids.is_empty() {
            return Ok(Vec::new());
        }
        let placeholders: Vec<String> = (1..=ids.len()).map(|i| format!("?{i}")).collect();
        let sql = format!(
            "SELECT id, path FROM images WHERE id IN ({})",
            placeholders.join(",")
        );
        let mut stmt = self.tx.prepare(&sql)?;
        let params_vec: Vec<&dyn rusqlite::ToSql> =
            ids.iter().map(|i| i as &dyn rusqlite::ToSql).collect();
        let mut rows = stmt.query(params_vec.as_slice())?;
        let mut map: std::collections::HashMap<i64, String> = std::collections::HashMap::new();
        while let Some(row) = rows.next()? {
            let id: i64 = row.get(0)?;
            let path: String = row.get(1)?;
            map.insert(id, path);
        }
        // 按入参顺序输出，缺失的 id 跳过
        let mut out = Vec::with_capacity(ids.len());
        for id in ids {
            if let Some(p) = map.remove(id) {
                out.push((*id, p));
            }
        }
        Ok(out)
    }

    /// 按 id 查询。
    pub fn get_by_id(&self, id: i64) -> Result<Option<ImageRow>, DatabaseError> {
        let mut stmt = self
            .tx
            .prepare("SELECT id, path, hash, size, width, height, captured_at, thumbnail_path, thumbnail_status, scan_status,
                    COALESCE(face_count, 0) as face_count, COALESCE(object_count, 0) as object_count
             FROM images WHERE id = ?1")?;
        let mut rows = stmt.query(params![id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row_to_image(row)?))
        } else {
            Ok(None)
        }
    }

    /// 按 id 批量查询（Audit B2：替代逐 candidate 单查事务）。
    ///
    /// 一次 `IN (...)` 子句查回多张图片。SQLite 单语句参数上限默认 32766，按
    /// 500 分块保证任意规模安全。
    pub fn list_by_ids(&self, ids: &[i64]) -> Result<Vec<ImageRow>, DatabaseError> {
        const CHUNK: usize = 500;
        let mut out = Vec::with_capacity(ids.len());
        for chunk in ids.chunks(CHUNK) {
            let placeholders = chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!(
                "SELECT id, path, hash, size, width, height, captured_at, thumbnail_path, thumbnail_status, scan_status,
                        COALESCE(face_count, 0) as face_count, COALESCE(object_count, 0) as object_count
                 FROM images WHERE id IN ({placeholders})"
            );
            let params: Vec<&dyn rusqlite::ToSql> = chunk
                .iter()
                .map(|v| v as &dyn rusqlite::ToSql)
                .collect();
            let mut stmt = self.tx.prepare(&sql)?;
            let mut rows = stmt.query(params.as_slice())?;
            while let Some(row) = rows.next()? {
                out.push(row_to_image(row)?);
            }
        }
        Ok(out)
    }

    /// 按路径查询。
    pub fn get_by_path(&self, path: &str) -> Result<Option<ImageRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, path, hash, size, width, height, captured_at, thumbnail_path, thumbnail_status, scan_status,
                    COALESCE(face_count, 0) as face_count, COALESCE(object_count, 0) as object_count
             FROM images WHERE path = ?1",
        )?;
        let mut rows = stmt.query(params![path])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row_to_image(row)?))
        } else {
            Ok(None)
        }
    }

    /// 按 hash 查询（去重）。
    pub fn get_by_hash(&self, hash: &str) -> Result<Option<ImageRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, path, hash, size, width, height, captured_at, thumbnail_path, thumbnail_status, scan_status,
                    COALESCE(face_count, 0) as face_count, COALESCE(object_count, 0) as object_count
             FROM images WHERE hash = ?1 LIMIT 1",
        )?;
        let mut rows = stmt.query(params![hash])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row_to_image(row)?))
        } else {
            Ok(None)
        }
    }

    /// 更新扫描状态。
    pub fn update_scan_status(&mut self, id: i64, status: ScanStatus) -> Result<(), DatabaseError> {
        self.tx.execute(
            "UPDATE images SET scan_status = ?1, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?2",
            params![status.as_str(), id],
        )?;
        Ok(())
    }

    /// 更新缩略图。
    pub fn update_thumbnail(
        &mut self,
        id: i64,
        path: &str,
        status: ThumbnailStatus,
    ) -> Result<(), DatabaseError> {
        self.tx.execute(
            "UPDATE images SET thumbnail_path = ?1, thumbnail_status = ?2, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?3",
            params![path, status.as_str(), id],
        )?;
        Ok(())
    }

    /// 更新人脸和物品计数（索引完成后回写）。
    /// 如果 face_count 或 object_count 为 -1，则不更新该字段。
    pub fn update_counts(&mut self, id: i64, face_count: i64, object_count: i64) -> Result<(), DatabaseError> {
        if face_count >= 0 && object_count >= 0 {
            self.tx.execute(
                "UPDATE images SET face_count = ?1, object_count = ?2, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?3",
                params![face_count, object_count, id],
            )?;
        } else if face_count >= 0 {
            self.tx.execute(
                "UPDATE images SET face_count = ?1, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?2",
                params![face_count, id],
            )?;
        } else if object_count >= 0 {
            self.tx.execute(
                "UPDATE images SET object_count = ?1, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?2",
                params![object_count, id],
            )?;
        }
        Ok(())
    }

    /// 仅更新物品计数（保留 face_count 不变）。
    pub fn update_object_count_only(&mut self, id: i64, object_count: i64) -> Result<(), DatabaseError> {
        self.tx.execute(
            "UPDATE images SET object_count = ?1, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?2",
            params![object_count, id],
        )?;
        Ok(())
    }

    /// 列出待索引的图片。
    pub fn list_pending(&self, limit: usize) -> Result<Vec<ImageRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, path, hash, size, width, height, captured_at, thumbnail_path, thumbnail_status, scan_status,
                    COALESCE(face_count, 0) as face_count, COALESCE(object_count, 0) as object_count
             FROM images WHERE scan_status = 'pending' LIMIT ?1",
        )?;
        let mut rows = stmt.query(params![limit as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_image(row)?);
        }
        Ok(out)
    }

    /// 统计。
    pub fn count(&self) -> Result<i64, DatabaseError> {
        let n: i64 = self
            .tx
            .query_row("SELECT COUNT(*) FROM images", [], |r| r.get(0))?;
        Ok(n)
    }

    /// 按状态统计。
    pub fn count_by_status(&self, status: ScanStatus) -> Result<i64, DatabaseError> {
        let n: i64 = self.tx.query_row(
            "SELECT COUNT(*) FROM images WHERE scan_status = ?1",
            params![status.as_str()],
            |r| r.get(0),
        )?;
        Ok(n)
    }

    /// 分页列出所有图片（按 id DESC）。
    pub fn list_paged(&self, limit: usize, offset: usize) -> Result<Vec<ImageRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, path, hash, size, width, height, captured_at, thumbnail_path, thumbnail_status, scan_status,
                    COALESCE(face_count, 0) as face_count, COALESCE(object_count, 0) as object_count
             FROM images ORDER BY id DESC LIMIT ?1 OFFSET ?2",
        )?;
        let mut rows = stmt.query(params![limit as i64, offset as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_image(row)?);
        }
        Ok(out)
    }
}

/// 数据库行 → ImageRow。
fn row_to_image(row: &rusqlite::Row<'_>) -> Result<ImageRow, DatabaseError> {
    let captured_str: Option<String> = row.get(6)?;
    let captured_at = captured_str
        .map(|s| {
            DateTime::parse_from_rfc3339(&s)
                .map(|dt| dt.with_timezone(&Utc))
                .map_err(|e| DatabaseError::Conversion(format!("captured_at parse: {e}")))
        })
        .transpose()?;

    let thumb_status_str: String = row.get(8)?;
    let scan_status_str: String = row.get(9)?;

    Ok(ImageRow {
        id: row.get(0)?,
        path: row.get(1)?,
        hash: row.get(2)?,
        size: row.get::<_, i64>(3)? as u64,
        width: row.get::<_, i64>(4)? as u32,
        height: row.get::<_, i64>(5)? as u32,
        captured_at,
        thumbnail_path: row.get(7)?,
        thumbnail_status: ThumbnailStatus::parse(&thumb_status_str)?,
        scan_status: ScanStatus::parse(&scan_status_str)?,
        face_count: row.get::<_, i64>(10).unwrap_or(0),
        object_count: row.get::<_, i64>(11).unwrap_or(0),
    })
}

/// Duplicate group row。
#[derive(Debug, Clone)]
pub struct DuplicateGroupRow {
    pub id: i64,
    pub hash: String,
    pub file_count: i64,
    pub created_at: DateTime<Utc>,
}

/// 添加 image 到 duplicate group（hash 已存在则 increment，不存在则创建）。
pub fn add_to_duplicate_group(tx: &Transaction<'_>, image_hash: &str) -> Result<DuplicateGroupRow, DatabaseError> {
    tx.execute(
        "INSERT INTO duplicate_groups (hash, file_count) VALUES (?1, 1)
         ON CONFLICT(hash) DO UPDATE SET file_count = file_count + 1",
        params![image_hash],
    )?;
    get_duplicate_group_by_hash(tx, image_hash)?
        .ok_or_else(|| DatabaseError::Conversion("failed to fetch duplicate group after insert".into()))
}

/// 按 hash 查询 duplicate group。
pub fn get_duplicate_group_by_hash(tx: &Transaction<'_>, hash: &str) -> Result<Option<DuplicateGroupRow>, DatabaseError> {
    let mut stmt = tx.prepare(
        "SELECT id, hash, file_count, created_at FROM duplicate_groups WHERE hash = ?1"
    )?;
    let mut rows = stmt.query(params![hash])?;
    if let Some(row) = rows.next()? {
        let created_str: String = row.get(3)?;
        let created_at = DateTime::parse_from_rfc3339(&created_str)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| DatabaseError::Conversion(format!("created_at parse: {e}")))?;
        Ok(Some(DuplicateGroupRow {
            id: row.get(0)?,
            hash: row.get(1)?,
            file_count: row.get(2)?,
            created_at,
        }))
    } else {
        Ok(None)
    }
}

/// 列出所有 duplicate groups（file_count > 1）。
pub fn list_duplicate_groups(tx: &Transaction<'_>, limit: usize, offset: usize) -> Result<Vec<DuplicateGroupRow>, DatabaseError> {
    let mut stmt = tx.prepare(
        "SELECT id, hash, file_count, created_at FROM duplicate_groups
         WHERE file_count > 1 ORDER BY file_count DESC LIMIT ?1 OFFSET ?2"
    )?;
    let mut rows = stmt.query(params![limit as i64, offset as i64])?;
    let mut out = Vec::new();
    while let Some(row) = rows.next()? {
        let created_str: String = row.get(3)?;
        let created_at = DateTime::parse_from_rfc3339(&created_str)
            .map(|dt| dt.with_timezone(&Utc))
            .map_err(|e| DatabaseError::Conversion(format!("created_at parse: {e}")))?;
        out.push(DuplicateGroupRow {
            id: row.get(0)?,
            hash: row.get(1)?,
            file_count: row.get(2)?,
            created_at,
        });
    }
    Ok(out)
}

/// 图片行（DB 层表示）。
#[derive(Debug, Clone)]
pub struct ImageRow {
    /// id
    pub id: i64,
    /// 路径
    pub path: String,
    /// hash
    pub hash: String,
    /// 大小
    pub size: u64,
    /// 宽
    pub width: u32,
    /// 高
    pub height: u32,
    /// 拍摄时间
    pub captured_at: Option<DateTime<Utc>>,
    /// 缩略图路径
    pub thumbnail_path: Option<String>,
    /// 缩略图状态
    pub thumbnail_status: ThumbnailStatus,
    /// 扫描状态
    pub scan_status: ScanStatus,
    /// 人脸数（索引完成后回写；旧 DB 缺列为 0）
    pub face_count: i64,
    /// 物品数（索引完成后回写；旧 DB 缺列为 0）
    pub object_count: i64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scan_status_roundtrip() {
        assert_eq!(ScanStatus::Pending.as_str(), "pending");
        assert_eq!(ScanStatus::Indexed.as_str(), "indexed");
        assert_eq!(ScanStatus::Failed.as_str(), "failed");
        assert_eq!(ScanStatus::parse("pending").unwrap(), ScanStatus::Pending);
        assert!(ScanStatus::parse("garbage").is_err());
    }

    #[test]
    fn thumbnail_status_roundtrip() {
        assert_eq!(ThumbnailStatus::Pending.as_str(), "pending");
        assert_eq!(ThumbnailStatus::parse("generated").unwrap(), ThumbnailStatus::Generated);
    }
}