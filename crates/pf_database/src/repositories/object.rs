//! Object repository。

use chrono::{DateTime, Utc};
use pf_core::BBox;
use rusqlite::params;

use crate::error::DatabaseError;
use crate::Transaction;

/// ROI 来源类型（Phase 2 引入）。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RoiType {
    /// 整图（全图 ROI）
    FullImage,
    /// Multi-scale sliding window
    SlidingWindow,
}

impl RoiType {
    pub fn as_str(&self) -> &'static str {
        match self {
            RoiType::FullImage => "FullImage",
            RoiType::SlidingWindow => "SlidingWindow",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "SlidingWindow" => RoiType::SlidingWindow,
            _ => RoiType::FullImage,
        }
    }
}

/// 新插入的对象。
#[derive(Debug, Clone)]
pub struct NewObject {
    /// 所属图片
    pub image_id: i64,
    /// 类别 ID
    pub class_id: i32,
    /// 类别名
    pub class_name: String,
    /// 置信度
    pub confidence: f32,
    /// bbox
    pub bbox: BBox,
    /// 模型版本
    pub model_version: String,
    /// pf_vector 中的 id
    pub vector_id: i64,
    /// Phase 2 新增：ROI 所在 scale（用于 rank 调权）
    pub roi_scale: f32,
    /// Phase 2 新增：ROI 来源类型
    pub roi_type: RoiType,
}

/// 数据库行 → ObjectRow。
#[derive(Debug, Clone)]
pub struct ObjectRow {
    /// id
    pub id: i64,
    /// image_id
    pub image_id: i64,
    /// 类别 ID
    pub class_id: i32,
    /// 类别名
    pub class_name: String,
    /// 置信度
    pub confidence: f32,
    /// bbox
    pub bbox: BBox,
    /// 模型版本
    pub model_version: String,
    /// vector_id
    pub vector_id: i64,
    /// Phase 2 新增：ROI 所在 scale
    pub roi_scale: f32,
    /// Phase 2 新增：ROI 来源类型
    pub roi_type: RoiType,
    /// 创建时间
    pub created_at: DateTime<Utc>,
}

/// Object repository。
pub struct ObjectRepository<'tx, 'db> {
    pub(crate) tx: &'tx mut Transaction<'db>,
}

impl<'tx, 'db> ObjectRepository<'tx, 'db> {
    /// 插入新对象。
    pub fn insert(&mut self, obj: &NewObject) -> Result<i64, DatabaseError> {
        self.tx.execute(
            "INSERT INTO objects
             (image_id, class_id, class_name, confidence,
              bbox_x, bbox_y, bbox_w, bbox_h,
              model_version, vector_id, roi_scale, roi_type)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
            params![
                obj.image_id,
                obj.class_id,
                obj.class_name,
                obj.confidence,
                obj.bbox.x,
                obj.bbox.y,
                obj.bbox.w,
                obj.bbox.h,
                obj.model_version,
                obj.vector_id,
                obj.roi_scale,
                obj.roi_type.as_str(),
            ],
        )?;
        Ok(self.tx.last_insert_rowid())
    }

    /// 按 id 查询。
    pub fn get_by_id(&self, id: i64) -> Result<Option<ObjectRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, image_id, class_id, class_name, confidence,
                    bbox_x, bbox_y, bbox_w, bbox_h,
                    model_version, vector_id, roi_scale, roi_type, created_at
             FROM objects WHERE id = ?1",
        )?;
        let mut rows = stmt.query(params![id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row_to_object(row)?))
        } else {
            Ok(None)
        }
    }

    /// 按 id 批量查询（Audit B2：替代逐 candidate 单查事务）。
    ///
    /// 一次 `IN (...)` 子句查回多个对象。SQLite 单语句参数上限默认 32766，按
    /// 500 分块保证任意规模安全。
    pub fn list_by_ids(&self, ids: &[i64]) -> Result<Vec<ObjectRow>, DatabaseError> {
        const CHUNK: usize = 500;
        let mut out = Vec::with_capacity(ids.len());
        for chunk in ids.chunks(CHUNK) {
            let placeholders = chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!(
                "SELECT id, image_id, class_id, class_name, confidence,
                        bbox_x, bbox_y, bbox_w, bbox_h,
                        model_version, vector_id, roi_scale, roi_type, created_at
                 FROM objects WHERE id IN ({placeholders})"
            );
            let params: Vec<&dyn rusqlite::ToSql> = chunk
                .iter()
                .map(|v| v as &dyn rusqlite::ToSql)
                .collect();
            let mut stmt = self.tx.prepare(&sql)?;
            let mut rows = stmt.query(params.as_slice())?;
            while let Some(row) = rows.next()? {
                out.push(row_to_object(row)?);
            }
        }
        Ok(out)
    }

    /// 按 vector_id 查询。
    pub fn get_by_vector_id(&self, vector_id: i64) -> Result<Option<ObjectRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, image_id, class_id, class_name, confidence,
                    bbox_x, bbox_y, bbox_w, bbox_h,
                    model_version, vector_id, roi_scale, roi_type, created_at
             FROM objects WHERE vector_id = ?1 LIMIT 1",
        )?;
        let mut rows = stmt.query(params![vector_id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row_to_object(row)?))
        } else {
            Ok(None)
        }
    }

    /// 按 vector_id 批量查询（Audit B1：替代逐 hit 单查事务）。
    ///
    /// 一次 `IN (...)` 子句查回多个对象，避免搜索热路径上每个 HNSW hit 单独
    /// 开事务（fetch_k=200 时 = 200 次 BEGIN/COMMIT）。SQLite 单语句参数上限
    /// 默认 32766，这里按 500 分块，任意输入规模都安全。
    pub fn list_by_vector_ids(
        &self,
        vector_ids: &[i64],
    ) -> Result<Vec<ObjectRow>, DatabaseError> {
        const CHUNK: usize = 500;
        let mut out = Vec::with_capacity(vector_ids.len());
        for chunk in vector_ids.chunks(CHUNK) {
            let placeholders = chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!(
                "SELECT id, image_id, class_id, class_name, confidence,
                        bbox_x, bbox_y, bbox_w, bbox_h,
                        model_version, vector_id, roi_scale, roi_type, created_at
                 FROM objects WHERE vector_id IN ({placeholders})"
            );
            let params: Vec<&dyn rusqlite::ToSql> = chunk
                .iter()
                .map(|v| v as &dyn rusqlite::ToSql)
                .collect();
            let mut stmt = self.tx.prepare(&sql)?;
            let mut rows = stmt.query(params.as_slice())?;
            while let Some(row) = rows.next()? {
                out.push(row_to_object(row)?);
            }
        }
        Ok(out)
    }

    /// 按 image_id 列出所有 vector_id（Phase 2：用于 remove_image 清理 HNSW）。
    pub fn list_vector_ids_by_image(&self, image_id: i64) -> Result<Vec<i64>, DatabaseError> {
        let mut stmt = self
            .tx
            .prepare("SELECT vector_id FROM objects WHERE image_id = ?1")?;
        let mut rows = stmt.query(params![image_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row.get(0)?);
        }
        Ok(out)
    }

    /// 按类别 ID 列出。
    pub fn list_by_class(&self, class_id: i32, limit: usize) -> Result<Vec<ObjectRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, image_id, class_id, class_name, confidence,
                    bbox_x, bbox_y, bbox_w, bbox_h,
                    model_version, vector_id, roi_scale, roi_type, created_at
             FROM objects WHERE class_id = ?1 ORDER BY confidence DESC LIMIT ?2",
        )?;
        let mut rows = stmt.query(params![class_id, limit as i64])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_object(row)?);
        }
        Ok(out)
    }

    /// 列出某 image 的所有对象。
    pub fn list_by_image(&self, image_id: i64) -> Result<Vec<ObjectRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, image_id, class_id, class_name, confidence,
                    bbox_x, bbox_y, bbox_w, bbox_h,
                    model_version, vector_id, roi_scale, roi_type, created_at
             FROM objects WHERE image_id = ?1 ORDER BY confidence DESC",
        )?;
        let mut rows = stmt.query(params![image_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_object(row)?);
        }
        Ok(out)
    }

    /// 删除某 image 的所有对象（rows 返回删除的 vector_id 列表，供 remove_image 清理 HNSW）。
    pub fn delete_by_image(&mut self, image_id: i64) -> Result<Vec<i64>, DatabaseError> {
        let vector_ids = self.list_vector_ids_by_image(image_id)?;
        self.tx.execute(
            "DELETE FROM objects WHERE image_id = ?1",
            params![image_id],
        )?;
        Ok(vector_ids)
    }

    /// 统计总对象数。
    pub fn count(&self) -> Result<i64, DatabaseError> {
        let n: i64 = self
            .tx
            .query_row("SELECT COUNT(*) FROM objects", [], |r| r.get(0))?;
        Ok(n)
    }
}

fn row_to_object(row: &rusqlite::Row<'_>) -> Result<ObjectRow, DatabaseError> {
    let created_str: String = row.get(13)?;
    let created_at = DateTime::parse_from_rfc3339(&created_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| DatabaseError::Conversion(format!("created_at parse: {e}")))?;
    let roi_type_str: String = row.get(12)?;
    Ok(ObjectRow {
        id: row.get(0)?,
        image_id: row.get(1)?,
        class_id: row.get(2)?,
        class_name: row.get(3)?,
        confidence: row.get(4)?,
        bbox: BBox {
            x: row.get(5)?,
            y: row.get(6)?,
            w: row.get(7)?,
            h: row.get(8)?,
        },
        model_version: row.get(9)?,
        vector_id: row.get(10)?,
        roi_scale: row.get(11)?,
        roi_type: RoiType::from_str(&roi_type_str),
        created_at,
    })
}