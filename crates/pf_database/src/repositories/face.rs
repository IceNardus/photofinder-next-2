//! Face repository。
//!
//! Phase 1 (plan §3, §5) schema 升级：
//! - NewFace / FaceRow 增加 detector_model、alignment_version、embedding_model、cluster_score、cluster_method
//! - vector_id 改为可空（pending 状态先写 face row,后回填 vector_id）
//! - insert_embedding 接受 model_name / model_version / dimension / vector_id / normalized
//! - get_embedding / list_unassigned_embeddings 按调用方传入的 model_name 精确匹配
//!   （实际存的 model_name 为 `FACE_MODEL_NAME` = "arcface-w600k-r50"，见 pf_core）
//!
//! Phase 2 (plan §11)：face 索引状态机
//! - NewFace.status 默认为 Pending(insert 时即写 pending + vector_id=NULL)
//! - FaceRow 加 status / indexed_at / error_message / last_attempt_at
//! - mark_indexed / mark_failed / list_by_status / count_by_status

use chrono::{DateTime, Utc};
use pf_core::BBox;
use rusqlite::params;

use crate::error::DatabaseError;
use crate::Transaction;

/// face 索引状态机(plan §11)。
///
/// - Pending: 已写入 DB 但还没写 HNSW(vector_id=NULL)
/// - Indexed: HNSW + DB 一致,有 vector_id 与 embedding
/// - Failed:  HNSW 写入失败,DB 留 row + error_message 用于诊断
///
/// 014 后:face 同时有 `status`(旧,保留兼容)与 `index_status`(新,权威)
/// 两列语义一致,Phase 2+ 写 index_status。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaceStatus {
    Pending,
    Indexed,
    Failed,
}

impl FaceStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            FaceStatus::Pending => "pending",
            FaceStatus::Indexed => "indexed",
            FaceStatus::Failed => "failed",
        }
    }

    pub fn from_str(s: &str) -> Self {
        match s {
            "indexed" => FaceStatus::Indexed,
            "failed" => FaceStatus::Failed,
            _ => FaceStatus::Pending,
        }
    }
}

impl Default for FaceStatus {
    fn default() -> Self {
        FaceStatus::Pending
    }
}

/// `index_status` 字段别名 — 与 `FaceStatus` 相同枚举,语义上一致;
/// 但字段独立(启动时与 `status` 同值,Phase 2+ 由 mark_index_status 写入)。
pub type FaceIndexStatus = FaceStatus;

/// 新插入的人脸（不含 id）。
///
/// Phase 2: `status` 字段在 insert 时即设 `Pending`,vector_id 必为 None;
/// `mark_indexed` / `mark_failed` 负责后续状态转换。
#[derive(Debug, Clone)]
pub struct NewFace {
    /// 所属图片
    pub image_id: i64,
    /// bbox
    pub bbox: BBox,
    /// detector 置信度
    pub detector_score: f32,
    /// detector 模型标识
    pub detector_model: String,
    /// 关键点 JSON（5 点）
    pub keypoints_json: Option<String>,
    /// yaw / pitch / roll（可空,Phase 1 写 NULL）
    pub yaw_pitch_roll: Option<(f32, f32, f32)>,
    /// 综合质量分
    pub quality: f32,
    /// blur score（Laplacian 方差归一化）
    pub blur_score: Option<f32>,
    /// pose score（关键点对称性）
    pub pose_score: Option<f32>,
    /// face area score（归一化最小边长）
    pub face_area_score: Option<f32>,
    /// 对齐版本（`ALIGNMENT_VERSION`）
    pub alignment_version: Option<String>,
    /// embedding 模型名（"arcface-w600k-r50"，与 `FACE_MODEL_NAME` 一致）
    pub embedding_model: Option<String>,
    /// 模型版本（"arcface-w600k-r50@v1.1.0"）
    pub model_version: String,
    /// pf_vector 中的 id。Phase 2 默认 None（pending 状态）,
    /// `mark_indexed` 时回填。
    pub vector_id: Option<i64>,
    /// HNSW 内部 handle — 与 vector_id 语义相同,但 014 后是 hnsw 内部 node_id 的最新已知值。
    /// 与 vector_id 并存保证旧代码不破。
    pub hnsw_handle: Option<i64>,
    /// 初始索引状态。Phase 2 默认 Pending。
    pub status: FaceStatus,
    /// 索引世代号。Phase 0: 默认 0 (迁移回填)。reindex 时 += 1。
    pub index_generation: i64,
    /// 聚类得分（Phase 4 填）
    pub cluster_score: Option<f32>,
    /// 聚类方法（Phase 4 填）
    pub cluster_method: Option<String>,
}

/// 数据库行 → FaceRow。
#[derive(Debug, Clone)]
pub struct FaceRow {
    /// id
    pub id: i64,
    /// image_id
    pub image_id: i64,
    /// person_id（None = 尚未聚类）
    pub person_id: Option<i64>,
    /// bbox
    pub bbox: BBox,
    /// detector 置信度
    pub detector_score: f32,
    /// detector 模型
    pub detector_model: String,
    /// 关键点 JSON
    pub keypoints_json: Option<String>,
    /// yaw
    pub yaw: Option<f32>,
    /// pitch
    pub pitch: Option<f32>,
    /// roll
    pub roll: Option<f32>,
    /// 综合质量
    pub quality: f32,
    /// blur score
    pub blur_score: Option<f32>,
    /// pose score
    pub pose_score: Option<f32>,
    /// face area score
    pub face_area_score: Option<f32>,
    /// 对齐版本
    pub alignment_version: Option<String>,
    /// embedding 模型名
    pub embedding_model: Option<String>,
    /// 模型版本
    pub model_version: String,
    /// vector_id（pending 时为 None）
    pub vector_id: Option<i64>,
    /// HNSW 内部 handle（与 vector_id 等价,014 后权威）
    pub hnsw_handle: Option<i64>,
    /// 索引状态（Phase 2 — 旧字段,保留）
    pub status: FaceStatus,
    /// 当前权威索引状态（014+ — 与 status 同步起步,Phase 2 之后由 mark_index_status 写入）
    pub index_status: FaceIndexStatus,
    /// 索引世代号（reindex 时 += 1）
    pub index_generation: i64,
    /// 索引成功时间（Phase 2）
    pub indexed_at: Option<DateTime<Utc>>,
    /// 错误消息（Phase 2 failed 时填）
    pub error_message: Option<String>,
    /// 最近一次尝试时间（Phase 2）
    pub last_attempt_at: Option<DateTime<Utc>>,
    /// 聚类得分
    pub cluster_score: Option<f32>,
    /// 聚类方法
    pub cluster_method: Option<String>,
    /// 创建时间
    pub created_at: DateTime<Utc>,
}

/// Face repository。
pub struct FaceRepository<'tx, 'db> {
    pub(crate) tx: &'tx mut Transaction<'db>,
}

impl<'tx, 'db> FaceRepository<'tx, 'db> {
    /// 插入新 face。返回 id。
    ///
    /// Phase 2: status 默认 Pending;vector_id 在 insert 时通常为 None。
    /// 若调用方传 `Indexed` + Some(vector_id),等价于一次完成的状态(批量导入场景)。
    pub fn insert(&mut self, face: &NewFace) -> Result<i64, DatabaseError> {
        let (yaw, pitch, roll) = face
            .yaw_pitch_roll
            .map(|(y, p, r)| (Some(y), Some(p), Some(r)))
            .unwrap_or((None, None, None));
        // 双写 vector_id + hnsw_handle；status + index_status 都写
        self.tx.execute(
            "INSERT INTO faces
             (image_id, bbox_x, bbox_y, bbox_w, bbox_h,
              detector_score, detector_model, keypoints_json,
              yaw, pitch, roll,
              quality_score, blur_score, pose_score, face_area_score,
              alignment_version, embedding_model, model_version,
              vector_id, hnsw_handle, status, index_status, last_attempt_at,
              index_generation, cluster_score, cluster_method)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21, ?22, ?23, ?24, ?25, ?26)",
            params![
                face.image_id,
                face.bbox.x,
                face.bbox.y,
                face.bbox.w,
                face.bbox.h,
                face.detector_score,
                face.detector_model,
                face.keypoints_json,
                yaw,
                pitch,
                roll,
                face.quality,
                face.blur_score,
                face.pose_score,
                face.face_area_score,
                face.alignment_version,
                face.embedding_model,
                face.model_version,
                face.vector_id,
                face.hnsw_handle,
                face.status.as_str(),
                // 014: index_status 默认 = status(初始 Pending);插入后由 mark_index_status 推动
                face.status.as_str(),
                chrono::Utc::now().to_rfc3339(),
                face.index_generation,
                face.cluster_score,
                face.cluster_method,
            ],
        )?;
        Ok(self.tx.last_insert_rowid())
    }

    /// 按 id 查询。
    pub fn get_by_id(&self, id: i64) -> Result<Option<FaceRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, image_id, person_id, bbox_x, bbox_y, bbox_w, bbox_h,
                    detector_score, detector_model, keypoints_json,
                    yaw, pitch, roll,
                    quality_score, blur_score, pose_score, face_area_score,
                    alignment_version, embedding_model, model_version,
                    vector_id, hnsw_handle, status, index_status,
                    indexed_at, error_message, last_attempt_at,
                    index_generation, cluster_score, cluster_method, created_at
             FROM faces WHERE id = ?1",
        )?;
        let mut rows = stmt.query(params![id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row_to_face(row)?))
        } else {
            Ok(None)
        }
    }

    /// 按 vector_id 查询（用于反查 search 结果）。
    pub fn get_by_vector_id(&self, vector_id: i64) -> Result<Option<FaceRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, image_id, person_id, bbox_x, bbox_y, bbox_w, bbox_h,
                    detector_score, detector_model, keypoints_json,
                    yaw, pitch, roll,
                    quality_score, blur_score, pose_score, face_area_score,
                    alignment_version, embedding_model, model_version,
                    vector_id, hnsw_handle, status, index_status,
                    indexed_at, error_message, last_attempt_at,
                    index_generation, cluster_score, cluster_method, created_at
             FROM faces WHERE vector_id = ?1 LIMIT 1",
        )?;
        let mut rows = stmt.query(params![vector_id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row_to_face(row)?))
        } else {
            Ok(None)
        }
    }

    /// 按 vector_id 批量查询（Audit B1/B3：替代逐 hit 单查事务）。
    ///
    /// 一次 `IN (...)` 子句查回多张人脸，避免搜索热路径上每个 HNSW hit 单独
    /// 开事务。SQLite 单语句参数上限默认 32766，按 500 分块保证任意规模安全。
    pub fn list_by_vector_ids(&self, vector_ids: &[i64]) -> Result<Vec<FaceRow>, DatabaseError> {
        const CHUNK: usize = 500;
        let mut out = Vec::with_capacity(vector_ids.len());
        for chunk in vector_ids.chunks(CHUNK) {
            let placeholders = chunk.iter().map(|_| "?").collect::<Vec<_>>().join(",");
            let sql = format!(
                "SELECT id, image_id, person_id, bbox_x, bbox_y, bbox_w, bbox_h,
                        detector_score, detector_model, keypoints_json,
                        yaw, pitch, roll,
                        quality_score, blur_score, pose_score, face_area_score,
                        alignment_version, embedding_model, model_version,
                        vector_id, hnsw_handle, status, index_status,
                        indexed_at, error_message, last_attempt_at,
                        index_generation, cluster_score, cluster_method, created_at
                 FROM faces WHERE vector_id IN ({placeholders})"
            );
            let params: Vec<&dyn rusqlite::ToSql> = chunk
                .iter()
                .map(|v| v as &dyn rusqlite::ToSql)
                .collect();
            let mut stmt = self.tx.prepare(&sql)?;
            let mut rows = stmt.query(params.as_slice())?;
            while let Some(row) = rows.next()? {
                out.push(row_to_face(row)?);
            }
        }
        Ok(out)
    }

    /// 按 hnsw_handle 查询（014 后权威反查路径）。
    pub fn get_by_hnsw_handle(&self, handle: i64) -> Result<Option<FaceRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, image_id, person_id, bbox_x, bbox_y, bbox_w, bbox_h,
                    detector_score, detector_model, keypoints_json,
                    yaw, pitch, roll,
                    quality_score, blur_score, pose_score, face_area_score,
                    alignment_version, embedding_model, model_version,
                    vector_id, hnsw_handle, status, index_status,
                    indexed_at, error_message, last_attempt_at,
                    index_generation, cluster_score, cluster_method, created_at
             FROM faces WHERE hnsw_handle = ?1 LIMIT 1",
        )?;
        let mut rows = stmt.query(params![handle])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row_to_face(row)?))
        } else {
            Ok(None)
        }
    }

    /// 列出某 image 的所有人脸。
    pub fn list_by_image(&self, image_id: i64) -> Result<Vec<FaceRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, image_id, person_id, bbox_x, bbox_y, bbox_w, bbox_h,
                    detector_score, detector_model, keypoints_json,
                    yaw, pitch, roll,
                    quality_score, blur_score, pose_score, face_area_score,
                    alignment_version, embedding_model, model_version,
                    vector_id, hnsw_handle, status, index_status,
                    indexed_at, error_message, last_attempt_at,
                    index_generation, cluster_score, cluster_method, created_at
             FROM faces WHERE image_id = ?1 ORDER BY id",
        )?;
        let mut rows = stmt.query(params![image_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_face(row)?);
        }
        Ok(out)
    }

    /// 列出某 person 的所有人脸。
    pub fn list_by_person(&self, person_id: i64) -> Result<Vec<FaceRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, image_id, person_id, bbox_x, bbox_y, bbox_w, bbox_h,
                    detector_score, detector_model, keypoints_json,
                    yaw, pitch, roll,
                    quality_score, blur_score, pose_score, face_area_score,
                    alignment_version, embedding_model, model_version,
                    vector_id, hnsw_handle, status, index_status,
                    indexed_at, error_message, last_attempt_at,
                    index_generation, cluster_score, cluster_method, created_at
             FROM faces WHERE person_id = ?1 ORDER BY quality_score DESC",
        )?;
        let mut rows = stmt.query(params![person_id])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_face(row)?);
        }
        Ok(out)
    }

    /// 列出所有人脸（不筛选 person_id，供 rebuild_face_index 使用）。
    pub fn list_all(&self) -> Result<Vec<FaceRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, image_id, person_id, bbox_x, bbox_y, bbox_w, bbox_h,
                    detector_score, detector_model, keypoints_json,
                    yaw, pitch, roll,
                    quality_score, blur_score, pose_score, face_area_score,
                    alignment_version, embedding_model, model_version,
                    vector_id, hnsw_handle, status, index_status,
                    indexed_at, error_message, last_attempt_at,
                    index_generation, cluster_score, cluster_method, created_at
             FROM faces ORDER BY id",
        )?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_face(row)?);
        }
        Ok(out)
    }

    /// 列出所有未聚类的人脸（person_id IS NULL）。
    pub fn list_unassigned(&self) -> Result<Vec<FaceRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, image_id, person_id, bbox_x, bbox_y, bbox_w, bbox_h,
                    detector_score, detector_model, keypoints_json,
                    yaw, pitch, roll,
                    quality_score, blur_score, pose_score, face_area_score,
                    alignment_version, embedding_model, model_version,
                    vector_id, hnsw_handle, status, index_status,
                    indexed_at, error_message, last_attempt_at,
                    index_generation, cluster_score, cluster_method, created_at
             FROM faces WHERE person_id IS NULL ORDER BY quality_score DESC",
        )?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_face(row)?);
        }
        Ok(out)
    }

    /// 014: 列出 index_status='pending' 的 faces(供 RebuildFaceIndexExecutor 补 HNSW)。
    /// 默认只补指定 generation 的,避免旧 rebuild 覆盖新 face。
    pub fn list_pending_for_generation(
        &self,
        generation: i64,
    ) -> Result<Vec<FaceRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, image_id, person_id, bbox_x, bbox_y, bbox_w, bbox_h,
                    detector_score, detector_model, keypoints_json,
                    yaw, pitch, roll,
                    quality_score, blur_score, pose_score, face_area_score,
                    alignment_version, embedding_model, model_version,
                    vector_id, hnsw_handle, status, index_status,
                    indexed_at, error_message, last_attempt_at,
                    index_generation, cluster_score, cluster_method, created_at
             FROM faces
             WHERE index_status = 'pending' AND index_generation = ?1
             ORDER BY id",
        )?;
        let mut rows = stmt.query(params![generation])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_face(row)?);
        }
        Ok(out)
    }

    /// 按 id 设置 person_id。
    pub fn set_person(&mut self, face_id: i64, person_id: Option<i64>) -> Result<(), DatabaseError> {
        self.tx.execute(
            "UPDATE faces SET person_id = ?1 WHERE id = ?2",
            params![person_id, face_id],
        )?;
        Ok(())
    }

    /// 统计某 person 的 face 数。
    pub fn count_by_person(&self, person_id: i64) -> Result<i64, DatabaseError> {
        let n: i64 = self.tx.query_row(
            "SELECT COUNT(*) FROM faces WHERE person_id = ?1",
            params![person_id],
            |r| r.get(0),
        )?;
        Ok(n)
    }

    /// 删除某 image 的所有 faces（用于重建索引）。
    pub fn delete_by_image(&mut self, image_id: i64) -> Result<u64, DatabaseError> {
        let n = self.tx.execute(
            "DELETE FROM faces WHERE image_id = ?1",
            params![image_id],
        )?;
        Ok(n as u64)
    }

    /// 统计总 face 数。
    pub fn count(&self) -> Result<i64, DatabaseError> {
        let n: i64 = self
            .tx
            .query_row("SELECT COUNT(*) FROM faces", [], |r| r.get(0))?;
        Ok(n)
    }

    /// Phase 1: 更新 face 的 vector_id（HNSW 写入后回填）。
    pub fn update_vector_id(
        &mut self,
        face_id: i64,
        vector_id: Option<i64>,
    ) -> Result<(), DatabaseError> {
        self.tx.execute(
            "UPDATE faces SET vector_id = ?1 WHERE id = ?2",
            params![vector_id, face_id],
        )?;
        Ok(())
    }

    /// Phase 1: 插入(覆盖)一张 face 的 embedding（小端 f32 BLOB）。
    ///
    /// Phase 1 schema：face_embeddings 拆 1:1 → 1:N。同一 face 对同一 (model_name, model_version)
    /// 组合只允许一条记录(`UNIQUE` 约束)。
    ///
    /// Phase 3: 不再需要 image_id(migration 012 后 face_embeddings 已去掉该列,
    /// image_id 通过 faces JOIN 反查)。
    pub fn insert_embedding(
        &mut self,
        face_id: i64,
        model_name: &str,
        model_version: &str,
        dimension: i64,
        vector: &[f32],
        vector_id: Option<i64>,
        normalized: bool,
    ) -> Result<i64, DatabaseError> {
        let bytes = vector_to_blob(vector);
        self.tx.execute(
            "INSERT OR REPLACE INTO face_embeddings
             (face_id, model_name, model_version, dimension, vector, vector_id, normalized)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            params![
                face_id,
                model_name,
                model_version,
                dimension,
                bytes,
                vector_id,
                if normalized { 1i64 } else { 0i64 },
            ],
        )?;
        Ok(self.tx.last_insert_rowid())
    }

    /// 取一张 face 的 embedding（按 model_name 过滤,默认 'arcface'）。
    pub fn get_embedding(
        &self,
        face_id: i64,
        model_name: &str,
    ) -> Result<Option<Vec<f32>>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT dimension, vector FROM face_embeddings
             WHERE face_id = ?1 AND model_name = ?2 LIMIT 1",
        )?;
        let mut rows = stmt.query(params![face_id, model_name])?;
        if let Some(row) = rows.next()? {
            let dim: i64 = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            Ok(Some(blob_to_vector(&blob, dim as usize)?))
        } else {
            Ok(None)
        }
    }

    /// 列出所有未聚类的 face 的 embedding（按 quality_score DESC,默认 model_name='arcface'）。
    pub fn list_unassigned_embeddings(
        &self,
        model_name: &str,
    ) -> Result<Vec<(i64, i64, Vec<f32>)>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT f.id, f.image_id, e.dimension, e.vector
             FROM faces f
             JOIN face_embeddings e ON e.face_id = f.id
             WHERE f.person_id IS NULL AND e.model_name = ?1
             ORDER BY f.quality_score DESC",
        )?;
        let mut rows = stmt.query(params![model_name])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            let face_id: i64 = row.get(0)?;
            let image_id: i64 = row.get(1)?;
            let dim: i64 = row.get(2)?;
            let blob: Vec<u8> = row.get(3)?;
            let vec = blob_to_vector(&blob, dim as usize)?;
            out.push((face_id, image_id, vec));
        }
        Ok(out)
    }

    /// 删除某 image 的所有 embeddings(通过 JOIN faces 反查)。
    pub fn delete_embeddings_by_image(&mut self, image_id: i64) -> Result<u64, DatabaseError> {
        let n = self.tx.execute(
            "DELETE FROM face_embeddings WHERE face_id IN (SELECT id FROM faces WHERE image_id = ?1)",
            params![image_id],
        )?;
        Ok(n as u64)
    }

    /// 列出所有 embeddings（按 face_id 顺序,默认 model_name='arcface'）。Phase 2 rebuild HNSW 用。
    pub fn list_embeddings(
        &self,
        model_name: &str,
    ) -> Result<Vec<(i64, i64, Vec<f32>)>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT face_id, image_id, dimension, vector FROM face_embeddings
             WHERE model_name = ?1 ORDER BY face_id",
        )?;
        let mut rows = stmt.query(params![model_name])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            let face_id: i64 = row.get(0)?;
            let image_id: i64 = row.get(1)?;
            let dim: i64 = row.get(2)?;
            let blob: Vec<u8> = row.get(3)?;
            let vec = blob_to_vector(&blob, dim as usize)?;
            out.push((face_id, image_id, vec));
        }
        Ok(out)
    }

    // ========================================================================
    // Phase 2: face 索引状态机
    // ========================================================================

    /// Phase 2: 把 face 标记为 Indexed,回填 vector_id + hnsw_handle。
    ///
    /// 同时记录 indexed_at 与 last_attempt_at。
    /// 不会修改 embedding row(由调用方在事务内 insert_embedding)。
    pub fn mark_indexed(
        &mut self,
        face_id: i64,
        vector_id: i64,
    ) -> Result<(), DatabaseError> {
        self.tx.execute(
            "UPDATE faces SET
                status = ?1,
                index_status = ?1,
                vector_id = ?2,
                hnsw_handle = ?2,
                indexed_at = strftime('%Y-%m-%dT%H:%M:%fZ','now'),
                last_attempt_at = strftime('%Y-%m-%dT%H:%M:%fZ','now'),
                error_message = NULL
             WHERE id = ?3",
            params![FaceStatus::Indexed.as_str(), vector_id, face_id],
        )?;
        Ok(())
    }

    /// Phase 2: 把 face 标记为 Failed,记录错误信息。
    pub fn mark_failed(
        &mut self,
        face_id: i64,
        err: &str,
    ) -> Result<(), DatabaseError> {
        self.tx.execute(
            "UPDATE faces SET
                status = ?1,
                index_status = ?1,
                last_attempt_at = strftime('%Y-%m-%dT%H:%M:%fZ','now'),
                error_message = ?2
             WHERE id = ?3",
            params![FaceStatus::Failed.as_str(), err, face_id],
        )?;
        Ok(())
    }

    /// Phase 2: 通用状态变更(主要为 debug/测试用)。
    pub fn set_status(
        &mut self,
        face_id: i64,
        status: FaceStatus,
        err: Option<&str>,
    ) -> Result<(), DatabaseError> {
        self.tx.execute(
            "UPDATE faces SET
                status = ?1,
                index_status = ?1,
                last_attempt_at = strftime('%Y-%m-%dT%H:%M:%fZ','now'),
                error_message = ?2
             WHERE id = ?3",
            params![status.as_str(), err, face_id],
        )?;
        Ok(())
    }

    /// 014: 标记 face 进入指定 index_status + index_generation。
    /// 主要给 RebuildFaceIndexExecutor 用:从 pending → indexed(成功)或 failed。
    /// 同时同步 status 列(冗余字段,保留向后兼容)。
    pub fn mark_index_status(
        &mut self,
        face_id: i64,
        status: FaceIndexStatus,
        generation: i64,
    ) -> Result<(), DatabaseError> {
        self.tx.execute(
            "UPDATE faces SET
                index_status = ?1,
                index_generation = ?2,
                status = ?1,
                last_attempt_at = strftime('%Y-%m-%dT%H:%M:%fZ','now'),
                error_message = NULL
             WHERE id = ?3",
            params![status.as_str(), generation, face_id],
        )?;
        Ok(())
    }

    /// 014: 把 hnsw_handle 回填到 faces(成功 insert HNSW 后)。
    /// 同时保持 vector_id 同步,旧代码读 vector_id 不报错。
    pub fn set_hnsw_handle(&mut self, face_id: i64, handle: i64) -> Result<(), DatabaseError> {
        self.tx.execute(
            "UPDATE faces SET hnsw_handle = ?1, vector_id = ?1 WHERE id = ?2",
            params![handle, face_id],
        )?;
        Ok(())
    }

    /// Phase 2: 列出指定 status 的 faces(诊断 / 重建用)。
    pub fn list_by_status(
        &self,
        status: FaceStatus,
    ) -> Result<Vec<FaceRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT id, image_id, person_id, bbox_x, bbox_y, bbox_w, bbox_h,
                    detector_score, detector_model, keypoints_json,
                    yaw, pitch, roll,
                    quality_score, blur_score, pose_score, face_area_score,
                    alignment_version, embedding_model, model_version,
                    vector_id, hnsw_handle, status, index_status,
                    indexed_at, error_message, last_attempt_at,
                    index_generation, cluster_score, cluster_method, created_at
             FROM faces WHERE status = ?1 ORDER BY id",
        )?;
        let mut rows = stmt.query(params![status.as_str()])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_face(row)?);
        }
        Ok(out)
    }

    /// Phase 2: 按 status 统计(诊断面板)。
    pub fn count_by_status(&self) -> Result<Vec<(FaceStatus, i64)>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            "SELECT status, COUNT(*) FROM faces GROUP BY status",
        )?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            let s: String = row.get(0)?;
            let n: i64 = row.get(1)?;
            out.push((FaceStatus::from_str(&s), n));
        }
        Ok(out)
    }

    /// Phase 1.3: 列出 `face_embeddings` 表里所有出现过的 `model_version`。
    ///
    /// 设计意图(migration 014 §5):多版本同时存在意味着 HNSW index 与
    /// 真实 embedding 不一致(向量维度不同 / 模型语义不同),需要 reindex。
    ///
    /// 返回的版本按字典序排序;空表返回空 Vec。版本号为 NULL 也会被列出
    /// (用字符串 `"<NULL>"` 表示,便于诊断)。
    pub fn distinct_model_versions(&mut self) -> Result<Vec<String>, DatabaseError> {
        let mut stmt = self
            .tx
            .prepare("SELECT DISTINCT model_version FROM face_embeddings")?;
        let mut rows = stmt.query([])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            let v: Option<String> = row.get(0)?;
            out.push(v.unwrap_or_else(|| "<NULL>".to_string()));
        }
        out.sort();
        Ok(out)
    }
}

/// f32 → 小端 BLOB（用于 BLOB 存储）。
fn vector_to_blob(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// BLOB → f32 vector（要求长度匹配）。
fn blob_to_vector(blob: &[u8], dim: usize) -> Result<Vec<f32>, DatabaseError> {
    if blob.len() != dim * 4 {
        return Err(DatabaseError::Conversion(format!(
            "embedding blob size mismatch: {} bytes for dim={}",
            blob.len(),
            dim
        )));
    }
    let mut out = Vec::with_capacity(dim);
    for chunk in blob.chunks_exact(4) {
        let bytes: [u8; 4] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        out.push(f32::from_le_bytes(bytes));
    }
    Ok(out)
}

fn row_to_face(row: &rusqlite::Row<'_>) -> Result<FaceRow, DatabaseError> {
    // SELECT 列序(014 后,共 31 列):
    //   0:id 1:image_id 2:person_id 3..6:bbox 7:detector_score 8:detector_model
    //   9:keypoints_json 10..12:yaw/pitch/roll 13..16:quality/blur/pose/area
    //   17:alignment_version 18:embedding_model 19:model_version
    //   20:vector_id 21:hnsw_handle 22:status 23:index_status
    //   24:indexed_at 25:error_message 26:last_attempt_at
    //   27:index_generation 28:cluster_score 29:cluster_method 30:created_at
    let created_str: String = row.get(30)?;
    let created_at = DateTime::parse_from_rfc3339(&created_str)
        .map(|dt| dt.with_timezone(&Utc))
        .map_err(|e| DatabaseError::Conversion(format!("created_at parse: {e}")))?;
    let status_str: String = row.get(22)?;
    let index_status_str: String = row.get(23)?;
    let indexed_at = match row.get::<_, Option<String>>(24)? {
        Some(s) => Some(
            DateTime::parse_from_rfc3339(&s)
                .map(|dt| dt.with_timezone(&Utc))
                .map_err(|e| DatabaseError::Conversion(format!("indexed_at parse: {e}")))?,
        ),
        None => None,
    };
    let last_attempt_at = match row.get::<_, Option<String>>(26)? {
        Some(s) => Some(
            DateTime::parse_from_rfc3339(&s)
                .map(|dt| dt.with_timezone(&Utc))
                .map_err(|e| DatabaseError::Conversion(format!("last_attempt_at parse: {e}")))?,
        ),
        None => None,
    };
    Ok(FaceRow {
        id: row.get(0)?,
        image_id: row.get(1)?,
        person_id: row.get(2)?,
        bbox: BBox {
            x: row.get(3)?,
            y: row.get(4)?,
            w: row.get(5)?,
            h: row.get(6)?,
        },
        detector_score: row.get(7)?,
        detector_model: row.get(8)?,
        keypoints_json: row.get(9)?,
        yaw: row.get(10)?,
        pitch: row.get(11)?,
        roll: row.get(12)?,
        quality: row.get(13)?,
        blur_score: row.get(14)?,
        pose_score: row.get(15)?,
        face_area_score: row.get(16)?,
        alignment_version: row.get(17)?,
        embedding_model: row.get(18)?,
        model_version: row.get(19)?,
        vector_id: row.get(20)?,
        hnsw_handle: row.get(21)?,
        status: FaceStatus::from_str(&status_str),
        index_status: FaceStatus::from_str(&index_status_str),
        indexed_at,
        error_message: row.get(25)?,
        last_attempt_at,
        index_generation: row.get(27)?,
        cluster_score: row.get(28)?,
        cluster_method: row.get(29)?,
        created_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blob_roundtrip_unit_vector() {
        let v = vec![0.6f32, 0.8];
        let blob = vector_to_blob(&v);
        assert_eq!(blob.len(), 8);
        let back = blob_to_vector(&blob, 2).unwrap();
        assert!((back[0] - 0.6).abs() < 1e-6);
        assert!((back[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn blob_mismatch_size_errors() {
        let blob = vec![0u8; 7];
        let err = blob_to_vector(&blob, 2).unwrap_err();
        assert!(matches!(err, DatabaseError::Conversion(_)));
    }

    #[test]
    fn face_status_roundtrip() {
        for s in [FaceStatus::Pending, FaceStatus::Indexed, FaceStatus::Failed] {
            assert_eq!(FaceStatus::from_str(s.as_str()), s);
        }
        assert_eq!(FaceStatus::from_str("nonsense"), FaceStatus::Pending);
        assert_eq!(FaceStatus::default(), FaceStatus::Pending);
    }
}
