//! IndexService — 单张图片人脸 / 对象索引。
//!
//! 流程（face）：
//! 1) 读 image bytes
//! 2) FacePipeline.process → Vec<FaceFeature>
//! 3) 每个 face: insert into vector index + insert into DB face row
//! 4) 更新 image.scan_status
//!
//! 流程（object）：图片 → extract_rois 滑动窗口 → MobileCLIP embed → HNSW 索引。

use std::path::Path;
use std::sync::Arc;

use pf_ai::object::roi::extract_rois;
use pf_ai::{BodyFeature, BodyPipeline, FaceFeature, FacePipeline, ImageData, ObjectEmbedder, PatchExtractor};
use pf_core::{BBox, Embedding, Face, Object, SearchResult};
use pf_database::{Database, FaceStatus, NewBody, NewFace, NewObject, RoiType, ScanStatus};
use pf_platform::{PhotoId, PhotoProvider};
use pf_task::{Priority, TaskKind, TaskScheduler};
use pf_vector::VectorIndex;
use rusqlite::params;
use tracing::{debug, info, warn};

use crate::error::ApplicationError;

/// 单张图片索引结果。
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct IndexSummary {
    /// image_id
    pub image_id: i64,
    /// 检测到的人脸数
    pub face_count: usize,
    /// 入向量库的 face 数（通过 quality）
    pub indexed_face_count: usize,
    /// 检测到的对象数
    pub object_count: usize,
    /// 入向量库的对象数
    pub indexed_object_count: usize,
    /// 检测到的 body 数
    pub body_count: usize,
    /// 入向量库的 body 数
    pub indexed_body_count: usize,
}

/// IndexService。
pub struct IndexService {
    db: Arc<Database>,
    face_pipeline: Arc<FacePipeline>,
    face_index: Arc<dyn VectorIndex>,
    body_pipeline: Option<Arc<BodyPipeline>>,
    body_index: Option<Arc<dyn VectorIndex>>,
    object_embedder: Option<Arc<dyn ObjectEmbedder>>,
    object_index: Option<Arc<dyn VectorIndex>>,
    photo_provider: Arc<dyn PhotoProvider>,
    /// SuperPoint patch extractor (for patch indexing)
    patch_extractor: Option<Arc<PatchExtractor>>,
    /// Patch VLAD vector index
    patch_index: Option<Arc<dyn VectorIndex>>,
}

impl IndexService {
    /// 构造（人脸 + 对象）。
    pub fn new(
        db: Arc<Database>,
        face_pipeline: Arc<FacePipeline>,
        face_index: Arc<dyn VectorIndex>,
        body_pipeline: Option<Arc<BodyPipeline>>,
        body_index: Option<Arc<dyn VectorIndex>>,
        object_embedder: Option<Arc<dyn ObjectEmbedder>>,
        object_index: Option<Arc<dyn VectorIndex>>,
        photo_provider: Arc<dyn PhotoProvider>,
        patch_extractor: Option<Arc<PatchExtractor>>,
        patch_index: Option<Arc<dyn VectorIndex>>,
    ) -> Self {
        Self {
            db,
            face_pipeline,
            face_index,
            body_pipeline,
            body_index,
            object_embedder,
            object_index,
            photo_provider,
            patch_extractor,
            patch_index,
        }
    }

    /// 仅人脸（无 body）。
    #[allow(dead_code)]
    pub fn face_only(
        db: Arc<Database>,
        face_pipeline: Arc<FacePipeline>,
        face_index: Arc<dyn VectorIndex>,
        photo_provider: Arc<dyn PhotoProvider>,
    ) -> Self {
        Self::new(
            db,
            face_pipeline,
            face_index,
            None,
            None,
            None,
            None,
            photo_provider,
            None,
            None,
        )
    }

    /// 是否配置了 patch_extractor（决定 reindex_all 是否入队 IndexPatch）。
    pub fn has_patch_extractor(&self) -> bool {
        self.patch_extractor.is_some() && self.patch_index.is_some()
    }

    /// 索引一张图片（人脸 + 对象）。
    pub async fn index_image(
        &self,
        image_id: i64,
        photo_id: &PhotoId,
    ) -> Result<IndexSummary, ApplicationError> {
        let bytes = self
            .photo_provider
            .get_image(photo_id)
            .await
            .map_err(ApplicationError::Platform)?;
        let img = ImageData::from_bytes(&bytes).map_err(ApplicationError::AI)?;
        let (w, h) = (img.width(), img.height());

        let mut summary = IndexSummary {
            image_id,
            ..Default::default()
        };

        // 1) Face pipeline
        let face_features = match self.face_pipeline.process(&img).await {
            Ok(features) => {
                summary.face_count = features.len();
                summary.indexed_face_count = self.index_faces(image_id, &features, w, h)?;
                Some(features)
            }
            Err(e) => {
                warn!(image_id, error = %e, "face pipeline failed");
                None
            }
        };

        // 2) Body pipeline — uses best face bbox for body crop
        if let (Some(bp), Some(bi)) = (&self.body_pipeline, &self.body_index) {
            if let Some(ref features) = face_features {
                if let Some(best_face) = features.first() {
                    match bp.process(&img, &best_face.detection.bbox, best_face.detection.score) {
                        Ok(body_feat) => {
                            match self.index_bodies(image_id, &[body_feat]).await {
                                Ok(count) => {
                                    summary.body_count = 1;
                                    summary.indexed_body_count = count;
                                }
                                Err(e) => {
                                    warn!(image_id, error = %e, "body indexing failed");
                                }
                            }
                        }
                        Err(e) => {
                            warn!(image_id, error = %e, "body pipeline failed");
                        }
                    }
                }
            }
        }

        // 3) Object — multi_scale_roi + MobileCLIP（全量区域索引，与 ai-next 一致）
        if let (Some(oe), Some(oi)) = (&self.object_embedder, &self.object_index) {
            match self.index_objects_roi(image_id, &img, oe.clone(), oi.clone()).await {
                Ok(count) => {
                    summary.object_count = count;
                    summary.indexed_object_count = count;
                }
                Err(e) => {
                    warn!(image_id, error = %e, "object ROI indexing failed");
                }
            }
        }

        // 3) 更新 image.scan_status = Indexed + 回写 face/object counts
        let face_cnt = summary.indexed_face_count as i64;
        let obj_cnt = summary.indexed_object_count as i64;
        let _ = self
            .db
            .transaction(|tx| {
                let mut repo = tx.images();
                repo.update_scan_status(image_id, ScanStatus::Indexed)?;
                repo.update_counts(image_id, face_cnt, obj_cnt)
            })
            .map_err(|e| warn!(image_id, error = %e, "update scan_status/counts failed"));

        info!(image_id, face_cnt, obj_cnt, "image indexed");
        Ok(summary)
    }

    /// 仅索引人脸（供 IndexFaceExecutor 调用，避免重复处理）
    pub async fn index_faces_for_image(
        &self,
        image_id: i64,
        photo_id: &PhotoId,
    ) -> Result<IndexSummary, ApplicationError> {
        let bytes = self
            .photo_provider
            .get_image(photo_id)
            .await
            .map_err(ApplicationError::Platform)?;
        let img = ImageData::from_bytes(&bytes).map_err(ApplicationError::AI)?;
        let (w, h) = (img.width(), img.height());

        let mut summary = IndexSummary {
            image_id,
            ..Default::default()
        };

        // Face pipeline only
        match self.face_pipeline.process(&img).await {
            Ok(features) => {
                debug!(image_id, detected = features.len(), "face detection returned features");
                summary.face_count = features.len();
                summary.indexed_face_count = self.index_faces(image_id, &features, w, h)?;
                debug!(image_id, indexed = summary.indexed_face_count, "faces indexed");
            }
            Err(e) => {
                warn!(image_id, error = %e, "face pipeline failed");
            }
        }

        // 更新 counts（保留现有 object_count）— 重试 SQLITE_BUSY 兜底
        let face_cnt = summary.indexed_face_count as i64;
        let _ = self
            .db
            .transaction_with_retry(|tx| {
                let mut repo = tx.images();
                repo.update_counts(image_id, face_cnt, -1) // -1 表示不更新 object_count
            })
            .map_err(|e| warn!(image_id, error = %e, "update face_counts failed"));

        info!(image_id, face_cnt, "indexed faces only");
        Ok(summary)
    }

    /// 仅索引对象（供 IndexObjectExecutor 调用，避免重复处理）
    pub async fn index_objects_for_image(
        &self,
        image_id: i64,
        photo_id: &PhotoId,
    ) -> Result<IndexSummary, ApplicationError> {
        let bytes = self
            .photo_provider
            .get_image(photo_id)
            .await
            .map_err(ApplicationError::Platform)?;
        let img = ImageData::from_bytes(&bytes).map_err(ApplicationError::AI)?;

        let mut summary = IndexSummary {
            image_id,
            ..Default::default()
        };

        // Object — multi_scale_roi + MobileCLIP
        if let (Some(oe), Some(oi)) = (&self.object_embedder, &self.object_index) {
            match self.index_objects_roi(image_id, &img, oe.clone(), oi.clone()).await {
                Ok(count) => {
                    summary.object_count = count;
                    summary.indexed_object_count = count;
                }
                Err(e) => {
                    warn!(image_id, error = %e, "object ROI indexing failed");
                }
            }
        }

        // 更新 counts（保留现有 face_count）
        let obj_cnt = summary.indexed_object_count as i64;
        let _ = self
            .db
            .transaction(|tx| {
                let mut repo = tx.images();
                repo.update_object_count_only(image_id, obj_cnt)
            })
            .map_err(|e| warn!(image_id, error = %e, "update object_count failed"));

        info!(image_id, obj_cnt, "indexed objects only");
        Ok(summary)
    }

    /// 插入 faces 到向量库 + DB（批量化：一张图片的所有 face 在单个事务中写入）。
    ///
    /// 状态机(plan §11 Phase 2):
    /// 1. 准备阶段:计算每个 feature 的 vector_id,过滤已 indexed 的旧 face
    /// 2. Phase A(单事务):插入所有 face row,status=Pending,vector_id=NULL
    /// 3. Phase B(不在事务):逐个写 HNSW,失败则记入 failures 列表
    /// 4. Phase C(单事务):成功的 mark_indexed + 写 embedding;失败的 mark_failed
    ///
    /// 不变量:
    /// - Indexed face 一定有 embedding row + vector_id 不为 NULL
    /// - Failed face 一定有 error_message,vector_id 为 NULL
    /// - Pending face 应只在 Phase A→C 之间存在,持久化后立刻被 mark 成终态
    ///
    /// 故障语义:hnsw_rs 不支持元素级 remove。若 Phase C 失败,HNSW 中可能有
    /// 成功条目对应的 vector_id,但 DB 中 face 仍是 Pending。下次
    /// `rebuild_face_index` 会从 DB 状态重建,清掉残留。
    fn index_faces(
        &self,
        image_id: i64,
        features: &[FaceFeature],
        _img_w: u32,
        _img_h: u32,
    ) -> Result<usize, ApplicationError> {
        if features.is_empty() {
            return Ok(0);
        }

        // 步骤 1: 准备阶段 — 计算 vector_id,过滤已 indexed,构造 NewFace(Pending)
        let mut prepared: Vec<(NewFace, i64, Embedding)> = Vec::with_capacity(features.len());
        for feat in features {
            let emb: &Embedding = &feat.embedding;
            let vector_id = derive_face_vector_id(image_id, feat);

            // 去重:已存在 status=Indexed 的同 vector_id face,跳过
            let already_indexed = self
                .db
                .transaction(|tx| tx.faces().get_by_vector_id(vector_id))
                .ok()
                .flatten()
                .map(|row| row.status == FaceStatus::Indexed)
                .unwrap_or(false);
            if already_indexed {
                debug!(vector_id, "face already indexed, skipping");
                continue;
            }

            let det = &feat.detection;
            let keypoints_json = serde_json::to_string(&det.keypoints).ok();

            let new_face = NewFace {
                image_id,
                bbox: det.bbox,
                detector_score: det.score,
                detector_model: "scrfd-500m-bnkps".to_string(),
                keypoints_json,
                yaw_pitch_roll: feat.yaw_pitch_roll,
                quality: crate::quality_bridge(det, feat),
                blur_score: Some(feat.blur_score),
                pose_score: Some(feat.pose_score),
                face_area_score: Some(feat.face_area_score),
                alignment_version: Some(pf_ai::face::ALIGNMENT_VERSION.to_string()),
                embedding_model: Some(emb.model.name().to_string()),
                model_version: emb.model.as_str().to_string(),
                vector_id: None, // Phase 2: 先写 NULL,mark_indexed 时回填
                hnsw_handle: None, // 014: 与 vector_id 同步,mark_indexed 时回填
                status: FaceStatus::Pending,
                index_generation: crate::index_generation::current(),
                cluster_score: None,
                cluster_method: None,
            };

            prepared.push((new_face, vector_id, emb.clone()));
        }

        if prepared.is_empty() {
            return Ok(0);
        }

        // 步骤 2: Phase A — 单事务插入所有 face row 为 Pending
        let face_ids: Vec<i64> = self
            .db
            .transaction_with_retry(|tx| {
                let mut ids = Vec::with_capacity(prepared.len());
                for (new_face, _, _) in &prepared {
                    ids.push(tx.faces().insert(new_face)?);
                }
                Ok::<_, pf_database::DatabaseError>(ids)
            })
            .map_err(ApplicationError::Database)?;

        // 步骤 3: Phase B — 逐个写 HNSW(不在事务中)
        let mut successes: Vec<(i64, i64, Embedding)> = Vec::new();
        let mut failures: Vec<(i64, String)> = Vec::new();
        for ((_, vector_id, emb), face_id) in prepared.iter().zip(face_ids.iter()) {
            match self.face_index.insert(*vector_id, emb.as_slice()) {
                Ok(()) => successes.push((*face_id, *vector_id, emb.clone())),
                Err(e) => failures.push((*face_id, format!("hnsw insert: {e}"))),
            }
        }

        // 步骤 4: Phase C — 单事务批量 mark_indexed + insert_embedding,失败 mark_failed
        let indexed = self
            .db
            .transaction_with_retry(|tx| {
                for (face_id, vector_id, emb) in &successes {
                    tx.faces().mark_indexed(*face_id, *vector_id)?;
                    tx.faces().insert_embedding(
                        *face_id,
                        emb.model.name(),
                        emb.model.version(),
                        emb.dim as i64,
                        emb.as_slice(),
                        Some(*vector_id),
                        true,
                    )?;
                }
                for (face_id, err) in &failures {
                    tx.faces().mark_failed(*face_id, err)?;
                }
                Ok::<_, pf_database::DatabaseError>(successes.len())
            })
            .map_err(|e| {
                warn!(
                    image_id,
                    error = %e,
                    "Phase C mark_indexed/mark_failed DB failed; HNSW entries may be stale \
                  (will be fixed by rebuild_face_index)"
                );
                ApplicationError::Database(e)
            })?;

        // 立即持久化 HNSW(避免进程崩溃丢失)
        if !successes.is_empty() {
            if let Err(e) = self.face_index.save() {
                warn!(
                    error = %e,
                    "face_index save failed after indexing {} faces (rebuild will correct)",
                    indexed
                );
            }
        }

        if !failures.is_empty() {
            warn!(
                image_id,
                indexed,
                failed = failures.len(),
                "some faces failed HNSW insert; marked status=failed in DB"
            );
        }

        Ok(indexed)
    }

    /// 插入 bodies 到向量库 + DB。
    ///
    /// 状态机与 face 类似：
    /// 1. 准备阶段：计算 vector_id，过滤已 indexed
    /// 2. Phase A：插入所有 body row，status=Pending
    /// 3. Phase B：写 HNSW
    /// 4. Phase C：mark_indexed + insert_embedding
    async fn index_bodies(
        &self,
        image_id: i64,
        features: &[BodyFeature],
    ) -> Result<usize, ApplicationError> {
        if features.is_empty() {
            return Ok(0);
        }

        let body_index = self
            .body_index
            .as_ref()
            .ok_or_else(|| ApplicationError::InvalidState("body_index not configured".into()))?;

        // Phase 1: 准备阶段
        let mut prepared: Vec<(NewBody, i64, Embedding)> = Vec::with_capacity(features.len());
        for feat in features {
            let emb: &Embedding = &feat.embedding;
            let vector_id = Self::derive_body_vector_id(image_id, feat);

            let new_body = NewBody {
                image_id,
                bbox_x: feat.bbox.x,
                bbox_y: feat.bbox.y,
                bbox_w: feat.bbox.w,
                bbox_h: feat.bbox.h,
                crop_type: format!("{:?}", feat.crop_strategy),
                embedding_model: "ytu_reid".to_string(),
                model_version: emb.model.as_str().to_string(),
                quality_score: feat.quality_score,
            };

            prepared.push((new_body, vector_id, emb.clone()));
        }

        if prepared.is_empty() {
            return Ok(0);
        }

        // Phase A: 单事务插入所有 body row
        let body_ids: Vec<i64> = self
            .db
            .transaction_with_retry(|tx| {
                let mut ids = Vec::with_capacity(prepared.len());
                for (new_body, _, _) in &prepared {
                    ids.push(tx.bodies().insert(new_body)?);
                }
                Ok::<_, pf_database::DatabaseError>(ids)
            })
            .map_err(ApplicationError::Database)?;

        // Phase B: 逐个写 HNSW
        let mut successes: Vec<(i64, i64, Embedding)> = Vec::new();
        let mut failures: Vec<(i64, String)> = Vec::new();
        for ((_, vector_id, emb), body_id) in prepared.iter().zip(body_ids.iter()) {
            match body_index.insert(*vector_id, emb.as_slice()) {
                Ok(()) => successes.push((*body_id, *vector_id, emb.clone())),
                Err(e) => failures.push((*body_id, format!("hnsw insert: {e}"))),
            }
        }

        // Phase C: mark_indexed + insert_embedding
        let indexed = self
            .db
            .transaction_with_retry(|tx| {
                let mut bodies = tx.bodies();
                for (body_id, vector_id, emb) in &successes {
                    bodies.mark_indexed(*body_id, *vector_id)?;
                    bodies.insert_embedding(*body_id, "ytu_reid", emb.model.as_str(), emb.as_slice(), *vector_id)?;
                }
                for (body_id, err) in &failures {
                    bodies.mark_failed(*body_id, err)?;
                }
                Ok::<_, pf_database::DatabaseError>(successes.len())
            })
            .map_err(ApplicationError::Database)?;

        // 立即持久化 HNSW
        if !successes.is_empty() {
            if let Err(e) = body_index.save() {
                warn!(error = %e, "body_index save failed after indexing {} bodies", indexed);
            }
        }

        Ok(indexed)
    }

    /// ROI-based 对象索引（与 ai-next process_objects 完全一致）。
    ///
    /// 流程：图片 → extract_rois 滑动窗口 → MobileCLIP embed → HNSW insert
    async fn index_objects_roi(
        &self,
        image_id: i64,
        img: &ImageData,
        embedder: Arc<dyn ObjectEmbedder>,
        object_index: Arc<dyn VectorIndex>,
    ) -> Result<usize, ApplicationError> {
        use pf_config::RoiConfig;

        let roi_config = RoiConfig::default();
        let rgb = img.as_rgb8();
        let rois = extract_rois(&rgb, &roi_config);

        debug!(image_id, roi_count = rois.len(), "ROI extraction result");
        if rois.is_empty() {
            return Ok(0);
        }

        let img_w = img.width() as f32;
        let img_h = img.height() as f32;

        // Phase 1: in-memory 预处理 — embed + HNSW + 构造 NewObject
        let mut to_insert: Vec<NewObject> = Vec::with_capacity(rois.len());
        let mut hnsw_inserted: Vec<i64> = Vec::with_capacity(rois.len());

        for (roi_idx, roi) in rois.iter().enumerate() {
            // Crop ROI (ImageData::crop clamps bounds internally)
            let x1 = roi.bbox.x.max(0.0) as u32;
            let y1 = roi.bbox.y.max(0.0) as u32;
            let x2 = (x1 + roi.bbox.w as u32).min(img.width());
            let y2 = (y1 + roi.bbox.h as u32).min(img.height());

            if x2 <= x1 || y2 <= y1 {
                continue;
            }

            let crop = img.crop(BBox::new(
                x1 as f32,
                y1 as f32,
                (x2 - x1) as f32,
                (y2 - y1) as f32,
            ))?;

            let embedding = embedder.embed(&crop).await?;

            let vector_id =
                Self::derive_object_vector_id_from_roi(image_id, roi_idx, img_w, img_h);

            if let Err(e) = object_index.insert(vector_id, embedding.as_slice()) {
                warn!(vector_id, error = %e, "object_index insert failed, skipping roi");
                continue;
            }
            hnsw_inserted.push(vector_id);

            // ROI[0] 始终是 full_image（按 extract_rois 约定），其余来自 sliding window
            let roi_type = if roi_idx == 0 {
                RoiType::FullImage
            } else {
                RoiType::SlidingWindow
            };

            let new_obj = NewObject {
                image_id,
                class_id: 0,
                class_name: format!("roi_{:.2}", roi.scale),
                confidence: 1.0,
                bbox: roi.bbox.clone(),
                model_version: embedding.model.as_str().to_string(),
                vector_id,
                roi_scale: roi.scale,
                roi_type,
            };

            to_insert.push(new_obj);
        }

        // Phase 2: 单事务写入所有 object
        let indexed = if to_insert.is_empty() {
            0
        } else {
            match self.db.transaction_with_retry(|tx| {
                let mut n = 0;
                for new_obj in &to_insert {
                    tx.objects().insert(new_obj)?;
                    n += 1;
                }
                Ok::<_, pf_database::DatabaseError>(n)
            }) {
                Ok(n) => n,
                Err(e) => {
                    warn!(image_id, error = %e, "object batch insert DB failed");
                    // 回滚全部 HNSW 写入
                    for vid in &hnsw_inserted {
                        let _ = object_index.remove(*vid);
                    }
                    return Err(ApplicationError::Database(e));
                }
            }
        };

        // 立即持久化（避免重启后 HNSW object 索引全丢，与 face 路径一致）
        if indexed > 0 {
            if let Err(e) = object_index.save() {
                warn!(error = %e, "object_index save failed after indexing {} rois", indexed);
            }
        }

        Ok(indexed)
    }

    /// 从 ROI 派生稳定的 vector_id（blake3 hash 避免线性组合碰撞，参考 ai-next hash_patch_id）。
    ///
    /// 强制落在负数区间（与 `derive_object_vector_id` 一致），避免与 face vector_id（正数区间）冲突。
    fn derive_object_vector_id_from_roi(image_id: i64, roi_idx: usize, img_w: f32, img_h: f32) -> i64 {
        let key = format!("object:{}:{}:{}:{}", image_id, roi_idx, img_w as u32, img_h as u32);
        let hash = blake3::hash(key.as_bytes());
        let bytes = hash.as_bytes();
        let raw = i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        // 强制负数区间（最高位置 1，与 face 路径分区）
        raw | i64::MIN
    }

    /// 从 body feature 派生稳定的 vector_id。
    fn derive_body_vector_id(image_id: i64, feat: &BodyFeature) -> i64 {
        let b = &feat.bbox;
        let cx = (b.x + b.w / 2.0) as i32;
        let cy = (b.y + b.h / 2.0) as i32;
        let key = format!("body:{}:{}:{}:{}:{}", image_id, cx, cy, b.w as i32, b.h as i32);
        let hash = blake3::hash(key.as_bytes());
        let bytes = hash.as_bytes();
        let raw = i64::from_le_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3],
            bytes[4], bytes[5], bytes[6], bytes[7],
        ]);
        // Body vector_id 使用 0x8000_0000_0000_0000 以上范围，避免与其他 ID 冲突
        raw & 0x7FFF_FFFF_FFFF_FFFF
    }

    /// 把 `Image.id` 映射到 `PhotoId`（桌面端：path 字符串）。
    pub async fn photo_id_for(&self, image_id: i64) -> Result<PhotoId, ApplicationError> {
        let row = self
            .db
            .transaction(|tx| {
                tx.images().get_by_id(image_id).map(|opt| opt.map(|r| r.path))
            })
            .map_err(|e| ApplicationError::Internal(format!("get_by_id: {e}")))?;
        row.map(PhotoId::from)
            .ok_or_else(|| ApplicationError::NotFound(format!("image {image_id}")))
    }

    /// 删除某 image 在向量库 + DB 中的所有对象条目（Phase 2 新增）。
    ///
    /// 流程：
    /// 1. DB transaction：list_vector_ids_by_image → delete_by_image（返回 vector_ids）
    /// 2. 对每个 vector_id 调用 `object_index.remove` 清理 HNSW
    /// 3. 持久化 `object_index.save()`
    ///
    /// 注意：face / patch 不在此处处理（按需单独调用）。
    pub async fn remove_image(&self, image_id: i64) -> Result<usize, ApplicationError> {
        let object_index = self
            .object_index
            .as_ref()
            .ok_or_else(|| ApplicationError::InvalidState("object_index not configured".into()))?;

        // Step 1: DB transaction — list & delete in single tx
        let vector_ids = self
            .db
            .transaction(|tx| tx.objects().delete_by_image(image_id))
            .map_err(ApplicationError::Database)?;

        // Step 2: remove from HNSW
        let mut removed = 0usize;
        for vid in &vector_ids {
            match object_index.remove(*vid) {
                Ok(true) => removed += 1,
                Ok(false) => {
                    debug!(vector_id = vid, "HNSW remove returned false (already absent)");
                }
                Err(e) => {
                    warn!(vector_id = vid, error = %e, "HNSW remove failed");
                }
            }
        }

        // Step 3: persist index
        if let Err(e) = object_index.save() {
            warn!(image_id, error = %e, "object_index save after remove failed");
        }

        info!(image_id, db_deleted = vector_ids.len(), hnsw_removed = removed, "object index removed");
        Ok(vector_ids.len())
    }

    /// 标记 image 已完成 patch 索引（Phase 3）。
    pub async fn mark_patched(&self, image_id: i64) -> Result<(), ApplicationError> {
        info!(image_id, "patch index marked");
        Ok(())
    }

    /// 对一张图片进行 patch 索引（SuperPoint + VLAD）。
    ///
    /// Phase E:
    /// 1. 读 image bytes
    /// 2. 切 grid patches (512×512, stride 256)
    /// 3. SuperPoint 提取关键点 + 描述子
    /// 4. VLAD 聚合 → 全局 256-d descriptor
    /// 5. 插入 patch_index (HNSW)
    /// 6. 存储 patch metadata 到 DB
    pub async fn index_patch_image(
        &self,
        image_id: i64,
        photo_id: &PhotoId,
    ) -> Result<usize, ApplicationError> {
        let patch_extractor = self
            .patch_extractor
            .as_ref()
            .ok_or_else(|| ApplicationError::InvalidState("patch_extractor not configured".into()))?;
        let patch_index = self
            .patch_index
            .as_ref()
            .ok_or_else(|| ApplicationError::InvalidState("patch_index not configured".into()))?;

        let bytes = self
            .photo_provider
            .get_image(photo_id)
            .await
            .map_err(ApplicationError::Platform)?;
        let img = ImageData::from_bytes(&bytes).map_err(ApplicationError::AI)?;

        // 提取 VLAD descriptor + patches metadata
        let (vlad_vec, patch_features) = patch_extractor
            .extract_vlad_descriptor(&img)
            .await
            .map_err(ApplicationError::AI)?;

        if patch_features.is_empty() {
            return Ok(0);
        }

        // 生成稳定 vector_id
        let vector_id = derive_patch_vector_id(image_id);

        // 插入 HNSW index
        if let Err(e) = patch_index.insert(vector_id, &vlad_vec) {
            warn!(vector_id, error = %e, "patch_index insert failed");
        }

        // 存储 patch metadata 到 DB
        let mut indexed = 0usize;
        for pf in &patch_features {
            let res = self.db.transaction(|tx| {
                tx.execute(
                    "INSERT INTO patches (image_id, patch_x, patch_y, patch_w, patch_h, keypoint_count, vector_id)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
                    params![
                        image_id,
                        pf.x as f64,
                        pf.y as f64,
                        pf.w as f64,
                        pf.h as f64,
                        pf.num_keypoints as i64,
                        vector_id,
                    ],
                )?;
                Ok::<_, pf_database::DatabaseError>(())
            });
            match res {
                Ok(_) => indexed += 1,
                Err(e) => warn!(image_id, error = %e, "patch insert DB failed"),
            }
        }

        info!(image_id, patch_count = indexed, "patch indexed");
        Ok(indexed)
    }
}

/// 一键重索引的返回摘要。
#[derive(Debug, Clone, serde::Serialize)]
pub struct ReindexSummary {
    /// 重扫涉及的总图片数
    pub images_total: usize,
    /// 入队的 IndexImage 任务数（一般 = images_total）
    pub queued_tasks: usize,
    /// 删除的 HNSW 文件名
    pub deleted_hnsw_files: Vec<String>,
    /// 清空的 faces 行数
    pub cleared_faces: usize,
    /// 清空的对象行数（include_objects=true 时才有意义）
    pub cleared_objects: usize,
}

/// 一键重索引：清掉人脸/对象 HNSW + DB 行 → 重新入队 IndexImage 任务。
///
/// 用途：模型/算法变更后（如 SCRFD KPS 修复）让旧索引失效，用最新代码重跑。
/// 流程：
/// 1. SQL: `DELETE FROM face_embeddings; DELETE FROM faces;`（可扩展到 objects）
/// 2. 删除磁盘上的 HNSW 文件（`data_dir/index/face.hnsw.{data,graph}`，可选 object）
/// 3. 列出所有 image → 每个 enqueue IndexImage 任务（带 face + object + patch）
/// 4. 返回计数，让前端展示进度
///
/// 注意：
/// - in-memory 的 `face_index`/`object_index` 不会立即清理，但 `search` 路径会通过
///   `faces().get_by_vector_id(...)` 把失效的 vector_id 过滤掉（见 search.rs::hits_to_face_results）。
///   下次进程启动时空 HNSW 会自动 rebuild（从 DB 的 embeddings 加载）。
/// - `persons` 表不动（手动命名/合并的 person 记录保留），face_id 失效会导致下次
///   `cluster_all` 重新生成 person→face 绑定。
pub async fn reindex_all(
    db: Arc<Database>,
    scheduler: Arc<dyn TaskScheduler>,
    data_dir: &Path,
    include_objects: bool,
    enable_patches: bool,
) -> Result<ReindexSummary, ApplicationError> {
    use std::fs;

    // 1) 清人脸 DB
    let (cleared_face_embeddings, cleared_faces) = db
        .transaction(|tx| {
            let emb_n = tx.execute("DELETE FROM face_embeddings", [])
                .map_err(pf_database::DatabaseError::from)?;
            let face_n = tx.execute("DELETE FROM faces", [])
                .map_err(pf_database::DatabaseError::from)?;
            Ok::<_, pf_database::DatabaseError>((emb_n, face_n))
        })
        .map_err(ApplicationError::Database)?;
    info!(
        cleared_face_embeddings,
        cleared_faces, "reindex_all: cleared face rows"
    );

    // 2) 清对象 DB（可选）
    let (cleared_object_embeddings, cleared_objects) = if include_objects {
        db.transaction(|tx| {
            let emb_n = tx.execute("DELETE FROM object_embeddings", [])
                .map_err(pf_database::DatabaseError::from)?;
            let obj_n = tx.execute("DELETE FROM objects", [])
                .map_err(pf_database::DatabaseError::from)?;
            Ok::<_, pf_database::DatabaseError>((emb_n, obj_n))
        })
        .map_err(ApplicationError::Database)?
    } else {
        (0, 0)
    };
    if include_objects {
        info!(
            cleared_object_embeddings,
            cleared_objects, "reindex_all: cleared object rows"
        );
    }

    // 3) 删 HNSW 文件
    let index_dir = data_dir.join("index");
    let candidates = [
        index_dir.join("face.hnsw.data"),
        index_dir.join("face.hnsw.graph"),
        index_dir.join("object.hnsw.data"),
        index_dir.join("object.hnsw.graph"),
    ];
    let mut deleted: Vec<String> = Vec::new();
    for path in &candidates {
        let is_object = path.to_string_lossy().contains("object");
        if is_object && !include_objects {
            continue;
        }
        match fs::remove_file(path) {
            Ok(()) => {
                deleted.push(path.file_name().unwrap_or_default().to_string_lossy().into_owned());
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                debug!(path = %path.display(), "hnsw file not present, skip");
            }
            Err(e) => {
                warn!(path = %path.display(), error = %e, "hnsw file delete failed");
            }
        }
    }
    info!(deleted = ?deleted, "reindex_all: deleted hnsw files");

    // 4) 列出所有 image + 入队 IndexImage
    let image_ids = db
        .transaction(|tx| tx.images().list_paged(i64::MAX as usize, 0))
        .map_err(ApplicationError::Database)?
        .into_iter()
        .map(|r| r.id)
        .collect::<Vec<_>>();
    let mut queued_tasks = 0usize;
    for image_id in &image_ids {
        if let Err(e) = scheduler
            .enqueue(TaskKind::IndexImage { image_id: *image_id }, Priority::Normal)
            .await
        {
            warn!(image_id, error = %e, "reindex_all: enqueue IndexImage failed");
            continue;
        }
        queued_tasks += 1;
        if enable_patches {
            if let Err(e) = scheduler
                .enqueue(TaskKind::IndexPatch { image_id: *image_id }, Priority::Normal)
                .await
            {
                warn!(image_id, error = %e, "reindex_all: enqueue IndexPatch failed");
            }
        }
    }
    info!(
        images_total = image_ids.len(),
        queued_tasks, "reindex_all: queued IndexImage tasks"
    );

    // reindex 清空了 faces 表,完成后自动 enqueue ClusterFaces — 否则所有 face 的 person_id 都是 NULL
    match scheduler
        .enqueue(TaskKind::ClusterFaces, Priority::Normal)
        .await
    {
        Ok(task_id) => info!(task_id, "reindex_all: queued ClusterFaces"),
        Err(e) => warn!(error = %e, "reindex_all: enqueue ClusterFaces failed"),
    }

    Ok(ReindexSummary {
        images_total: image_ids.len(),
        queued_tasks,
        deleted_hnsw_files: deleted,
        cleared_faces: cleared_faces as usize,
        cleared_objects: cleared_objects as usize,
    })
}

/// 根据 image_id + 关键点派生 face vector_id（稳定）。
fn derive_face_vector_id(image_id: i64, feat: &FaceFeature) -> i64 {
    // 用 image_id 的负值 + bbox 中心 hash 防止与 face.id 主键冲突
    let cx = (feat.detection.bbox.x + feat.detection.bbox.w / 2.0) as i32;
    let cy = (feat.detection.bbox.y + feat.detection.bbox.h / 2.0) as i32;
    let h = ((cx as u64) << 32) ^ (cy as u64 ^ image_id as u64);
    (h & 0x7FFF_FFFF_FFFF_FFFF) as i64
}

/// 根据 image_id 派生 patch vector_id（与 face/object 区分）。
fn derive_patch_vector_id(image_id: i64) -> i64 {
    // 用 image_id 乘一个大质数，确保与 face/object vector_id 范围不重叠
    (image_id as u64 * 0x9E3779B97F4A7C15) as i64
}

/// Bridge: FaceFeature → quality f32（统一走 formula v3，与 pipeline `quality::assess` 一致）。
pub fn quality_bridge(det: &pf_ai::FaceDetection, feat: &FaceFeature) -> f32 {
    pf_ai::quality::combined_quality(
        det.score,
        feat.face_area_score,
        feat.blur_score,
        feat.pose_score,
    )
}

// 让 pf_core 的 Face / Object 在 builder 里可用
#[allow(dead_code)]
fn _ensure_types_compile(_f: Face, _o: Object, _b: BBox, _s: SearchResult) {}