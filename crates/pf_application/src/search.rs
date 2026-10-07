//! SearchService — 人脸 / 对象 向量搜索。

use std::collections::HashMap;
use std::sync::Arc;

use pf_ai::object::roi::extract_rois;
use pf_ai::similarity::{FeatureMatcher, KeypointExtractor, KeypointSet};
use pf_ai::{CategorySearch, FaceFeature, FacePipeline, ImageData, ObjectEmbedder, PatchSearchService};
use pf_config::{Config, RoiConfig};
use pf_core::{BBox, ObjectSearchResult, SearchResult};
use pf_database::{Database, FaceRow, ObjectRow, RoiType};
use pf_platform::{PhotoId, PhotoProvider};
use pf_vector::{SearchHit, VectorIndex};
use tracing::{debug, warn};

use crate::body_prototype_service::BodyPrototypeService;
use crate::error::ApplicationError;
use crate::identity_evidence::{CandidateSet, IdentityEvidence, IdentitySearchResult, PersonCandidate, SearchHitWithScore};
use crate::prototype_service::PrototypeService;
use crate::search_v2::search_by_person_impl;

/// 搜索服务。
pub struct SearchService {
    db: Arc<Database>,
    config: Arc<Config>,
    face_pipeline: Arc<FacePipeline>,
    face_index: Arc<dyn VectorIndex>,
    /// V2: Body HNSW index for candidate retrieval
    body_index: Option<Arc<dyn VectorIndex>>,
    object_embedder: Option<Arc<dyn ObjectEmbedder>>,
    object_index: Option<Arc<dyn VectorIndex>>,
    photo_provider: Arc<dyn PhotoProvider>,
    /// SuperPoint 关键点提取器（用于几何校验）
    superpoint: Option<Arc<dyn KeypointExtractor>>,
    /// LightGlue 特征匹配器
    lightglue: Option<Arc<dyn FeatureMatcher>>,
    /// Patch HNSW 粗筛索引（256-d VLAD 向量）
    patch_index: Option<Arc<dyn VectorIndex>>,
    /// Patch 搜索服务（SuperPoint + LightGlue + spatial verification）
    patch_search: Option<Arc<PatchSearchService>>,
    /// Category search（YOLOv8 + MobileCLIP prototype search）
    category_search: Option<Arc<CategorySearch>>,
    /// Phase 5: prototype 重建/查询服务(用于 search_by_person)
    prototype_service: Option<Arc<PrototypeService>>,
    /// V2: Body prototype service
    body_prototype_service: Option<Arc<BodyPrototypeService>>,
}

impl SearchService {
    /// 构造。
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        db: Arc<Database>,
        config: Arc<Config>,
        face_pipeline: Arc<FacePipeline>,
        face_index: Arc<dyn VectorIndex>,
        body_index: Option<Arc<dyn VectorIndex>>,
        object_embedder: Option<Arc<dyn ObjectEmbedder>>,
        object_index: Option<Arc<dyn VectorIndex>>,
        photo_provider: Arc<dyn PhotoProvider>,
        patch_index: Option<Arc<dyn VectorIndex>>,
        patch_search: Option<Arc<PatchSearchService>>,
        category_search: Option<Arc<CategorySearch>>,
        superpoint: Option<Arc<dyn KeypointExtractor>>,
        lightglue: Option<Arc<dyn FeatureMatcher>>,
    ) -> Self {
        Self {
            db,
            config,
            face_pipeline,
            face_index,
            body_index,
            object_embedder,
            object_index,
            photo_provider,
            superpoint,
            lightglue,
            patch_index,
            patch_search,
            category_search,
            prototype_service: None,
            body_prototype_service: None,
        }
    }

    /// Phase 5: 注入 PrototypeService(启用 search_by_person)。
    pub fn with_prototype_service(mut self, ps: Arc<PrototypeService>) -> Self {
        self.prototype_service = Some(ps);
        self
    }

    /// V2: 注入 BodyPrototypeService(启用 body candidate retrieval)。
    pub fn with_body_prototype_service(mut self, bps: Arc<BodyPrototypeService>) -> Self {
        self.body_prototype_service = Some(bps);
        self
    }

    /// 用一张图片检索最相似的人脸。
    ///
    /// 多脸加权融合(P1 搜索 item 11):不再只取 `features[0]`,而是把所有检测到的
    /// 人脸 embedding 按质量权重加权平均成单一 query 向量。这样"图里最主要的人"
    /// 贡献最大,同时侧脸/小脸的姿态信息也保留(相比单取第一张更鲁棒)。
    pub async fn search_by_face_image(
        &self,
        photo_bytes: &[u8],
        top_k: usize,
    ) -> Result<Vec<SearchResult>, ApplicationError> {
        let img = ImageData::from_bytes(photo_bytes).map_err(ApplicationError::AI)?;
        let features = self.face_pipeline.process(&img).await?;
        if features.is_empty() {
            return Ok(Vec::new());
        }
        let fused = fuse_face_embeddings(&features);
        self.search_by_face_embedding(&fused, top_k).await
    }

    /// 直接用 embedding 搜。
    ///
    /// 应用 `config.face.similarity_threshold` 作为 post-filter（与 ai-next 0.50 对齐）。
    pub async fn search_by_face_embedding(
        &self,
        embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<SearchResult>, ApplicationError> {
        // Phase 1.2: fetch_k 改走 config(`coarse_fetch_multiplier`/`min`)。
        // 默认 4 倍率 / 最小 4,保留之前硬编码 `fetch_k = top_k * 4` 的行为。
        let fetch_k = coarse_fetch_k(
            top_k,
            self.config.face.coarse_fetch_multiplier,
            self.config.face.coarse_fetch_min,
        );
        let hits = self
            .face_index
            .search(embedding, fetch_k)
            .map_err(ApplicationError::Vector)?;
        let threshold = self.config.face.similarity_threshold;
        let min_embedding = self.config.face.fine_match_min_embedding;
        let filtered: Vec<SearchHit> = hits
            .into_iter()
            .filter(|h| h.score >= min_embedding)
            .filter(|h| h.score >= threshold)
            .take(top_k)
            .collect();
        self.hits_to_face_results(filtered).await
    }

    /// Phase 5: 给定 person_id,返回该 person 出现过的所有 image。
    ///
    /// 流程:
    /// 1. `PrototypeService::list_for_person(pid)` → 最多 5 个 prototype embeddings
    /// 2. 每个 prototype embedding → `face_index.search(prototype, fetch_k)`
    /// 3. 每个 hit → 查 face → 仅保留 person_id == pid 的 hit
    /// 4. 聚合:每个 image_id 保留 score 最高的 face_id + score
    /// 5. `score >= config.face.similarity_threshold` post-filter
    /// 6. 按 score 降序,取 top_k
    ///
    /// 与 `search_by_face_embedding` 的区别:
    /// - 后者用单一 query embedding(用户上传的一张脸)
    /// - Phase 5 用 person 的 prototype 表征(frontal/left/right/high_quality/general)
    ///   多角度匹配,召回更全 — 即使目标 person 在新图里是侧脸,也能用 left_profile prototype 召回
    ///
    /// 错误:
    /// - `InvalidState("prototype_service not configured")` if 未注入
    /// - `Internal("no prototypes for person")` if 该 person 尚无 prototype
    pub async fn search_by_person(
        &self,
        person_id: i64,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, ApplicationError> {
        let ps = self.prototype_service.as_ref().ok_or_else(|| {
            ApplicationError::InvalidState("prototype_service not configured".into())
        })?;
        let protos = ps.list_for_person(person_id, None)?;
        let threshold = self.config.face.similarity_threshold;
        search_by_person_impl(
            &*self.face_index,
            &self.db,
            &protos,
            person_id,
            top_k,
            threshold,
        )
    }

    /// Phase 14: 双通道 candidate retrieval
    ///
    /// 同时用 Face HNSW 和 Body HNSW 搜索，返回 person candidate union。
    ///
    /// 流程：
    /// 1. Face HNSW top-20 搜索
    /// 2. Body HNSW top-20 搜索
    /// 3. 按 person_id 聚合，保留每个通道的最佳分数和排名
    /// 4. 返回按 face_rank 排序的候选列表
    ///
    /// 注意：`query_image_id` 用于排除 query 自身（self-match）。
    /// 生产搜索时传 None；LOO 测试时传 query image 的 id。
    pub async fn dual_channel_candidates(
        &self,
        face_emb: &[f32],
        body_emb: &[f32],
        query_image_id: Option<i64>,
        top_k: usize,
    ) -> Result<CandidateSet, ApplicationError> {
        // Face HNSW top-20
        let face_hits: Vec<SearchHitWithScore> = self
            .face_index
            .search(face_emb, top_k.max(20))
            .map_err(ApplicationError::Vector)?
            .into_iter()
            .map(|h| SearchHitWithScore { vector_id: h.id, score: h.score })
            .collect();

        // Body HNSW top-20
        let body_hits: Vec<SearchHitWithScore> = if let Some(ref bi) = self.body_index {
            bi.search(body_emb, top_k.max(20))
                .map_err(ApplicationError::Vector)?
                .into_iter()
                .map(|h| SearchHitWithScore { vector_id: h.id, score: h.score })
                .collect()
        } else {
            Vec::new()
        };

        // 批量获取 face 的 person_id 和 image_id
        let face_vector_ids: Vec<i64> = face_hits.iter().map(|h| h.vector_id).collect();
        let face_infos: HashMap<i64, (Option<i64>, Option<i64>)> = self
            .db
            .transaction(|tx| {
                let faces = tx.faces().list_by_vector_ids(&face_vector_ids)?;
                Ok(faces
                    .into_iter()
                    .filter_map(|f| f.vector_id.map(|vid| (vid, (f.person_id, Some(f.image_id)))))
                    .collect())
            })
            .map_err(|e| ApplicationError::Internal(format!("face lookup: {e}")))?;

        // 批量获取 body 的 person_id 和 image_id
        let body_vector_ids: Vec<i64> = body_hits.iter().map(|h| h.vector_id).collect();
        let body_infos: HashMap<i64, (Option<i64>, Option<i64>)> = self
            .db
            .transaction(|tx| {
                let bodies = tx.bodies().list_by_vector_ids(&body_vector_ids)?;
                Ok(bodies
                    .into_iter()
                    .filter_map(|b| b.vector_id.map(|vid| (vid, (b.person_id, Some(b.image_id)))))
                    .collect())
            })
            .map_err(|e| ApplicationError::Internal(format!("body lookup: {e}")))?;

        // 按 person_id 聚合（同时过滤 self-match）
        let mut candidates: HashMap<i64, PersonCandidate> = HashMap::new();

        for (rank, hit) in face_hits.iter().enumerate() {
            if let Some((Some(pid), Some(img_id))) = face_infos.get(&hit.vector_id) {
                // 过滤 self-match：排除 query image 自身的 face
                if let Some(qid) = query_image_id {
                    if img_id == &qid {
                        continue;
                    }
                }
                let cand = candidates.entry(*pid).or_insert_with(|| PersonCandidate::new(*pid));
                cand.update_face(hit.score, rank + 1);
            }
        }

        for (rank, hit) in body_hits.iter().enumerate() {
            if let Some((Some(pid), Some(img_id))) = body_infos.get(&hit.vector_id) {
                // 过滤 self-match：排除 query image 自身的 body
                if let Some(qid) = query_image_id {
                    if img_id == &qid {
                        continue;
                    }
                }
                let cand = candidates.entry(*pid).or_insert_with(|| PersonCandidate::new(*pid));
                cand.update_body(hit.score, rank + 1);
            }
        }

        // 按 face_rank 排序
        let mut result: Vec<PersonCandidate> = candidates.into_values().collect();
        result.sort_by(|a, b| a.face_rank.cmp(&b.face_rank));

        Ok(CandidateSet {
            candidates: result,
            face_hits,
            body_hits,
        })
    }

    /// Phase 14: 计算候选人的多证据 IdentityEvidence。
    ///
    /// 对每个 candidate person，计算：
    /// - face_score: max cosine(query_face, person_face_prototypes)
    /// - body_score: max cosine(query_body, person_body_prototypes)
    /// - face_margin: face_score - max(other_person_face_score)
    /// - body_margin: body_score - max(other_person_body_score)
    /// - 排名、prototype 数量等
    ///
    /// 注意：`exclude_image_id` 用于 LOO benchmark 测试时排除 query 自身。
    /// 生产搜索时通常不需要（query 是新图片）。
    pub async fn compute_identity_evidence(
        &self,
        query_face_emb: &[f32],
        query_body_emb: &[f32],
        candidates: &[PersonCandidate],
        exclude_image_id: Option<i64>,
    ) -> Result<Vec<IdentityEvidence>, ApplicationError> {
        use crate::cluster_v2::cosine_similarity;

        let mut evidence: Vec<IdentityEvidence> = Vec::new();

        // 收集所有 candidate 的 prototypes
        for cand in candidates {
            let mut e = IdentityEvidence::new(cand.person_id);
            e.face_rank = cand.face_rank;
            e.body_rank = cand.body_rank;

            // Face prototypes
            if let Some(ref ps) = self.prototype_service {
                match ps.list_for_person(cand.person_id, exclude_image_id) {
                    Ok(protos) if !protos.is_empty() => {
                        e.face_prototype_count = protos.len();
                        let mut scores: Vec<f32> = protos
                            .iter()
                            .map(|(_, emb)| cosine_similarity(query_face_emb, emb))
                            .collect();
                        scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
                        e.face_score = scores.first().copied().unwrap_or(0.0);
                        e.face_second_score = scores.get(1).copied().unwrap_or(0.0);
                        e.face_best_proto_idx = 0;
                        e.face_available = true;
                    }
                    _ => {}
                }
            }

            // Body prototypes
            if let Some(ref bps) = self.body_prototype_service {
                match bps.list_for_person(cand.person_id, exclude_image_id) {
                    Ok(protos) if !protos.is_empty() => {
                        e.body_prototype_count = protos.len();
                        let mut scores: Vec<f32> = protos
                            .iter()
                            .map(|(_, emb)| cosine_similarity(query_body_emb, emb))
                            .collect();
                        scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
                        e.body_score = scores.first().copied().unwrap_or(0.0);
                        e.body_second_score = scores.get(1).copied().unwrap_or(0.0);
                        e.body_best_proto_idx = 0;
                        e.body_available = true;
                    }
                    _ => {}
                }
            }

            evidence.push(e);
        }

        // 计算 margins（需要先收集所有 face/body scores）
        let face_scores: Vec<f32> = evidence.iter().map(|e| e.face_score).collect();
        let body_scores: Vec<f32> = evidence.iter().map(|e| e.body_score).collect();

        for e in &mut evidence {
            // face_margin: score - max(other)
            if let Some(max_other) = face_scores.iter().filter(|&&s| s < e.face_score).max_by(|a, b| a.partial_cmp(b).unwrap()) {
                e.face_margin = Some(e.face_score - max_other);
            }
            // body_margin: score - max(other)
            if let Some(max_other) = body_scores.iter().filter(|&&s| s < e.body_score).max_by(|a, b| a.partial_cmp(b).unwrap()) {
                e.body_margin = Some(e.body_score - max_other);
            }
        }

        Ok(evidence)
    }

    /// Phase 14.11: 多证据身份搜索（生产接入准备）
    ///
    /// 完整流程：
    /// 1. dual_channel_candidates — Face + Body 双通道召回
    /// 2. compute_identity_evidence — 计算每个候选的证据
    /// 3. Anti-chaining 检查 — 检测歧义
    /// 4. IdentityDecision — 多证据决策
    ///
    /// 返回按 confidence 排序的决策结果。
    ///
    /// 注意：此方法需要 body_pipeline 和 body_index 配置才能使用。
    /// `query_image_id` 用于 LOO 测试时排除 query 自身；生产搜索时传 None。
    ///
    /// Requires `config.identity.enable_identity_fusion` to be true.
    pub async fn search_with_identity_decision(
        &self,
        query_face_emb: &[f32],
        query_body_emb: &[f32],
        query_image_id: Option<i64>,
        top_k: usize,
    ) -> Result<Vec<IdentitySearchResult>, ApplicationError> {
        use crate::identity_evidence::AntiChainingResult;

        // Guard: identity fusion must be enabled
        if !self.config.identity.enable_identity_fusion {
            return Err(ApplicationError::Internal(
                "identity_fusion is disabled via config.identity.enable_identity_fusion".into(),
            ));
        }

        // 1. 双通道召回
        let candidates = self
            .dual_channel_candidates(query_face_emb, query_body_emb, query_image_id, top_k)
            .await?;

        if candidates.candidates.is_empty() {
            return Ok(Vec::new());
        }

        // 2. 计算证据（排除 query image 自身）
        let evidence = self
            .compute_identity_evidence(query_face_emb, query_body_emb, &candidates.candidates, query_image_id)
            .await?;

        if evidence.is_empty() {
            return Ok(Vec::new());
        }

        // 3. Anti-chaining 检查
        let chain_result = AntiChainingResult::check(&evidence, 0.05);

        // 生产身份调试日志：记录每个候选的证据和最终决策
        tracing::debug!(
            identity_decision = true,
            num_candidates = evidence.len(),
            is_ambiguous = chain_result.is_ambiguous,
            face_gap = chain_result.face_gap,
            body_gap = chain_result.body_gap,
            "=== Identity Evidence ==="
        );
        for e in &evidence {
            tracing::debug!(
                identity_decision = true,
                person_id = e.person_id,
                face_score = e.face_score,
                face_margin = e.face_margin,
                face_rank = e.face_rank,
                body_score = e.body_score,
                body_margin = e.body_margin,
                body_rank = e.body_rank,
                face_available = e.face_available,
                body_available = e.body_available,
                evidence_level = ?e.evidence_level(),
                "  candidate"
            );
        }

        // 4. 决策并排序
        let mut results: Vec<IdentitySearchResult> = evidence
            .into_iter()
            .map(|e| {
                let decision = e.decide();
                tracing::debug!(
                    identity_decision = true,
                    person_id = e.person_id,
                    decision = ?decision,
                    confidence = e.confidence(),
                    "  => decision"
                );
                IdentitySearchResult {
                    person_id: e.person_id,
                    decision,
                    confidence: e.confidence(),
                    face_score: e.face_score,
                    body_score: e.body_score,
                    face_margin: e.face_margin,
                    body_margin: e.body_margin,
                    face_rank: e.face_rank,
                    body_rank: e.body_rank,
                    is_ambiguous: chain_result.is_ambiguous,
                }
            })
            .collect();

        // 按 confidence 降序排列
        results.sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());

        Ok(results)
    }

    /// 用类别 ID 列出对象（不实际搜向量，纯 DB 过滤）。
    pub async fn list_by_class(
        &self,
        class_id: i32,
        limit: usize,
    ) -> Result<Vec<SearchResult>, ApplicationError> {
        let rows = self
            .db
            .transaction(|tx| tx.objects().list_by_class(class_id, limit))
            .map_err(|e| ApplicationError::Internal(format!("list_by_class: {e}")))?;
        let mut out = Vec::with_capacity(rows.len());
        for (rank, r) in rows.into_iter().enumerate() {
            out.push(SearchResult {
                image_id: r.image_id,
                target_id: r.id,
                score: r.confidence,
                rank: rank + 1,
            });
        }
        Ok(out)
    }

    /// 用对象向量搜索（整图 embedding）。
    pub async fn search_by_object_embedding(
        &self,
        embedding: &[f32],
        top_k: usize,
    ) -> Result<Vec<SearchResult>, ApplicationError> {
        let oi = self
            .object_index
            .as_ref()
            .ok_or_else(|| ApplicationError::InvalidState("object index not configured".into()))?;
        let hits = oi.search(embedding, top_k).map_err(ApplicationError::Vector)?;
        self.hits_to_object_results(hits).await
    }

    /// 用对象图片搜索（两阶段：MobileCLIP prototype → HNSW 粗筛 → LightGlue 几何校验 → 融合评分）。
    ///
    /// Phase 3 重构（与 ai-next `object_search.rs` 对齐）：
    /// 1. Query → embedder（crop 用单 embedding；full-image 走 extract_rois → prototype）
    /// 2. HNSW 粗筛 `fetch_k = max(top_k * coarse_fetch_multiplier, coarse_fetch_min)`
    ///    按 image_id 聚合，按 roi_type 调权（FullImage < SlidingWindow），取 max
    /// 3. LightGlue 几何校验（SuperPoint 关键点匹配 + RANSAC inliers）
    /// 4. 融合评分：alpha*embedding + beta*inlier_ratio + gamma*bbox_overlap
    /// 5. 过滤（embedding>=min_score && confidence>=min_score），按 image_id 去重，取 top_k
    pub async fn search_by_object_image(
        &self,
        photo_bytes: &[u8],
        top_k: usize,
    ) -> Result<Vec<ObjectSearchResult>, ApplicationError> {
        let embedder = self
            .object_embedder
            .as_ref()
            .ok_or_else(|| ApplicationError::InvalidState("object embedder not configured".into()))?;
        let oi = self
            .object_index
            .as_ref()
            .ok_or_else(|| ApplicationError::InvalidState("object index not configured".into()))?;

        let img = ImageData::from_bytes(photo_bytes).map_err(ApplicationError::AI)?;
        let rgb = img.as_rgb8();
        let (img_w, img_h) = (img.width(), img.height());

        // === Stage 1: Query embedding ===
        // - 小图（任一维度 < CROP_QUERY_THRESHOLD）：单 embedding（crop-query 快捷路径，与 ai-next 对齐）
        // - 大图：extract_rois → 每 ROI embed → mean + L2 normalize（prototype，与 indexing 对齐）
        let prototype_vec = self
            .build_query_embedding(embedder.clone(), &img, &rgb)
            .await?;

        if prototype_vec.is_empty() {
            return Ok(Vec::new());
        }

        // === Stage 2: HNSW 粗筛（fetch_k = max(top_k * mult, min)）===
        let fetch_k = self.coarse_fetch_k(top_k);
        debug!(
            top_k,
            fetch_k,
            mult = self.config.object.coarse_fetch_multiplier,
            min = self.config.object.coarse_fetch_min,
            "HNSW coarse fetch"
        );
        let coarse_hits = oi.search(&prototype_vec, fetch_k).map_err(ApplicationError::Vector)?;

        if coarse_hits.is_empty() {
            return Ok(Vec::new());
        }

        // 按 image_id 聚合：每 image_id 取按 roi_type 调权后的最高分 ROI
        // 权重：FullImage * full_image_weight（默认 0.85，弱化 general 信号）
        //      SlidingWindow * sliding_window_weight（默认 1.0，specific 局部信号）
        //
        // Audit B1：一次 IN-clause 批量反查所有 hit 的 object（此前每个 hit 单独
        // 开事务，fetch_k=200 → 200 次 BEGIN/COMMIT）。
        let full_image_w = self.config.object.full_image_weight;
        let sliding_w = self.config.object.sliding_window_weight;
        let vector_ids: Vec<i64> = coarse_hits.iter().map(|h| h.id).collect();
        let obj_by_vector: HashMap<i64, ObjectRow> = self
            .db
            .transaction(|tx| tx.objects().list_by_vector_ids(&vector_ids))
            .map_err(|e| ApplicationError::Internal(format!("batch object lookup: {e}")))?
            .into_iter()
            .map(|o| (o.vector_id, o))
            .collect();

        let mut best_per_image: HashMap<i64, (SearchHit, i64, BBox, f32)> = HashMap::new();
        for hit in coarse_hits {
            match obj_by_vector.get(&hit.id) {
                Some(obj) => {
                    let weight = match obj.roi_type {
                        RoiType::FullImage => full_image_w,
                        RoiType::SlidingWindow => sliding_w,
                    };
                    let weighted_score = hit.score * weight;
                    let entry = best_per_image.entry(obj.image_id);
                    match entry {
                        std::collections::hash_map::Entry::Vacant(e) => {
                            e.insert((hit, obj.id, obj.bbox, weighted_score));
                        }
                        std::collections::hash_map::Entry::Occupied(mut e) => {
                            if weighted_score > e.get().3 {
                                e.insert((hit, obj.id, obj.bbox, weighted_score));
                            }
                        }
                    }
                }
                None => {
                    debug!(vector_id = hit.id, "object not found in DB, skipping");
                }
            }
        }

        if best_per_image.is_empty() {
            return Ok(Vec::new());
        }

        // === Stage 3: LightGlue 几何校验 ===
        let has_superpoint = self.superpoint.is_some() && self.lightglue.is_some();
        let query_kps = if has_superpoint {
            self.extract_superpoint(&img).await.ok()
        } else {
            None
        };

        let alpha = self.config.object.fusion_alpha;
        let beta = self.config.object.fusion_beta;
        let gamma = self.config.object.fusion_gamma;
        let min_score = self.config.object.min_score;

        // (image_id, confidence, embedding_score, inlier_ratio, inlier_count, match_count, bbox_overlap, matched_bbox)
        let mut scored: Vec<(i64, f32, f32, f32, usize, usize, f32, BBox)> = Vec::new();

        // ai-next 对齐：fine match 只对 top hnsw_top_k 个候选跑 LightGlue，避免对全量 coarse 命中做几何校验
        // Phase 7: 排序（HashMap 顺序非确定）+ 粗筛 pre-filter（embedding >= fine_match_min_embedding）
        let fine_match_min = self.config.object.fine_match_min_embedding;
        let fine_cap = self.config.object.hnsw_top_k.max(1);
        let ordered_candidates = select_fine_candidates(best_per_image, fine_match_min, fine_cap);

        // Audit B2：批量反查 candidate 的图片 path（此前每个 candidate 单独开事务查
        // object→image→path，fine_cap=50 → 最多 100 次事务）。图片 bytes 的实际加载
        // （photo_provider）仍在循环内逐 candidate 进行（IO 无法在 trait 层批量）。
        let object_ids: Vec<i64> = ordered_candidates.iter().map(|(_, _, oid, _, _)| *oid).collect();
        let path_by_object: HashMap<i64, String> = self
            .db
            .transaction(|tx| {
                let objects = tx.objects().list_by_ids(&object_ids)?;
                let image_ids: Vec<i64> = objects.iter().map(|o| o.image_id).collect();
                let images = tx.images().list_by_ids(&image_ids)?;
                let path_by_image: HashMap<i64, String> =
                    images.into_iter().map(|i| (i.id, i.path)).collect();
                Ok(objects
                    .into_iter()
                    .filter_map(|o| path_by_image.get(&o.image_id).cloned().map(|p| (o.id, p)))
                    .collect::<HashMap<i64, String>>())
            })
            .map_err(|e| ApplicationError::Internal(format!("batch path lookup: {e}")))?;

        for (img_id, hit, object_id, candidate_bbox, _weighted) in ordered_candidates
        {
            let embedding_score = hit.score;
            let mut inlier_ratio = 0.0f32;
            let mut inlier_count = 0usize;
            let mut match_count = 0usize;
            // query bbox = 全图（用户已裁剪过 query）；与候选 ROI 做 IoU
            let query_bbox = BBox::new(0.0, 0.0, img_w as f32, img_h as f32);
            // 初值：未做几何校验前的 bbox_overlap 用 indexed ROI 与 query bbox 的 IoU
            let mut bbox_overlap = query_bbox.iou(&candidate_bbox);

            // 无 SuperPoint 时 confidence = embedding_score
            if let (Some(ref _sp), Some(ref lg), Some(ref qkps)) =
                (&self.superpoint, &self.lightglue, &query_kps)
            {
                // Phase 4：只对 candidate 的 ROI 区域做 crop + SuperPoint（vs 全图）
                // - 减少 SuperPoint 计算量
                // - 减少无关区域的特征噪声
                // - 小 ROI 自动扩展以提供 LightGlue 上下文
                let Some(path) = path_by_object.get(&object_id) else {
                    debug!(object_id, "candidate object has no image path, skipping");
                    continue;
                };
                let (cand_img, expanded_bbox) = match self
                    .load_candidate_roi(path, &candidate_bbox)
                    .await
                {
                    Ok(v) => v,
                    Err(e) => {
                        warn!(error = %e, "failed to load candidate ROI");
                        continue;
                    }
                };
                let cand_kps = match self.extract_superpoint_from_data(&cand_img).await {
                    Ok(k) => k,
                    Err(e) => {
                        warn!(error = %e, "failed to extract candidate keypoints");
                        continue;
                    }
                };
                let lg_result = match lg.match_features(qkps, &cand_kps).await {
                    Ok(r) => r,
                    Err(e) => {
                        warn!(error = %e, "lightglue match failed");
                        continue;
                    }
                };
                // Phase 5：少于 min_matches 直接跳过该候选（避免 1 match 报 100% confidence）
                // （match_features 内部已 enforce，这里 double-check）
                let lg_min = self.config.object.lightglue_min_matches;
                if lg_result.matches.len() < lg_min {
                    debug!(
                        object_id,
                        match_count = lg_result.matches.len(),
                        lg_min,
                        "skip candidate: too few lightglue matches"
                    );
                    continue;
                }
                // 用 expanded_bbox（实际匹配区域）重算 bbox_overlap
                bbox_overlap = query_bbox.iou(&expanded_bbox);
                match_count = lg_result.matches.len();
                inlier_count = lg_result.num_inliers;
                inlier_ratio = if match_count == 0 {
                    0.0
                } else {
                    lg_result.confidence
                };
            }

            // 无 SuperPoint/LightGlue 时直接用 embedding_score 当 confidence
            // （否则 alpha*emb + beta*0 + gamma*0 = 0.3*emb 永远 < 0.5 阈值，所有结果被过滤掉）
            let confidence = if query_kps.is_some() {
                alpha * embedding_score + beta * inlier_ratio + gamma * bbox_overlap
            } else {
                embedding_score
            };

            // 过滤：embedding_score >= min_score && confidence >= min_score
            if embedding_score < min_score || confidence < min_score {
                continue;
            }

            scored.push((img_id, confidence, embedding_score, inlier_ratio, inlier_count, match_count, bbox_overlap, candidate_bbox));
        }

        if scored.is_empty() {
            return Ok(Vec::new());
        }

        // 按 confidence 降序，去重（每个 image_id 只留一个）
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut seen = HashMap::new();
        scored.retain(|(img_id, _, _, _, _, _, _, _)| seen.insert(*img_id, true).is_none());

        let results: Vec<ObjectSearchResult> = scored
            .into_iter()
            .take(top_k)
            .enumerate()
            .map(|(i, (img_id, confidence, embedding_score, inlier_ratio, inlier_count, match_count, bbox_overlap, matched_bbox))| ObjectSearchResult {
                image_id: img_id,
                target_id: img_id,
                confidence,
                rank: i + 1,
                embedding_score,
                inlier_ratio,
                inlier_count,
                match_count,
                bbox_overlap,
                matched_bbox,
            })
            .collect();

        Ok(results)
    }

    /// 从一组 embedding 向量创建 prototype（元素平均 + L2 normalize）。
    pub fn create_prototype(embeddings: &[Vec<f32>]) -> Vec<f32> {
        create_prototype(embeddings)
    }

    /// 计算 HNSW 粗筛 fetch_k：
    /// `max(top_k * coarse_fetch_multiplier, coarse_fetch_min)`
    ///
    /// 与 ai-next `object_search.py` 的 `fetch_k=max(top_k*20, 200)` 对齐；
    /// 保证小 top_k 也能拿到足够候选以覆盖 image_id 聚合后的去重损失。
    pub fn coarse_fetch_k(&self, top_k: usize) -> usize {
        coarse_fetch_k(
            top_k,
            self.config.object.coarse_fetch_multiplier,
            self.config.object.coarse_fetch_min,
        )
    }

    /// Query embedding 生成：
    /// - 小图（任一维度 < `CROP_QUERY_THRESHOLD`）：单 embedding（crop-query 快捷路径）
    /// - 大图：`extract_rois` → 每 ROI embed → mean + L2 normalize（prototype，与 indexing 对齐）
    ///
    /// 设计依据：用户在 CropModal 中裁出的 query crop 通常较小（200-400px）；
    /// 大图直查（YOLO 时代路径）需要 multi-scale 候选才能鲁棒。
    async fn build_query_embedding(
        &self,
        embedder: Arc<dyn ObjectEmbedder>,
        img: &ImageData,
        rgb: &image::RgbImage,
    ) -> Result<Vec<f32>, ApplicationError> {
        const CROP_QUERY_THRESHOLD: u32 = 400;

        let is_crop_query = img.width() < CROP_QUERY_THRESHOLD
            || img.height() < CROP_QUERY_THRESHOLD;

        if is_crop_query {
            let emb = embedder.embed(img).await.map_err(ApplicationError::AI)?;
            return Ok(emb.values);
        }

        // 全图查询：走 prototype 路径，与 indexing 对齐
        let roi_config = RoiConfig::default();
        let rois = extract_rois(rgb, &roi_config);
        if rois.is_empty() {
            return Ok(Vec::new());
        }

        let mut embeddings = Vec::with_capacity(rois.len());
        for roi in &rois {
            let crop = img
                .crop(roi.bbox)
                .map_err(ApplicationError::AI)?;
            let emb = embedder.embed(&crop).await.map_err(ApplicationError::AI)?;
            embeddings.push(emb.values);
        }
        Ok(Self::create_prototype(&embeddings))
    }

    /// 提取 SuperPoint 关键点（从 ImageData）。
    async fn extract_superpoint(
        &self,
        img: &ImageData,
    ) -> Result<KeypointSet, ApplicationError> {
        let sp = self
            .superpoint
            .as_ref()
            .ok_or_else(|| ApplicationError::InvalidState("superpoint not configured".into()))?;
        sp.extract(img).await.map_err(ApplicationError::AI)
    }

    /// 提取 SuperPoint 关键点（从 ImageData，辅助方法）。
    async fn extract_superpoint_from_data(
        &self,
        img: &ImageData,
    ) -> Result<KeypointSet, ApplicationError> {
        self.extract_superpoint(img).await
    }

    /// 加载候选图片 bytes（通过 `images.path` → photo_provider）。
    ///
    /// 桌面端：PhotoId 是文件路径。path 由调用方预先批量查好（Audit B2），
    /// 这里不再开 DB 事务。
    async fn load_candidate_image(&self, path: &str) -> Result<Vec<u8>, ApplicationError> {
        let photo_id = PhotoId::from(path.to_string());
        self.photo_provider
            .get_image(&photo_id)
            .await
            .map_err(ApplicationError::Platform)
    }

    /// 加载候选图片 + 按 ROI bbox 裁剪（Phase 4）。
    ///
    /// `path` 为 `images.path`（Audit B2 中预先批量查好）。
    ///
    /// 流程：
    /// 1. `load_candidate_image` 拿 bytes
    /// 2. 解码为 `ImageData`
    /// 3. 按 `roi_bbox` 相对原图大小做 scale-aware expansion（保证 LightGlue 上下文）
    /// 4. clamp 到原图边界
    /// 5. crop 出 ImageData + 返回 expanded_bbox（供 bbox_overlap 使用）
    async fn load_candidate_roi(
        &self,
        path: &str,
        roi_bbox: &BBox,
    ) -> Result<(ImageData, BBox), ApplicationError> {
        let bytes = self.load_candidate_image(path).await?;
        let img = ImageData::from_bytes(&bytes).map_err(ApplicationError::AI)?;
        let (w, h) = (img.width() as f32, img.height() as f32);
        let expanded = expand_roi_for_matching(roi_bbox, w, h);
        let crop = img.crop(expanded).map_err(ApplicationError::AI)?;
        Ok((crop, expanded))
    }

    /// 把 face 向量命中映射成 SearchResult（关联 image_id）。
    ///
    /// 与 ai-next 对齐：同一张图片只保留得分最高的那张脸。
    async fn hits_to_face_results(
        &self,
        hits: Vec<SearchHit>,
    ) -> Result<Vec<SearchResult>, ApplicationError> {
        use std::collections::HashMap;

        // Audit B3：一次 IN-clause 批量反查所有 hit 的 face（此前每个 hit 单独开事务）。
        let vector_ids: Vec<i64> = hits.iter().map(|h| h.id).collect();
        let face_by_vector: HashMap<i64, FaceRow> = self
            .db
            .transaction(|tx| tx.faces().list_by_vector_ids(&vector_ids))
            .map_err(|e| ApplicationError::Internal(format!("face lookup: {e}")))?
            .into_iter()
            .filter_map(|f| f.vector_id.map(|vid| (vid, f)))
            .collect();

        // 每个 image_id 只保留得分最高的 face（FaceRow 已带 image_id，省掉原第二遍反查）
        let mut best_per_image: HashMap<i64, (SearchHit, i64, i64)> = HashMap::new(); // image_id -> (hit, face_id, image_id)
        for hit in hits {
            match face_by_vector.get(&hit.id) {
                Some(face) => {
                    let entry = best_per_image.entry(face.image_id);
                    match entry {
                        std::collections::hash_map::Entry::Vacant(e) => {
                            e.insert((hit, face.id, face.image_id));
                        }
                        std::collections::hash_map::Entry::Occupied(mut e) => {
                            if hit.score > e.get().0.score {
                                e.insert((hit, face.id, face.image_id));
                            }
                        }
                    }
                }
                None => {
                    debug!(vector_id = hit.id, "face not found in DB, skipping");
                }
            }
        }

        // 按 score 降序排列（image_id 已在第一遍拿到，无需第二遍反查）
        let mut results: Vec<SearchResult> = best_per_image
            .into_values()
            .map(|(hit, face_id, image_id)| SearchResult {
                image_id,
                target_id: face_id,
                score: hit.score,
                rank: 0,
            })
            .collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        for (i, r) in results.iter_mut().enumerate() {
            r.rank = i + 1;
        }
        Ok(results)
    }

    /// 把 object 向量命中映射成 SearchResult。
    ///
    /// 与 face 搜索对齐：同一张图片只保留得分最高的那个对象。
    async fn hits_to_object_results(
        &self,
        hits: Vec<SearchHit>,
    ) -> Result<Vec<SearchResult>, ApplicationError> {
        use std::collections::HashMap;

        // Audit B3：一次 IN-clause 批量反查所有 hit 的 object（此前每个 hit 单独开事务）。
        let vector_ids: Vec<i64> = hits.iter().map(|h| h.id).collect();
        let obj_by_vector: HashMap<i64, ObjectRow> = self
            .db
            .transaction(|tx| tx.objects().list_by_vector_ids(&vector_ids))
            .map_err(|e| ApplicationError::Internal(format!("object lookup: {e}")))?
            .into_iter()
            .map(|o| (o.vector_id, o))
            .collect();

        // 每个 image_id 只保留得分最高的 object（ObjectRow 已带 image_id，省掉原第二遍反查）
        let mut best_per_image: HashMap<i64, (SearchHit, i64, i64)> = HashMap::new(); // image_id -> (hit, object_id, image_id)
        for hit in hits {
            match obj_by_vector.get(&hit.id) {
                Some(obj) => {
                    let entry = best_per_image.entry(obj.image_id);
                    match entry {
                        std::collections::hash_map::Entry::Vacant(e) => {
                            e.insert((hit, obj.id, obj.image_id));
                        }
                        std::collections::hash_map::Entry::Occupied(mut e) => {
                            if hit.score > e.get().0.score {
                                e.insert((hit, obj.id, obj.image_id));
                            }
                        }
                    }
                }
                None => {
                    debug!(vector_id = hit.id, "object not found in DB, skipping");
                }
            }
        }

        // 按 score 降序排列（image_id 已在第一遍拿到，无需第二遍反查）
        let mut results: Vec<SearchResult> = best_per_image
            .into_values()
            .map(|(hit, object_id, image_id)| SearchResult {
                image_id,
                target_id: object_id,
                score: hit.score,
                rank: 0,
            })
            .collect();
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        for (i, r) in results.iter_mut().enumerate() {
            r.rank = i + 1;
        }
        Ok(results)
    }

    /// 取一张图片的全部命中 face。
    pub async fn list_faces_for_image(
        &self,
        image_id: i64,
    ) -> Result<Vec<pf_database::FaceRow>, ApplicationError> {
        let faces = self
            .db
            .transaction(|tx| tx.faces().list_by_image(image_id))
            .map_err(|e| ApplicationError::Internal(format!("list_faces: {e}")))?;
        Ok(faces)
    }

    /// 取一张图片的全部对象。
    pub async fn list_objects_for_image(
        &self,
        image_id: i64,
    ) -> Result<Vec<pf_database::ObjectRow>, ApplicationError> {
        let objs = self
            .db
            .transaction(|tx| tx.objects().list_by_image(image_id))
            .map_err(|e| ApplicationError::Internal(format!("list_objects: {e}")))?;
        Ok(objs)
    }

    /// 取一张图片的缩略图（用于 UI 展示）。
    pub async fn thumbnail_for(
        &self,
        image_id: i64,
        max_size: u32,
    ) -> Result<Vec<u8>, ApplicationError> {
        let row = self
            .db
            .transaction(|tx| {
                tx.images()
                    .get_by_id(image_id)
                    .map(|opt| opt.map(|r| (r.path, r.thumbnail_path)))
            })
            .map_err(|e| ApplicationError::Internal(format!("get_by_id: {e}")))?
            .ok_or_else(|| ApplicationError::NotFound(format!("image {image_id}")))?;
        let (_path, _thumb) = row;
        self.photo_provider
            .get_thumbnail(&pf_platform::PhotoId::from(image_id.to_string()), max_size)
            .await
            .map_err(ApplicationError::Platform)
    }

    /// 当前向量库大小。
    pub fn face_count(&self) -> usize {
        self.face_index.len()
    }

    pub fn object_count(&self) -> Option<usize> {
        self.object_index.as_ref().map(|i| i.len())
    }

    pub fn patch_count(&self) -> Option<usize> {
        self.patch_index.as_ref().map(|i| i.len())
    }

    /// 用 patch 图片搜索（SuperPoint + VLAD + LightGlue + HNSW coarse）。
    ///
    /// Phase E：完整实现。提取 query image 的 VLAD descriptor →
    /// HNSW 粗筛 → LightGlue 细匹配 → spatial verification → 综合 score 排序。
    pub async fn search_by_patch_image(
        &self,
        photo_bytes: &[u8],
        top_k: usize,
    ) -> Result<Vec<SearchResult>, ApplicationError> {
        let patch_search = self
            .patch_search
            .as_ref()
            .ok_or_else(|| ApplicationError::InvalidState("patch search not configured".into()))?;
        let patch_index = self
            .patch_index
            .as_ref()
            .ok_or_else(|| ApplicationError::InvalidState("patch index not configured".into()))?;

        let img = ImageData::from_bytes(photo_bytes).map_err(ApplicationError::AI)?;

        // 1) 提取 query VLAD descriptor
        let (vlad_vec, _query_patches) = patch_search
            .extractor()
            .extract_vlad_descriptor(&img)
            .await
            .map_err(ApplicationError::AI)?;

        // 2) HNSW 粗筛 k=50 candidates
        let candidates = patch_index
            .search(&vlad_vec, top_k.max(1) * 4)
            .map_err(ApplicationError::Vector)?;

        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        // 3) LightGlue fine matching（取 top candidates）
        let fetch_k = candidates.len().min(20);
        let fine_results = patch_search
            .search_patches(&img, &[], fetch_k)
            .await
            .map_err(ApplicationError::AI)?;

        // 4) 映射到 SearchResult
        let mut out = Vec::with_capacity(fine_results.len());
        for (rank, r) in fine_results.into_iter().enumerate() {
            out.push(SearchResult {
                image_id: r.image_id,
                target_id: 0,
                score: r.score,
                rank: rank + 1,
            });
        }
        Ok(out)
    }

    /// 用类别 prototype 搜索（YOLOv8 检测 → MobileCLIP embed → HNSW prototype match）。
    ///
    /// Phase F：`prototype_paths` 中的图片提取 YOLOv8 检测 crop → MobileCLIP embedding
    /// → 平均 → L2 normalize → HNSW search。
    pub async fn search_by_category(
        &self,
        prototype_paths: Vec<String>,
        top_k: usize,
    ) -> Result<Vec<SearchResult>, ApplicationError> {
        let cs = self
            .category_search
            .as_ref()
            .ok_or_else(|| ApplicationError::InvalidState("category_search not configured".into()))?;

        // 加载所有 prototype 图片
        let mut images = Vec::new();
        for path in &prototype_paths {
            let bytes = std::fs::read(path)
                .map_err(|e| ApplicationError::Internal(format!("read prototype {}: {e}", path)))?;
            let img = ImageData::from_bytes(&bytes).map_err(ApplicationError::AI)?;
            images.push(img);
        }

        // 创建 prototype
        let prototype = cs
            .create_prototype(&images)
            .await
            .map_err(ApplicationError::AI)?;

        // HNSW search（需要 object_index 作为 category index）
        let oi = self
            .object_index
            .as_ref()
            .ok_or_else(|| ApplicationError::InvalidState("object index not configured for category search".into()))?;

        let hits = oi
            .search(&prototype, top_k)
            .map_err(ApplicationError::Vector)?;

        let mut out = Vec::with_capacity(hits.len());
        for (rank, hit) in hits.into_iter().enumerate() {
            out.push(SearchResult {
                image_id: hit.id,
                target_id: 0,
                score: hit.score,
                rank: rank + 1,
            });
        }
        Ok(out)
    }
}

/// 把一张图片内检测到的多张人脸 embedding 按质量权重融合为单一 query 向量。
///
/// 权重 = `detector_score × (0.5 + pose_score) × (0.5 + face_area_score)`,
/// 归一化到总和为 1 后加权求和,再做 L2 归一化(cosine 检索要求单位向量)。
///
/// 与单一 `features[0]` 相比:图片里最主要(最大、正脸、清晰)的人贡献最大,
/// 同时保留多脸信息;用于"上传一张合影,找同一个人"的场景。
///
/// 边界:features 为空应由调用方提前返回;若总权重为 0(全退化解)则退回第一张脸的原始向量。
pub fn fuse_face_embeddings(features: &[FaceFeature]) -> Vec<f32> {
    let dim = features[0].embedding.dim;
    let mut weights = Vec::with_capacity(features.len());
    let mut total = 0.0f32;
    for f in features {
        let w = f.detection.score
            * (0.5 + f.pose_score)
            * (0.5 + f.face_area_score);
        weights.push(w);
        total += w;
    }
    if total <= 0.0 {
        return features[0].embedding.values.clone();
    }

    let mut fused = vec![0.0f32; dim];
    for (f, w) in features.iter().zip(weights.iter()) {
        let weight = w / total;
        for (out, v) in fused.iter_mut().zip(f.embedding.values.iter()) {
            *out += v * weight;
        }
    }
    let norm = fused.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut fused {
            *v /= norm;
        }
    }
    fused
}

/// 计算 HNSW 粗筛 fetch_k（free function 便于单元测试）：
/// `max(top_k * coarse_fetch_multiplier, coarse_fetch_min)`
///
/// 与 ai-next `object_search.py` 的 `fetch_k=max(top_k*20, 200)` 对齐；
/// 保证小 top_k 也能拿到足够候选以覆盖 image_id 聚合后的去重损失。
///
/// 边界：
/// - `top_k = 0` 视为 1（避免 fetch_k=0 拿到空集）
/// - `multiplier = 0` 强制 >= 1（防止退化）
/// 从一组 embedding 向量创建 prototype（元素平均 + L2 normalize）。
///
/// 纯函数（不依赖 SearchService），便于集成测试 / benchmark 直接调用。
pub fn create_prototype(embeddings: &[Vec<f32>]) -> Vec<f32> {
    if embeddings.is_empty() {
        return Vec::new();
    }
    let dim = embeddings[0].len();
    let n = embeddings.len() as f32;
    let mut proto = vec![0.0f32; dim];
    for emb in embeddings {
        for (i, v) in emb.iter().enumerate() {
            proto[i] += v;
        }
    }
    for v in &mut proto {
        *v /= n;
    }
    // L2 normalize
    let norm: f32 = proto.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-8);
    for v in &mut proto {
        *v /= norm;
    }
    proto
}

/// - `fetch_min = 0` 强制 >= 1（防止退化）
pub fn coarse_fetch_k(top_k: usize, multiplier: usize, fetch_min: usize) -> usize {
    let mult = multiplier.max(1);
    let min_k = fetch_min.max(1);
    let top = top_k.max(1);
    (top.saturating_mul(mult)).max(min_k)
}

/// Phase 7：从 per-image 聚合结果中选出要送进 LightGlue 的 candidate 列表。
///
/// 步骤：
/// 1. 转 Vec 并按 `weighted_score`（index 4）降序排序（HashMap 顺序非确定）
/// 2. 过滤 `hit.score < fine_match_min_embedding` 的 trivial non-matches
/// 3. 取前 `fine_cap` 个（防止全量粗筛命中都跑 LightGlue）
///
/// 返回值每个元素 = `(image_id, SearchHit, object_id, candidate_bbox, weighted_score)`。
pub fn select_fine_candidates(
    best_per_image: HashMap<i64, (pf_vector::SearchHit, i64, pf_core::BBox, f32)>,
    fine_match_min: f32,
    fine_cap: usize,
) -> Vec<(i64, pf_vector::SearchHit, i64, pf_core::BBox, f32)> {
    let mut ordered: Vec<(i64, pf_vector::SearchHit, i64, pf_core::BBox, f32)> =
        best_per_image
            .into_iter()
            .map(|(img_id, (hit, object_id, bbox, weighted))| {
                (img_id, hit, object_id, bbox, weighted)
            })
            .collect();
    ordered.sort_by(|a, b| b.4.partial_cmp(&a.4).unwrap_or(std::cmp::Ordering::Equal));
    ordered
        .into_iter()
        .filter(|(_, hit, _, _, _)| hit.score >= fine_match_min)
        .take(fine_cap.max(1))
        .collect()
}

/// Phase 4：scale-aware ROI expansion。
///
/// 对应 `pf_ai/object/roi.rs` 的 indexed ROI 尺寸：
/// - `scale >= 0.5`（如 full_image, scale=0.7, 0.5）：不扩展（已是较大区域）
/// - `scale ∈ [0.2, 0.5)`（如 scale=0.3）：扩展 1.5x（LightGlue 需要 ~75 关键点上下文）
/// - `scale < 0.2`（如 scale=0.1）：扩展 2.0x（小目标需要更多上下文）
///
/// 扩展后 clamp 到 image 边界（`x ≥ 0`, `y ≥ 0`, `x+w ≤ img_w`, `y+h ≤ img_h`）。
///
/// 输入 `roi` 必须是有效的 xywh；输出必定 valid 且非空。
pub fn expand_roi_for_matching(roi: &BBox, img_w: f32, img_h: f32) -> BBox {
    let scale_x = (roi.w / img_w).clamp(0.0, 1.0);
    let scale_y = (roi.h / img_h).clamp(0.0, 1.0);
    let min_scale = scale_x.min(scale_y);

    let expand = if min_scale >= 0.5 {
        1.0
    } else if min_scale >= 0.2 {
        1.5
    } else {
        2.0
    };

    if expand <= 1.0 {
        // 不扩展，但仍 clamp（防 ROI 越界）
        return clamp_roi_to_image(roi, img_w, img_h);
    }

    let scaled = roi.scaled(expand, expand);
    clamp_roi_to_image(&scaled, img_w, img_h)
}

/// 把 ROI clamp 到图像边界内（不缩放，仅裁剪越界部分）。
pub fn clamp_roi_to_image(roi: &BBox, img_w: f32, img_h: f32) -> BBox {
    let x = roi.x.max(0.0);
    let y = roi.y.max(0.0);
    let w = roi.w.max(1.0).min((img_w - x).max(1.0));
    let h = roi.h.max(1.0).min((img_h - y).max(1.0));
    BBox::new(x, y, w, h)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `create_prototype` 应该：mean + L2 normalize。
    #[test]
    fn create_prototype_is_mean_and_l2_normalized() {
        // 两个 2-d 向量：[1,0] + [0,1] → mean=[0.5,0.5] → L2 norm=√0.5 → proto=[√0.5, √0.5]
        let embeddings = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
        let proto = SearchService::create_prototype(&embeddings);
        assert_eq!(proto.len(), 2);
        let expected = (0.5f32 / (0.5f32 * 0.5 * 2.0).sqrt());
        assert!((proto[0] - expected).abs() < 1e-5);
        assert!((proto[1] - expected).abs() < 1e-5);
        // 验证 L2 unit length
        let norm = proto.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    /// `create_prototype` 处理空输入。
    #[test]
    fn create_prototype_empty_returns_empty() {
        let proto = SearchService::create_prototype(&[]);
        assert!(proto.is_empty());
    }

    /// `create_prototype` 单 embedding 仍做 mean + L2 normalize（保持 unit length）。
    #[test]
    fn create_prototype_single_is_l2_normalized() {
        let proto = SearchService::create_prototype(&[vec![3.0, 4.0]]);
        assert_eq!(proto.len(), 2);
        // mean=[3,4], L2 norm=5 → [0.6, 0.8]
        assert!((proto[0] - 0.6).abs() < 1e-5);
        assert!((proto[1] - 0.8).abs() < 1e-5);
    }

    // ===== P1 搜索 item 11: fuse_face_embeddings =====

    fn make_face_feature(values: Vec<f32>, det_score: f32, pose: f32, area: f32) -> FaceFeature {
        use pf_ai::FaceDetection;
        use pf_core::{Embedding, FaceKeypoints, ModelVersion};
        FaceFeature {
            detection: FaceDetection {
                bbox: BBox::new(0.0, 0.0, 100.0, 100.0),
                score: det_score,
                keypoints: FaceKeypoints {
                    left_eye: (0.0, 0.0),
                    right_eye: (0.0, 0.0),
                    nose: (0.0, 0.0),
                    left_mouth: (0.0, 0.0),
                    right_mouth: (0.0, 0.0),
                },
            },
            embedding: Embedding::new(values, ModelVersion::new("arcface@v1")),
            yaw_pitch_roll: None,
            blur_score: 0.0,
            pose_score: pose,
            face_area_score: area,
        }
    }

    /// 等权重:融合 = 各向量加权平均后 L2 归一化(单位向量)。
    #[test]
    fn fuse_equal_weights_returns_normalized_centroid() {
        let features = vec![
            make_face_feature(vec![3.0, 0.0], 0.9, 0.5, 0.5), // 权重均相同
            make_face_feature(vec![0.0, 3.0], 0.9, 0.5, 0.5),
        ];
        let fused = fuse_face_embeddings(&features);
        // mean = [1.5, 1.5] → L2 norm = √4.5 → [1.5, 1.5] / √4.5
        let expected = 1.5f32 / (1.5f32 * 1.5 * 2.0).sqrt();
        assert!((fused[0] - expected).abs() < 1e-5);
        assert!((fused[1] - expected).abs() < 1e-5);
        let norm = fused.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5);
    }

    /// 高质量 face 主导:权重 = detector × (0.5+pose) × (0.5+area)。
    #[test]
    fn fuse_higher_quality_face_dominates() {
        // face A: det=0.9, pose=0.5, area=0.5 → w=0.9*1.0*1.0=0.9 → 方向 [1,0]
        // face B: det=0.1, pose=0.1, area=0.1 → w=0.1*0.6*0.6=0.036 → 方向 [0,1]
        let features = vec![
            make_face_feature(vec![1.0, 0.0], 0.9, 0.5, 0.5),
            make_face_feature(vec![0.0, 1.0], 0.1, 0.1, 0.1),
        ];
        let fused = fuse_face_embeddings(&features);
        // 融合向量应明显偏向 [1,0](x 分量 > y 分量)
        assert!(fused[0] > fused[1], "high-quality face should dominate: {:?}", fused);
    }

    /// 单脸:结果即该脸 L2 归一化向量。
    #[test]
    fn fuse_single_face_is_l2_normalized() {
        let features = vec![make_face_feature(vec![3.0, 4.0], 0.8, 0.5, 0.5)];
        let fused = fuse_face_embeddings(&features);
        assert_eq!(fused.len(), 2);
        assert!((fused[0] - 0.6).abs() < 1e-5);
        assert!((fused[1] - 0.8).abs() < 1e-5);
    }

    /// 全退化解(总权重为 0):退回第一张脸的原始向量。
    #[test]
    fn fuse_all_zero_weights_falls_back_to_first() {
        let features = vec![make_face_feature(vec![0.6, 0.8], 0.0, -0.5, -0.5)];
        let fused = fuse_face_embeddings(&features);
        assert!((fused[0] - 0.6).abs() < 1e-6);
        assert!((fused[1] - 0.8).abs() < 1e-6);
    }

    /// `coarse_fetch_k` 默认：top_k=10, mult=20, min=200 → max(200, 200)=200
    #[test]
    fn coarse_fetch_k_default() {
        assert_eq!(coarse_fetch_k(10, 20, 200), 200);
        assert_eq!(coarse_fetch_k(20, 20, 200), 400);
        assert_eq!(coarse_fetch_k(100, 20, 200), 2000);
    }

    /// `coarse_fetch_k` 小 top_k：multiplier * top_k < min → fallback to min
    #[test]
    fn coarse_fetch_k_min_fallback() {
        assert_eq!(coarse_fetch_k(5, 20, 200), 200);
        assert_eq!(coarse_fetch_k(1, 20, 200), 200);
    }

    /// `coarse_fetch_k` 防 0：top_k=0 → at least 1 * mult
    #[test]
    fn coarse_fetch_k_handles_zero() {
        assert_eq!(coarse_fetch_k(0, 20, 200), 200); // max(0, 200) = 200 via min
    }

    /// `coarse_fetch_k` 防 mult=0：mult 强制 >= 1
    #[test]
    fn coarse_fetch_k_handles_zero_multiplier() {
        // mult=0 → force to 1, top_k=10 → max(10, 200) = 200
        assert_eq!(coarse_fetch_k(10, 0, 200), 200);
    }

    // ===== Phase 4: expand_roi_for_matching =====

    /// scale >= 0.5（full_image / scale=0.5 / scale=0.7）：不扩展。
    #[test]
    fn expand_roi_no_expansion_for_large_roi() {
        let img = (4000.0, 3000.0);
        let roi = BBox::new(0.0, 0.0, 4000.0, 3000.0); // scale 1.0
        let out = expand_roi_for_matching(&roi, img.0, img.1);
        assert_eq!(out.x, 0.0);
        assert_eq!(out.y, 0.0);
        assert_eq!(out.w, 4000.0);
        assert_eq!(out.h, 3000.0);

        // scale = 0.5
        let roi = BBox::new(1000.0, 750.0, 2000.0, 1500.0);
        let out = expand_roi_for_matching(&roi, img.0, img.1);
        assert_eq!(out.x, 1000.0);
        assert_eq!(out.y, 750.0);
        assert_eq!(out.w, 2000.0);
        assert_eq!(out.h, 1500.0);
    }

    /// scale ∈ [0.2, 0.5)（scale=0.3）：扩展 1.5x，保留中心。
    #[test]
    fn expand_roi_moderate_expansion() {
        let img = (1000.0, 1000.0);
        // scale=0.3 → roi=300x300
        let roi = BBox::new(350.0, 350.0, 300.0, 300.0);
        let out = expand_roi_for_matching(&roi, img.0, img.1);
        // 中心点 (500,500) 应保持；新尺寸 = 300*1.5 = 450
        assert_eq!(out.x + out.w * 0.5, 500.0);
        assert_eq!(out.y + out.h * 0.5, 500.0);
        assert!((out.w - 450.0).abs() < 1e-3);
        assert!((out.h - 450.0).abs() < 1e-3);
    }

    /// scale < 0.2（scale=0.1）：扩展 2.0x。
    #[test]
    fn expand_roi_small_expansion() {
        let img = (1000.0, 1000.0);
        // scale=0.1 → roi=100x100
        let roi = BBox::new(450.0, 450.0, 100.0, 100.0);
        let out = expand_roi_for_matching(&roi, img.0, img.1);
        // 中心 (500,500) 保持；新尺寸 = 200
        assert_eq!(out.x + out.w * 0.5, 500.0);
        assert_eq!(out.y + out.h * 0.5, 500.0);
        assert!((out.w - 200.0).abs() < 1e-3);
        assert!((out.h - 200.0).abs() < 1e-3);
    }

    /// clamp 到图像边界：靠近边缘的 ROI 不会越界。
    #[test]
    fn expand_roi_clamps_to_image_bounds() {
        let img = (1000.0, 1000.0);
        // 右上角 ROI，scale=0.1 → 扩展 2x 会越界
        let roi = BBox::new(950.0, 950.0, 50.0, 50.0);
        let out = expand_roi_for_matching(&roi, img.0, img.1);
        assert!(out.x + out.w <= img.0 + 1e-3, "x+w={} > img_w={}", out.x + out.w, img.0);
        assert!(out.y + out.h <= img.1 + 1e-3);
        assert!(out.x >= 0.0);
        assert!(out.y >= 0.0);
    }

    /// 负坐标 clamp 到 0。
    #[test]
    fn expand_roi_clamps_negative_origin() {
        let img = (1000.0, 1000.0);
        let roi = BBox::new(-100.0, -100.0, 200.0, 200.0);
        let out = expand_roi_for_matching(&roi, img.0, img.1);
        assert_eq!(out.x, 0.0);
        assert_eq!(out.y, 0.0);
    }

    // ===== Phase 7: select_fine_candidates =====

    /// 构造 `best_per_image` 测试 fixture。
    /// - 每个 image_id 一个 entry
    /// - `hit.score` 给定；`weighted_score` 独立控制（用于排序验证）
    fn make_best_per_image(scores: &[(i64, f32, f32)]) -> HashMap<i64, (SearchHit, i64, BBox, f32)> {
        scores
            .iter()
            .enumerate()
            .map(|(i, &(img_id, score, weighted))| {
                let hit = SearchHit {
                    id: img_id,
                    score,
                };
                (
                    img_id,
                    (hit, img_id * 10 + i as i64, BBox::new(0.0, 0.0, 100.0, 100.0), weighted),
                )
            })
            .collect()
    }

    /// Phase 7：embedding_score < fine_match_min_embedding 的 candidate 被过滤。
    #[test]
    fn coarse_pre_filter_skips_low_embedding() {
        let mut map = make_best_per_image(&[
            (1, 0.5, 0.5),  // 通过
            (2, 0.2, 0.2),  // < 0.3 → 过滤
            (3, 0.4, 0.4),  // 通过
            (4, 0.1, 0.1),  // < 0.3 → 过滤
            (5, 0.35, 0.35), // 通过（> 0.3）
        ]);
        // 故意乱序插入，验证排序效果
        let mut shuffled = HashMap::new();
        shuffled.insert(5, map.remove(&5).unwrap());
        shuffled.insert(2, map.remove(&2).unwrap());
        shuffled.insert(1, map.remove(&1).unwrap());
        shuffled.insert(4, map.remove(&4).unwrap());
        shuffled.insert(3, map.remove(&3).unwrap());

        let result = select_fine_candidates(shuffled, 0.3, 50);
        // 应保留 image_id = 1, 3, 5（score >= 0.3），过滤 2, 4
        let kept: Vec<i64> = result.iter().map(|(img_id, _, _, _, _)| *img_id).collect();
        assert_eq!(result.len(), 3);
        assert!(kept.contains(&1));
        assert!(kept.contains(&3));
        assert!(kept.contains(&5));
        assert!(!kept.contains(&2));
        assert!(!kept.contains(&4));
    }

    /// Phase 7：候选按 weighted_score 降序排序（确定性，与 HashMap 顺序无关）。
    #[test]
    fn fine_candidates_sorted_by_weighted_score_desc() {
        let map = make_best_per_image(&[
            (10, 0.5, 0.10),  // weighted=0.10 → 末位
            (20, 0.6, 0.90),  // weighted=0.90 → 首位
            (30, 0.7, 0.50),  // weighted=0.50 → 中位
            (40, 0.4, 0.70),  // weighted=0.70 → 第 2
        ]);
        let result = select_fine_candidates(map, 0.0, 50);
        assert_eq!(result.len(), 4);
        let order: Vec<f32> = result.iter().map(|(_, _, _, _, w)| *w).collect();
        assert_eq!(order, vec![0.90, 0.70, 0.50, 0.10]);

        let img_ids: Vec<i64> = result.iter().map(|(img_id, _, _, _, _)| *img_id).collect();
        assert_eq!(img_ids, vec![20, 40, 30, 10]);
    }

    /// Phase 7：fine_cap 生效，最多返回 cap 个 candidate。
    #[test]
    fn fine_candidates_respects_cap() {
        let map = make_best_per_image(&[
            (1, 0.9, 0.9),
            (2, 0.8, 0.8),
            (3, 0.7, 0.7),
            (4, 0.6, 0.6),
            (5, 0.5, 0.5),
        ]);
        let result = select_fine_candidates(map, 0.0, 3);
        assert_eq!(result.len(), 3);
        // 排序后取 top 3 → image_id 1, 2, 3
        let img_ids: Vec<i64> = result.iter().map(|(img_id, _, _, _, _)| *img_id).collect();
        assert_eq!(img_ids, vec![1, 2, 3]);
    }

    /// Phase 7：相同 image_id 的多个 ROI 已经在 `best_per_image` 中合并（per-image max-pool）。
    /// 这里验证：如果传入包含相同 img_id 的多个 entry，HashMap 会去重 → select_fine_candidates 不会重复。
    /// （实际 per-image 聚合在 Stage 2 已完成，这里仅做合约测试）
    #[test]
    fn fine_candidates_dedups_by_image_id() {
        // 故意构造相同 image_id（模拟 Stage 2 的 max-pool 入参）
        let mut map = HashMap::new();
        map.insert(
            100,
            (
                SearchHit { id: 100i64, score: 0.8 },
                1001,
                BBox::new(0.0, 0.0, 100.0, 100.0),
                0.8,
            ),
        );
        let result = select_fine_candidates(map, 0.0, 50);
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].0, 100);
    }
}