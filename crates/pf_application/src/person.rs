//! PersonService — 人脸聚类 + person 命名。
//!
//! 算法演进：
//! - Phase 2：centroid-based — 每 person 一个均值向量,与未聚类 face cosine 比较。
//! - Phase 3：引入 PrototypeService,每个 person 维护多 prototype(frontal/left_profile/...)。
//! - Phase 4：assign_or_create 改用 k-NN 候选 + prototype 验证。
//!   流程见 [`PersonService::assign_or_create`] 的 doc — 解决了"两张脸碰巧接近
//!   但属于不同人"的错并问题。
//!
//! Threshold：仍是 `policy.similarity_threshold = 0.65`(Phase 2 起,未改)。

use std::collections::HashMap;
use std::sync::Arc;

use pf_core::{Clusterer, Embedding, PersonAssignment, FACE_MODEL_NAME};
use pf_database::{
    Database, MatchStatus, NewFacePersonAssignment, NewPerson, NewShadowRecord,
};
use pf_vector::VectorIndex;
use tracing::{debug, info, warn};

use crate::cluster_v2::best_prototype_match;
use crate::error::ApplicationError;
use crate::identity_evidence::{
    CandidateResult, FaceEvidence, FaceQuality, IdentityDecision, IdentityExecutionMode,
    IdentityPipeline, IdentityQuery, ShadowDisagreementType, ShadowExecutionResult, ShadowSkipReason,
};

/// 聚类策略。
#[derive(Debug, Clone)]
pub struct ClusterPolicy {
    /// cosine 相似度阈值（>= 即视为同人）
    pub similarity_threshold: f32,
    /// 防 chaining：top-2 候选 prototype 分差 < 该 margin 时视为歧义,
    /// 创建新 person 而不是并入较近者（避免通过中间脸把两个不同人链式合并）
    pub chaining_margin: f32,
    /// 每个 person 最多保留多少代表 face（默认 5）
    pub max_representatives: usize,
    /// 单次最多取的 top_k
    pub search_top_k: usize,
}

impl Default for ClusterPolicy {
    fn default() -> Self {
        // 0.75：基于 stock 副本阈值扫描定标。错并(不同人)分∈[0.65,0.70),
        // 真实近似图(同人连拍)分≥0.85,空隙正中 0.75。LFW 在 0.65 处 recall 已≈0,
        // 调高无额外代价。（原 0.65 会让 asian-man+young-asian-girl+cottonbro 错并）
        Self {
            similarity_threshold: 0.75,
            chaining_margin: 0.05,
            max_representatives: 5,
            search_top_k: 20,
        }
    }
}

/// 聚类状态：每个 person 保留若干代表 face（其 embedding）。
pub struct ClusterState {
    /// person_id → 代表 face 的 embedding 平均向量
    person_centroids: HashMap<i64, Vec<f32>>,
}

impl ClusterState {
    /// 空状态。
    pub fn new() -> Self {
        Self {
            person_centroids: HashMap::new(),
        }
    }

    /// 加入一个人的初始 centroid。
    pub fn add_centroid(&mut self, person_id: i64, emb: &[f32]) {
        self.person_centroids.insert(person_id, emb.to_vec());
    }

    /// 更新 centroid：和已有 centroid 平均。
    pub fn merge_centroid(&mut self, person_id: i64, emb: &[f32]) {
        if let Some(c) = self.person_centroids.get_mut(&person_id) {
            let n = c.len().min(emb.len());
            for i in 0..n {
                c[i] = (c[i] + emb[i]) * 0.5;
            }
            // 重新归一化
            let norm: f32 = c.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            for x in c.iter_mut() {
                *x /= norm;
            }
        } else {
            self.add_centroid(person_id, emb);
        }
    }

    /// 与所有人比 cosine 相似度，返回最佳匹配。
    pub fn best_match(&self, emb: &[f32]) -> Option<(i64, f32)> {
        let mut best: Option<(i64, f32)> = None;
        for (pid, centroid) in &self.person_centroids {
            let score = cosine_similarity(emb, centroid);
            if best.map(|(_, s)| score > s).unwrap_or(true) {
                best = Some((*pid, score));
            }
        }
        best
    }
}

/// PersonService。
pub struct PersonService {
    db: Arc<Database>,
    face_index: Arc<dyn VectorIndex>,
    policy: ClusterPolicy,
    state: parking_lot::Mutex<ClusterState>,
    /// Phase 3:prototype 重建器
    prototype_service: Option<Arc<crate::prototype_service::PrototypeService>>,
    /// Phase 30: Shadow Mode - Identity Pipeline
    identity_pipeline: Option<Arc<IdentityPipeline>>,
    /// Phase 30: Execution mode (Legacy/Shadow/NewPipeline)
    execution_mode: IdentityExecutionMode,
}

impl PersonService {
    /// 构造。
    pub fn new(db: Arc<Database>, face_index: Arc<dyn VectorIndex>, policy: ClusterPolicy) -> Self {
        Self {
            db,
            face_index,
            policy,
            state: parking_lot::Mutex::new(ClusterState::new()),
            prototype_service: None,
            identity_pipeline: None,
            execution_mode: IdentityExecutionMode::Legacy,
        }
    }

    /// 注入 PrototypeService(Phase 3)。
    pub fn with_prototype_service(
        mut self,
        ps: Arc<crate::prototype_service::PrototypeService>,
    ) -> Self {
        self.prototype_service = Some(ps);
        self
    }

    /// 注入 IdentityPipeline for Shadow Mode (Phase 30 T11)
    pub fn with_identity_pipeline(
        mut self,
        pipeline: Arc<IdentityPipeline>,
    ) -> Self {
        self.identity_pipeline = Some(pipeline);
        self
    }

    /// 设置执行模式 (Phase 30 T11)
    pub fn with_execution_mode(
        mut self,
        mode: IdentityExecutionMode,
    ) -> Self {
        self.execution_mode = mode;
        self
    }

    /// 从所有现存 person 重建 cluster state。
    ///
    /// Phase 2：每个 person 取 highest-quality face 的 embedding 作为 centroid。
    /// 这样已有聚类结果不会被 cluster_all 重新打散。
    pub fn rebuild_state(&self) -> Result<(), ApplicationError> {
        let persons: Vec<pf_database::PersonRow> = self
            .db
            .transaction(|tx| tx.persons().list())
            .map_err(|e| ApplicationError::Internal(format!("rebuild: {e}")))?;
        let mut state = self.state.lock();
        state.person_centroids.clear();
        for p in persons {
            // 取该 person quality 最高的 face 的 embedding
            let emb = self
                .db
                .transaction(|tx| {
                    let faces = tx.faces().list_by_person(p.id)?;
                    let first = faces.into_iter().next();
                    match first {
                        Some(f) => tx.faces().get_embedding(f.id, FACE_MODEL_NAME),
                        None => Ok(None),
                    }
                })
                .map_err(|e| ApplicationError::Internal(format!("get_embedding: {e}")))?;
            if let Some(emb) = emb {
                state.add_centroid(p.id, &emb);
            }
        }
        Ok(())
    }

    /// 处理一个未聚类 face：找到最佳 person 或建议新建。
    pub fn assign(&self, face_vector_id: i64, embedding: &[f32]) -> PersonAssignment {
        let _ = face_vector_id;
        let state = self.state.lock();
        if let Some((pid, score)) = state.best_match(embedding) {
            if score >= self.policy.similarity_threshold {
                return PersonAssignment::Existing(pid);
            }
        }
        PersonAssignment::New { person_id: 0 }
    }

    /// 分配或新建一个 person（Phase 4 V2：k-NN 候选 + prototype 验证）。
    ///
    /// 流程：
    /// 1. k-NN search top_k → 候选 face 集合
    /// 2. 每个候选 face → person_id → 投票(cum_score, top_score)
    /// 3. 取 top 3 个候选(按 cum_score),逐个用 prototype 验证:
    ///    score = max cosine(query, that_person.prototype_embeddings)
    ///    若该 person 还没有 prototype → fallback 到 top neighbor score
    /// 4. 防 chaining:top-2 候选 prototype 分差 < `chaining_margin` → 歧义 → New
    /// 5. 选 prototype score 最高的候选;若 >= threshold → Existing;
    ///    否则 → New
    ///
    /// Phase 3 引入 prototype 后,验证不再只看"top neighbor 相似度",
    /// 而是用 person 的 canonical prototype 表征做 cosine 匹配 —
    /// 解决了"两张脸碰巧接近但属于不同人"被错并的问题。
    ///
    /// Threshold 没变(仍是 `policy.similarity_threshold` = 0.65)。
    /// 返回 `(assignment, best_score)` —— score 供 `assign_face` 记录到
    /// `face_person_assignments`(item 17/18)。
    pub fn assign_or_create(&self, embedding: &[f32]) -> Result<PersonAssignment, ApplicationError> {
        Ok(self.decide_assignment(embedding)?.0)
    }

    /// 决策核心：返回 `(assignment, best_prototype_score, margin, candidate_count)`。
    ///
    /// `best_prototype_score` 为选中 candidate 的 prototype 验证分数
    /// （New 时或歧义时为 None），margin 是 top-1 与 top-2 的分差（None 如果 <2 个候选）。
    /// `candidate_count` 是参与投票的不同 person 数量。
    /// 供调用方持久化 assignment 用。
    fn decide_assignment(
        &self,
        embedding: &[f32],
    ) -> Result<(PersonAssignment, Option<f32>, Option<f32>, usize), ApplicationError> {
        let k = self.policy.search_top_k.min(10);

        // 1) k-NN: candidate discovery
        let hits = self
            .face_index
            .search(embedding, k)
            .map_err(|e| ApplicationError::Internal(format!("face_index search: {e}")))?;

        // 2) Per-person vote aggregation
        let mut votes: HashMap<i64, (f32, f32)> = HashMap::new(); // pid → (cum_score, top_score)
        for hit in &hits {
            if let Some(face) = self
                .db
                .transaction(|tx| tx.faces().get_by_vector_id(hit.id))
                .map_err(|e| ApplicationError::Internal(format!("get_by_vector_id: {e}")))?
            {
                if let Some(pid) = face.person_id {
                    let entry = votes.entry(pid).or_insert((0.0, 0.0));
                    entry.0 += hit.score;
                    entry.1 = entry.1.max(hit.score);
                }
            }
        }

        let candidate_count = votes.len();

        // 3) Prototype verification: top-3 candidates by cumulative votes
        if !votes.is_empty() {
            let mut sorted: Vec<(i64, f32, f32)> = votes
                .into_iter()
                .map(|(pid, (cum, top))| (pid, cum, top))
                .collect();
            sorted.sort_by(|a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut scored: Vec<(i64, f32)> = sorted
                .iter()
                .take(3)
                .map(|(pid, _cum, top_score)| (*pid, self.prototype_score(*pid, embedding, *top_score)))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Compute margin: top-1 - top-2
            let margin = if scored.len() >= 2 {
                Some(scored[0].1 - scored[1].1)
            } else {
                None
            };

            if let Some(&(pid, score)) = scored.first() {
                // 4) 防 chaining：top-2 分差 < margin → 歧义,不并入任何一人,
                //    宁可新建 person(避免通过中间脸链式错并)
                let ambiguous = scored.len() >= 2
                    && (scored[0].1 - scored[1].1).abs() < self.policy.chaining_margin;
                if score >= self.policy.similarity_threshold && !ambiguous {
                    self.state.lock().merge_centroid(pid, embedding);
                    return Ok((PersonAssignment::Existing(pid), Some(score), margin, candidate_count));
                }
            }
        }

        // 5) Create new person
        let pid: i64 = self
            .db
            .transaction(|tx| {
                let mut repo = tx.persons();
                repo.insert(&NewPerson { name: None })
            })
            .map_err(|e| ApplicationError::Internal(format!("create person: {e}")))?;
        self.state.lock().add_centroid(pid, embedding);
        Ok((PersonAssignment::New { person_id: pid }, None, None, candidate_count))
    }

    /// 分配一个已入库的 face（`cluster_all` 使用）:决策 + 把结果记录到
    /// `face_person_assignments`(item 17 — Confirmed/Candidate 状态)。
    ///
    /// - Existing：写入 Confirmed assignment
    /// - New：不写(没有明确归属);调用方如需要可另记 Rejected
    pub fn assign_face(
        &self,
        face_id: i64,
        embedding: &[f32],
        quality: Option<f32>,
        pose: Option<f32>,
    ) -> Result<PersonAssignment, ApplicationError> {
        let (assignment, score, _margin, _candidate_count) = self.decide_assignment(embedding)?;
        match assignment {
            PersonAssignment::Existing(pid) => {
                if let Some(sim) = score {
                    self.record_assignment(
                        face_id,
                        pid,
                        sim,
                        quality,
                        pose,
                        MatchStatus::Confirmed,
                    )?;
                }
                Ok(PersonAssignment::Existing(pid))
            }
            PersonAssignment::New { person_id } => Ok(PersonAssignment::New { person_id }),
        }
    }

    /// Phase 30 T11: 分配 face 并运行 Shadow 评估
    ///
    /// 与 `assign_face` 相同，但在 Legacy 决策后运行 Shadow 评估。
    /// Shadow 评估是 Best-Effort，不会影响 Legacy 决策或返回结果。
    ///
    /// `image_id` 是必需的，因为 ShadowRecord 需要它。
    pub fn assign_face_with_shadow(
        &self,
        face_id: i64,
        image_id: i64,
        embedding: &[f32],
        quality: Option<f32>,
        pose: Option<f32>,
        yaw: f32,
        pitch: f32,
        roll: f32,
        blur_score: f32,
        face_size: f32,
    ) -> Result<PersonAssignment, ApplicationError> {
        // Phase 36: 根据 execution_mode 决定使用哪个 pipeline
        match self.execution_mode {
            IdentityExecutionMode::NewPipeline => {
                // NewPipeline: 使用 IdentityPipeline 决策（Phase 36 新逻辑）
                self.assign_face_with_identity_pipeline(face_id, image_id, embedding, quality, pose, yaw, pitch, roll, blur_score, face_size)
            }
            IdentityExecutionMode::Legacy => {
                // Legacy: 使用旧的 decide_assignment，不运行 shadow
                let (assignment, legacy_score, _, _) = self.decide_assignment(embedding)?;
                if let PersonAssignment::Existing(pid) = assignment {
                    if let Some(sim) = legacy_score {
                        self.record_assignment(face_id, pid, sim, quality, pose, MatchStatus::Confirmed)?;
                    }
                }
                Ok(assignment)
            }
            IdentityExecutionMode::Shadow => {
                // Shadow: Legacy 决策 + Shadow 评估（用于对比验证）
                self.assign_face_shadow_mode(face_id, image_id, embedding, quality, pose)
            }
        }
    }

    /// Phase 36: 使用 IdentityPipeline 的新决策逻辑
    fn assign_face_with_identity_pipeline(
        &self,
        face_id: i64,
        image_id: i64,
        embedding: &[f32],
        quality: Option<f32>,
        pose: Option<f32>,
        yaw: f32,
        pitch: f32,
        roll: f32,
        blur_score: f32,
        face_size: f32,
    ) -> Result<PersonAssignment, ApplicationError> {
        let pipeline = match self.identity_pipeline.as_ref() {
            Some(p) => p,
            None => {
                // 如果没有 IdentityPipeline，回退到 Legacy
                warn!("IdentityPipeline not available, falling back to Legacy");
                let (assignment, legacy_score, _, _) = self.decide_assignment(embedding)?;
                match assignment {
                    PersonAssignment::Existing(pid) => {
                        if let Some(sim) = legacy_score {
                            self.record_assignment(face_id, pid, sim, quality, pose, MatchStatus::Confirmed)?;
                        }
                    }
                    PersonAssignment::New { person_id } => {
                        // Legacy path: 直接设置 person_id，不走 create_and_assign
                        // (Legacy 不记录 face_person_assignments for New)
                        self.db.transaction(|tx| tx.faces().set_person(face_id, Some(person_id)))
                            .map_err(|e| ApplicationError::Internal(format!("set_person: {e}")))?;
                    }
                }
                return Ok(assignment);
            }
        };

        // 构建 FaceQuality（从 FaceRow 的质量字段）
        let face_quality = FaceQuality {
            face_size,
            detection_score: quality.unwrap_or(0.5),
            alignment_score: pose.unwrap_or(0.5),
            yaw,
            pitch,
            roll,
            blur_score,
        };

        let query = IdentityQuery {
            face_id,
            image_id,
            face_embedding: embedding.to_vec(),
            face_quality,
            body_embedding: None,
            body_quality: 0.0,
            body_available: false,
        };

        let result = pipeline.evaluate(&query);

        // 根据决策决定 assignment
        match result.shadow_decision {
            IdentityDecision::ConfidentMatch | IdentityDecision::WeakMatch => {
                if let Some(candidate) = result.shadow_candidate {
                    // 分配给已有 person
                    let pid = candidate.person_id;
                    let score = candidate.face_score;
                    self.record_assignment(face_id, pid, score, quality, pose, MatchStatus::Confirmed)?;
                    Ok(PersonAssignment::Existing(pid))
                } else {
                    // 没有 candidate → 创建新 person
                    self.create_and_assign(face_id, embedding)
                }
            }
            IdentityDecision::Ambiguous | IdentityDecision::NewPerson => {
                // UNKNOWN/REVIEW: 不分配，创建新 person 但标记为 pending
                self.create_and_assign(face_id, embedding)
            }
            IdentityDecision::StrongMatchWithCompetitor | IdentityDecision::Unreliable | IdentityDecision::SupportedMatch => {
                // 这些变体暂不处理，视为需要人工审核
                self.create_and_assign(face_id, embedding)
            }
        }
    }

    /// Legacy + Shadow 评估模式（用于验证对比）
    fn assign_face_shadow_mode(
        &self,
        face_id: i64,
        image_id: i64,
        embedding: &[f32],
        quality: Option<f32>,
        pose: Option<f32>,
    ) -> Result<PersonAssignment, ApplicationError> {
        // 1. Legacy 决策（这可能会改变 state）
        let (assignment, legacy_score, legacy_margin, candidate_count) =
            self.decide_assignment(embedding)?;

        // Record assignment if Existing
        if let PersonAssignment::Existing(pid) = assignment {
            if let Some(sim) = legacy_score {
                self.record_assignment(
                    face_id,
                    pid,
                    sim,
                    quality,
                    pose,
                    MatchStatus::Confirmed,
                )?;
            }
        }

        // 2. Shadow 评估（在 Legacy 决策之后，但不影响它）
        let face_quality = FaceQuality {
            face_size: 0.0,
            detection_score: quality.unwrap_or(0.5),
            alignment_score: pose.unwrap_or(0.5),
            yaw: 0.0,
            pitch: 0.0,
            roll: 0.0,
            blur_score: 0.5,
        };

        let _shadow_result = self.evaluate_shadow(
            face_id,
            image_id,
            embedding,
            face_quality,
            &assignment,
            legacy_score,
            legacy_margin,
            candidate_count,
        );

        // 3. 返回 Legacy 结果（Shadow 不影响）
        Ok(assignment)
    }

    /// 创建新 person 并分配
    fn create_and_assign(
        &self,
        face_id: i64,
        _embedding: &[f32],
    ) -> Result<PersonAssignment, ApplicationError> {
        let person_id = self.db.transaction(|tx| tx.persons().insert(&NewPerson { name: None }))
            .map_err(|e| ApplicationError::Internal(format!("create person: {e}")))?;
        self.record_assignment(face_id, person_id, 1.0, None, None, MatchStatus::Confirmed)?;
        self.db.transaction(|tx| tx.faces().set_person(face_id, Some(person_id)))
            .map_err(|e| ApplicationError::Internal(format!("set_person: {e}")))?;
        Ok(PersonAssignment::New { person_id })
    }

    /// 记录 face→person 的 assignment 行（face_person_assignments）。
    ///
    /// `confidence` 由 `assignment_confidence(similarity, quality, pose)` 计算(item 18)。
    /// `method` 记为 `"prototype_max"`(Phase 4 决策方式),便于追溯。
    fn record_assignment(
        &self,
        face_id: i64,
        person_id: i64,
        similarity: f32,
        quality: Option<f32>,
        pose: Option<f32>,
        status: MatchStatus,
    ) -> Result<(), ApplicationError> {
        let confidence = assignment_confidence(similarity, quality, pose);
        let m = NewFacePersonAssignment {
            face_id,
            person_id,
            similarity,
            confidence,
            method: "prototype_max".to_string(),
            status,
            model_version: "arcface@v1".to_string(),
        };
        self.db
            .transaction(|tx| tx.face_person_assignments().insert(&m))
            .map_err(|e| ApplicationError::Internal(format!("record_assignment: {e}")))?;
        Ok(())
    }

    /// 计算 person 的 prototype 验证分数。
    ///
    /// - 若 prototype_service 已注入且该 person 有 prototype → max cosine(query, prototype)
    /// - 否则(prototype_service 缺失 / 该 person 暂无 prototype) → fallback `top_neighbor_score`
    fn prototype_score(
        &self,
        person_id: i64,
        embedding: &[f32],
        top_neighbor_score: f32,
    ) -> f32 {
        let Some(ps) = self.prototype_service.as_ref() else {
            return top_neighbor_score;
        };
        match ps.list_for_person(person_id, None) {
            Ok(protos) if !protos.is_empty() => {
                let protos_vec: Vec<Vec<f32>> =
                    protos.into_iter().map(|(_, v)| v).collect();
                let one = vec![(person_id, protos_vec)];
                best_prototype_match(embedding, &one)
                    .map(|(_, s)| s)
                    .unwrap_or(0.0)
            }
            _ => top_neighbor_score,
        }
    }

    /// 列出所有 person。
    pub fn list_all(&self) -> Result<Vec<pf_database::PersonRow>, ApplicationError> {
        let persons = self
            .db
            .transaction(|tx| tx.persons().list())
            .map_err(|e| ApplicationError::Internal(format!("list persons: {e}")))?;
        Ok(persons)
    }

    /// 给 person 命名。
    pub fn rename(&self, person_id: i64, name: Option<&str>) -> Result<(), ApplicationError> {
        self.db
            .transaction(|tx| tx.persons().set_name(person_id, name))
            .map_err(|e| ApplicationError::Internal(format!("rename: {e}")))?;
        Ok(())
    }

    /// 列出某 person 的所有 face（按 quality DESC）。
    pub fn faces_of(
        &self,
        person_id: i64,
    ) -> Result<Vec<pf_database::FaceRow>, ApplicationError> {
        let faces = self
            .db
            .transaction(|tx| tx.faces().list_by_person(person_id))
            .map_err(|e| ApplicationError::Internal(format!("list faces: {e}")))?;
        Ok(faces)
    }

    /// 同步 person.face_count。
    pub fn refresh_count(&self, person_id: i64) -> Result<i64, ApplicationError> {
        let n = self
            .db
            .transaction(|tx| tx.persons().refresh_face_count(person_id))
            .map_err(|e| ApplicationError::Internal(format!("refresh: {e}")))?;
        Ok(n)
    }

    /// 全量聚类所有未分配 face。
    ///
    /// **Phase 2 修复**：从 `face_embeddings` 表读真实 ArcFace embeddings（不再用
    /// `vec![detector_score; 512]` 凑数）。每个未聚类 face 按 quality DESC 顺序：
    /// 1. 与已有 person 的 centroid 比 cosine（>= `policy.similarity_threshold` 归入）
    /// 2. 否则创建新 person
    pub async fn cluster_all(&self) -> Result<ClusterSummary, ApplicationError> {
        self.rebuild_state()?;
        // 1) 拉所有未聚类 face 的真实 embedding
        let entries = self
            .db
            .transaction(|tx| tx.faces().list_unassigned_embeddings(FACE_MODEL_NAME))
            .map_err(|e| ApplicationError::Internal(format!("list_unassigned_embeddings: {e}")))?;
        info!(count = entries.len(), "cluster_all started");

        let mut summary = ClusterSummary::default();
        for (face_id, image_id, embedding) in entries {
            // 拉 face 行拿 quality/pose/yaw/pitch/roll/blur_score 和 face_size
            let (quality, pose, yaw, pitch, roll, blur_score, face_size) = self
                .db
                .transaction(|tx| tx.faces().get_by_id(face_id))
                .map_err(|e| ApplicationError::Internal(format!("get_by_id: {e}")))?
                .map(|f| {
                    let face_size = f.bbox.w.min(f.bbox.h);
                    (f.quality, f.pose_score, f.yaw.unwrap_or(0.0), f.pitch.unwrap_or(0.0), f.roll.unwrap_or(0.0), f.blur_score.unwrap_or(0.5), face_size)
                })
                .unwrap_or((0.0, None, 0.0, 0.0, 0.0, 0.5, 0.0));

            let affected_pid = match self.assign_face_with_shadow(face_id, image_id, &embedding, Some(quality), pose, yaw, pitch, roll, blur_score, face_size)? {
                PersonAssignment::Existing(pid) => {
                    self.db
                        .transaction(|tx| tx.faces().set_person(face_id, Some(pid)))
                        .map_err(|e| ApplicationError::Internal(format!("set_person: {e}")))?;
                    summary.assigned += 1;
                    Some(pid)
                }
                PersonAssignment::New { person_id } => {
                    self.db
                        .transaction(|tx| tx.faces().set_person(face_id, Some(person_id)))
                        .map_err(|e| ApplicationError::Internal(format!("set_person: {e}")))?;
                    summary.created += 1;
                    Some(person_id)
                }
            };
            let _ = image_id;

            // Phase 3: 每个被影响的 person 重建 prototypes。
            // 失败仅 warn,不阻塞 cluster 流程(下次 cluster_all 会重试)。
            if let (Some(pid), Some(ps)) = (affected_pid, &self.prototype_service) {
                if let Err(e) = ps.rebuild_for_person(pid) {
                    warn!(
                        pid,
                        error = %e,
                        "prototype rebuild failed (will retry on next cluster_all)"
                    );
                }
            }
        }

        // 2) 同步所有 person.face_count
        let persons = self.list_all()?;
        for p in persons {
            let _ = self.refresh_count(p.id);
        }

        info!(?summary, "cluster_all done");
        Ok(summary)
    }

    // =========================================================================
    // Phase 30 T11: Shadow Mode Integration
    // =========================================================================

    /// 运行 Shadow 评估（Phase 30 T11）
    ///
    /// Shadow 在 Legacy 决策之后运行，使用相同的 embedding 进行评估。
    /// Shadow 失败不会影响 Legacy 决策或扫描流程。
    ///
    /// 返回 ShadowExecutionResult，表示评估是否成功、失败或被跳过。
    fn evaluate_shadow(
        &self,
        face_id: i64,
        image_id: i64,
        embedding: &[f32],
        face_quality: FaceQuality,
        legacy_decision: &PersonAssignment,
        legacy_score: Option<f32>,
        legacy_margin: Option<f32>,
        candidate_count: usize,
    ) -> ShadowExecutionResult {
        // 检查是否启用 Shadow 模式
        if !self.execution_mode.is_shadow() {
            return ShadowExecutionResult::Skipped {
                reason: ShadowSkipReason::ExecutionModeDisabled,
            };
        }

        // 检查 IdentityPipeline 是否可用
        let pipeline = match self.identity_pipeline.as_ref() {
            Some(p) => p,
            None => {
                return ShadowExecutionResult::Skipped {
                    reason: ShadowSkipReason::PipelineUnavailable,
                };
            }
        };

        // 构建 IdentityQuery
        let query = IdentityQuery {
            face_id,
            image_id,
            face_embedding: embedding.to_vec(),
            face_quality,
            body_embedding: None, // Face-only for now
            body_quality: 0.0,
            body_available: false,
        };

        // 运行 Shadow 评估（best-effort，带 panic 隔离）
        let start = std::time::Instant::now();
        let eval_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| pipeline.evaluate(&query)));

        let elapsed = start.elapsed().as_millis() as u64;

        let result = match eval_result {
            Ok(eval_result) => eval_result,
            Err(_) => {
                warn!(face_id, "Shadow evaluation panicked");
                return ShadowExecutionResult::Failed { error: "panicked".to_string() };
            }
        };

        // 记录 Shadow 日志
        debug!(
            image_id,
            face_id,
            legacy = ?legacy_decision,
            shadow = ?result.shadow_decision,
            confidence = result.shadow_confidence,
            anti_chain = ?result.anti_chain.as_ref().map(|ac| ac.is_ambiguous),
            elapsed_ms = elapsed,
            "[IDENTITY_SHADOW]"
        );

        // 将 Shadow 结果写入数据库（best-effort，失败不影响 Legacy）
        self.write_shadow_record(
            face_id,
            image_id,
            legacy_decision,
            legacy_score,
            legacy_margin,
            candidate_count,
            &result,
            elapsed,
        );

        ShadowExecutionResult::Evaluated { decision: result }
    }

    /// 将 Shadow 记录写入数据库（best-effort）。
    ///
    /// 失败时仅记录日志，不影响扫描流程。
    fn write_shadow_record(
        &self,
        face_id: i64,
        image_id: i64,
        legacy_decision: &PersonAssignment,
        legacy_score: Option<f32>,
        legacy_margin: Option<f32>,
        candidate_count: usize,
        result: &crate::identity_evidence::IdentityDecisionResult,
        elapsed_ms: u64,
    ) {
        // 构建 legacy 相关字段
        let (legacy_person_id, legacy_decision_str) = match legacy_decision {
            PersonAssignment::Existing(pid) => (Some(*pid), "confirmed"),
            PersonAssignment::New { .. } => (None, "unknown"),
        };

        // 构建 shadow 相关字段
        let shadow_person_id = result.shadow_candidate.as_ref().map(|c| c.person_id);
        let shadow_decision_str = match result.shadow_decision {
            IdentityDecision::ConfidentMatch => "confirmed",
            IdentityDecision::WeakMatch => "probable",
            IdentityDecision::Ambiguous => "conflict",
            IdentityDecision::NewPerson => "unknown",
            IdentityDecision::StrongMatchWithCompetitor => "strong_match_with_competitor",
            IdentityDecision::Unreliable => "unreliable",
            IdentityDecision::SupportedMatch => "supported_match",
        };

        // 计算 disagreement_type
        let disagreement = ShadowDisagreementType::from_comparison(
            Self::parse_legacy_decision(legacy_decision_str),
            legacy_person_id,
            result.shadow_decision,
            shadow_person_id,
        );

        // anti_chain / pollution / quality_gate 状态
        let anti_chain_status = result
            .anti_chain
            .as_ref()
            .map(|ac| if ac.is_ambiguous { "blocked" } else { "passed" })
            .unwrap_or("N/A");

        let pollution_status = result
            .pollution_result
            .as_ref()
            .map(|pr| if pr.is_polluted { "blocked" } else { "passed" })
            .unwrap_or("N/A");

        let quality_gate_status = if result.quality_gate_passed { "passed" } else { "rejected" };

        // face/body score 和 margin (Shadow's evidence)
        let face_score = result.face_evidence.as_ref().map(|e| e.score);
        let face_margin = result.face_evidence.as_ref().and_then(|e| e.margin);
        let body_score = result.body_evidence.as_ref().map(|e| e.score);
        let body_margin = result.body_evidence.as_ref().and_then(|e| e.margin);

        // face_quality composite and components
        let face_quality = result.face_evidence.as_ref().map(|e| e.quality.composite_quality());
        let body_quality = result.body_evidence.as_ref().map(|e| e.quality);

        // Face quality components from FaceQuality
        let (face_size, yaw, pitch, roll, blur_score) = if let Some(ref fe) = result.face_evidence {
            (
                Some(fe.quality.face_size),
                Some(fe.quality.yaw),
                Some(fe.quality.pitch),
                Some(fe.quality.roll),
                Some(fe.quality.blur_score),
            )
        } else {
            (None, None, None, None, None)
        };

        // pipeline_version 字符串
        let pipeline_version = format!(
            "{}/{}/{}/{}",
            result.pipeline_version.face_detector,
            result.pipeline_version.face_embedding,
            result.pipeline_version.body_embedding,
            result.pipeline_version.fusion
        );

        let record = NewShadowRecord {
            image_id,
            face_id: Some(face_id),
            body_id: None,
            legacy_person_id,
            legacy_decision: legacy_decision_str.to_string(),
            shadow_person_id,
            shadow_decision: shadow_decision_str.to_string(),
            face_score,
            face_margin,
            body_score,
            body_margin,
            confidence: result.shadow_confidence,
            face_quality,
            body_quality,
            anti_chain_status: anti_chain_status.to_string(),
            pollution_status: pollution_status.to_string(),
            quality_gate_status: quality_gate_status.to_string(),
            disagreement_type: format!("{:?}", disagreement),
            pipeline_version,
            threshold_version: "v1".to_string(),
            model_version: result.pipeline_version.face_embedding.clone(),
            // Phase 31.5: Enhanced fields
            legacy_score,
            legacy_margin,
            candidate_count: Some(candidate_count as i32),
            shadow_face_score: face_score,
            shadow_face_margin: face_margin,
            face_size,
            yaw,
            pitch,
            roll,
            blur_score,
        };

        // 写入数据库（best-effort）
        if let Err(e) = self.db.transaction(|tx| tx.shadow_records().insert(&record)) {
            warn!(face_id, image_id, error = %e, "[IDENTITY_SHADOW] failed to write shadow record");
        }
    }

    /// 将 legacy 决策字符串解析为 IdentityDecision。
    fn parse_legacy_decision(s: &str) -> IdentityDecision {
        match s {
            "confirmed" => IdentityDecision::ConfidentMatch,
            "probable" => IdentityDecision::WeakMatch,
            "conflict" => IdentityDecision::Ambiguous,
            _ => IdentityDecision::NewPerson,
        }
    }
}

/// 聚类结果。
#[derive(Debug, Default, Clone, serde::Serialize)]
pub struct ClusterSummary {
    /// 已分配到已有 person 的 face 数
    pub assigned: usize,
    /// 创建新 person 数
    pub created: usize,
}

/// 简单 Clusterer adapter（trait 实现，方便上层用）。
pub struct PersonClusterer<'a> {
    pub service: &'a PersonService,
}

impl<'a> Clusterer for PersonClusterer<'a> {
    fn assign_or_create(&self, embedding: &Embedding) -> PersonAssignment {
        match self.service.assign_or_create(embedding.as_slice()) {
            Ok(a) => a,
            Err(_) => PersonAssignment::New { person_id: 0 },
        }
    }
}

/// cosine 相似度（输入应已归一化）。
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..n {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na.sqrt() * nb.sqrt()).max(1e-6);
    (dot / denom).clamp(0.0, 1.0)
}

/// 综合置信度 = similarity 主导 + quality + pose 修正（item 18）。
///
/// 权重：similarity 0.6 / quality 0.25 / pose 0.15 —— 与 `face_person_assignments.confidence`
/// 字段语义对齐（0..1）。quality/pose 缺失时取中性值 0.5（不拉低 confidence）。
pub fn assignment_confidence(similarity: f32, quality: Option<f32>, pose: Option<f32>) -> f32 {
    let q = quality.unwrap_or(0.5).clamp(0.0, 1.0);
    let p = pose.unwrap_or(0.5).clamp(0.0, 1.0);
    (0.6 * similarity + 0.25 * q + 0.15 * p).clamp(0.0, 1.0)
}
#[cfg(test)]
mod tests {
    use super::*;

    /// item 18:confidence = 0.6*sim + 0.25*quality + 0.15*pose,clamp 到 [0,1]。
    #[test]
    fn confidence_blends_sim_quality_pose() {
        // sim=1.0, q=1.0, p=1.0 → 1.0
        assert!((assignment_confidence(1.0, Some(1.0), Some(1.0)) - 1.0).abs() < 1e-6);
        // sim=0.5, q=1.0, p=1.0 → 0.6*0.5 + 0.25 + 0.15 = 0.7
        assert!((assignment_confidence(0.5, Some(1.0), Some(1.0)) - 0.7).abs() < 1e-6);
        // sim=0, q=0, p=0 → 0
        assert!((assignment_confidence(0.0, Some(0.0), Some(0.0)) - 0.0).abs() < 1e-6);
    }

    /// 缺失 quality/pose 时取中性 0.5,不拉低 confidence。
    #[test]
    fn confidence_missing_quality_pose_uses_neutral() {
        // sim=0.5, q=missing→0.5, p=missing→0.5 → 0.6*0.5 + 0.25*0.5 + 0.15*0.5 = 0.5
        assert!((assignment_confidence(0.5, None, None) - 0.5).abs() < 1e-6);
    }

    /// confidence clamp 到 [0,1](负/超界输入)。
    #[test]
    fn confidence_clamps_to_unit_range() {
        assert!((assignment_confidence(2.0, Some(2.0), Some(2.0)) - 1.0).abs() < 1e-6);
        assert!((assignment_confidence(-1.0, Some(-1.0), Some(-1.0)) - 0.0).abs() < 1e-6);
    }
}
