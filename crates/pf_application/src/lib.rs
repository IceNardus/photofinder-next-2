//! # pf_application — 业务编排层
//!
//! 唯一允许同时调用多个下层 crate 的层。
//! 暴露 `async fn` 给前端（通过 Tauri command / JNI / FFI）。

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod bootstrap;
pub mod body_index;
pub mod body_prototype_selector;
pub mod body_prototype_service;
pub mod cluster_v2;
pub mod context;
pub mod error;
pub mod executors;
pub mod identity_evidence;
pub mod index;
pub mod index_generation;
pub mod person;
pub mod prototype_selector;
pub mod prototype_service;
pub mod scan;
pub mod search;
pub mod search_v2;

pub use bootstrap::{assemble, default_scheduler_config, start_workers, Bootstrapped};
pub use cluster_v2::{best_prototype_match, cosine_similarity, l2_normalize};
pub use context::AppContext;
pub use error::ApplicationError;
pub use executors::{
    register_executors, ClusterFacesExecutor, IndexFaceExecutor, IndexObjectExecutor,
    ScanExecutor,
};
pub use identity_evidence::{
    AntiChainingResult, BenchmarkMetrics, CandidateSet, DecisionThresholds,
    IdentityDecision, IdentityEvidence, IdentitySearchResult, PersonCandidate,
    PrimarySignal, ProductionCheckResult, ProductionRequirements, SearchHitWithScore,
};
pub use index::{quality_bridge, IndexService, IndexSummary};
pub use person::{ClusterPolicy, ClusterSummary, PersonService};
pub use prototype_selector::{classify_face_to_prototype_type, select_prototypes_for_person};
pub use prototype_service::PrototypeService;
pub use body_prototype_service::BodyPrototypeService;
pub use scan::{ScanService, ScanSummary};
pub use search::{
    clamp_roi_to_image, coarse_fetch_k, expand_roi_for_matching, select_fine_candidates,
    SearchService,
};
pub use search_v2::search_by_person_impl;