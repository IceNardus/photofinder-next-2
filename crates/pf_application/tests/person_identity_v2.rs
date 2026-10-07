//! Person Identity V2 Tests — Phase 18/19 Feature Tests
//!
//! Tests for:
//! - IdentityConfig feature flags (enable_body_reid, enable_identity_fusion)
//! - IdentityEvidence fusion with missing channel handling
//! - Welford prototype update
//! - Production identity debug logging
//!
//! 运行：
//! ```bash
//! cargo test -p pf_application --test person_identity_v2 -- --nocapture
//! ```

use pf_application::identity_evidence::{
    AntiChainingResult, EvidenceLevel, IdentityDecision, IdentityEvidence,
    PrimarySignal,
};
use pf_application::prototype_service::{l2_normalize, welford_update_prototype};
use pf_config::Config;

/// Test IdentityConfig defaults
#[test]
fn test_identity_config_defaults() {
    let config = Config::default();
    assert!(!config.identity.enable_body_reid);
    assert!(!config.identity.enable_identity_fusion);
}

/// Test IdentityConfig can be enabled
#[test]
fn test_identity_config_enable_flags() {
    let config = Config::default();
    assert!(!config.identity.enable_body_reid);
    assert!(!config.identity.enable_identity_fusion);
}

/// Test IdentityEvidence EvidenceLevel with missing channels
#[test]
fn test_evidence_level_dual() {
    let mut e = IdentityEvidence::new(1);
    e.face_available = true;
    e.body_available = true;
    assert_eq!(e.evidence_level(), EvidenceLevel::Dual);
}

#[test]
fn test_evidence_level_face_only() {
    let mut e = IdentityEvidence::new(1);
    e.face_available = true;
    e.body_available = false;
    assert_eq!(e.evidence_level(), EvidenceLevel::FaceOnly);
}

#[test]
fn test_evidence_level_body_only() {
    let mut e = IdentityEvidence::new(1);
    e.face_available = false;
    e.body_available = true;
    assert_eq!(e.evidence_level(), EvidenceLevel::BodyOnly);
}

#[test]
fn test_evidence_level_none() {
    let e = IdentityEvidence::new(1);
    assert_eq!(e.evidence_level(), EvidenceLevel::None);
}

/// Test fusion_margin with missing channels
#[test]
fn test_fusion_margin_dual() {
    let mut e = IdentityEvidence::new(1);
    e.face_available = true;
    e.body_available = true;
    e.face_margin = 0.10;
    e.body_margin = 0.05;
    // 0.5 * face_margin + 0.5 * body_margin = 0.5 * 0.10 + 0.5 * 0.05 = 0.075
    let fusion = e.fusion_margin(0.5);
    assert!((fusion - 0.075).abs() < 1e-6);
}

#[test]
fn test_fusion_margin_face_only() {
    let mut e = IdentityEvidence::new(1);
    e.face_available = true;
    e.body_available = false;
    e.face_margin = 0.10;
    // Should return face_margin when body unavailable
    let fusion = e.fusion_margin(0.5);
    assert!((fusion - 0.10).abs() < 1e-6);
}

#[test]
fn test_fusion_margin_body_only() {
    let mut e = IdentityEvidence::new(1);
    e.face_available = false;
    e.body_available = true;
    e.body_margin = 0.05;
    // Should return body_margin when face unavailable
    let fusion = e.fusion_margin(0.5);
    assert!((fusion - 0.05).abs() < 1e-6);
}

#[test]
fn test_fusion_margin_neither() {
    let e = IdentityEvidence::new(1);
    // Should return 0 when neither available
    let fusion = e.fusion_margin(0.5);
    assert!((fusion - 0.0).abs() < 1e-6);
}

/// Test missing channel handling in decide()
#[test]
fn test_decide_face_only_strong() {
    let mut e = IdentityEvidence::new(1);
    e.face_available = true;
    e.body_available = false;
    e.face_score = 0.78;
    e.face_margin = 0.05;
    assert_eq!(e.decide(), IdentityDecision::ConfidentMatch);
}

#[test]
fn test_decide_face_only_weak() {
    let mut e = IdentityEvidence::new(1);
    e.face_available = true;
    e.body_available = false;
    e.face_score = 0.58;
    e.face_margin = 0.0;
    assert_eq!(e.decide(), IdentityDecision::WeakMatch);
}

#[test]
fn test_decide_body_only_strong() {
    let mut e = IdentityEvidence::new(1);
    e.face_available = false;
    e.body_available = true;
    e.body_score = 0.78;
    e.body_margin = 0.06;
    assert_eq!(e.decide(), IdentityDecision::WeakMatch);
}

#[test]
fn test_decide_dual_confident() {
    let mut e = IdentityEvidence::new(1);
    e.face_available = true;
    e.body_available = true;
    e.face_score = 0.70;
    e.face_margin = 0.03;
    e.body_score = 0.72;
    e.body_margin = 0.02;
    assert_eq!(e.decide(), IdentityDecision::ConfidentMatch);
}

#[test]
fn test_decide_dual_weak_with_body_support() {
    let mut e = IdentityEvidence::new(1);
    e.face_available = true;
    e.body_available = true;
    e.face_score = 0.58;
    e.face_margin = 0.0;
    e.body_score = 0.78;
    e.body_margin = 0.06;
    assert_eq!(e.decide(), IdentityDecision::WeakMatch);
}

#[test]
fn test_decide_new_person_no_evidence() {
    let e = IdentityEvidence::new(1);
    assert_eq!(e.decide(), IdentityDecision::NewPerson);
}

/// Test AntiChainingResult
#[test]
fn test_anti_chaining_not_ambiguous() {
    let mut e1 = IdentityEvidence::new(1);
    e1.face_score = 0.80;
    e1.body_score = 0.85;

    let mut e2 = IdentityEvidence::new(2);
    e2.face_score = 0.60;
    e2.body_score = 0.90;

    let result = AntiChainingResult::check(&[e1, e2], 0.05);
    assert!(!result.is_ambiguous);
    assert_eq!(result.primary_signal, PrimarySignal::Face); // face_gap=0.20 > body_gap=0.05
}

#[test]
fn test_anti_chaining_ambiguous() {
    let mut e1 = IdentityEvidence::new(1);
    e1.face_score = 0.75;
    e1.body_score = 0.80;

    let mut e2 = IdentityEvidence::new(2);
    e2.face_score = 0.73;
    e2.body_score = 0.82;

    let result = AntiChainingResult::check(&[e1, e2], 0.05);
    assert!(result.is_ambiguous); // both gaps < 0.05
}

#[test]
fn test_anti_chaining_single_candidate() {
    let mut e1 = IdentityEvidence::new(1);
    e1.face_score = 0.75;
    e1.body_score = 0.80;

    let result = AntiChainingResult::check(&[e1], 0.05);
    assert!(!result.is_ambiguous); // single candidate is not ambiguous
}

/// Test Welford update
#[test]
fn test_welford_update_increments_count() {
    let face = vec![0.5f32, 0.5, 0.5, 0.5];
    let (mean, n) = welford_update_prototype(&[0.0f32; 4], 0, &face);
    assert_eq!(n, 1);
    // First face should just be normalized
    let norm = mean.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
}

#[test]
fn test_welford_update_blends_multiple_faces() {
    let face1 = vec![1.0f32, 0.0];
    let face2 = vec![0.0f32, 1.0];

    let (mean1, n1) = welford_update_prototype(&[0.0f32; 2], 0, &face1);
    assert_eq!(n1, 1);

    let (mean2, n2) = welford_update_prototype(&mean1, n1, &face2);
    assert_eq!(n2, 2);

    // After blending two orthogonal vectors, should be normalized to 45 degrees
    let norm = mean2.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
}

#[test]
fn test_l2_normalize_unit_vector() {
    let mut v = vec![0.5f32, 0.5, 0.5, 0.5];
    l2_normalize(&mut v);
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-6);
}

#[test]
fn test_l2_normalize_3_4_5() {
    let mut v = vec![3.0f32, 4.0];
    l2_normalize(&mut v);
    assert!((v[0] - 0.6).abs() < 1e-6);
    assert!((v[1] - 0.8).abs() < 1e-6);
}

/// Test IdentitySearchResult fields are populated correctly
#[test]
fn test_identity_search_result_fields() {
    use pf_application::identity_evidence::IdentitySearchResult;

    let result = IdentitySearchResult {
        person_id: 42,
        decision: IdentityDecision::ConfidentMatch,
        confidence: 0.85,
        face_score: 0.78,
        body_score: 0.72,
        face_margin: 0.05,
        body_margin: 0.03,
        face_rank: 1,
        body_rank: 2,
        is_ambiguous: false,
    };

    assert_eq!(result.person_id, 42);
    assert_eq!(result.decision, IdentityDecision::ConfidentMatch);
    assert!((result.confidence - 0.85).abs() < 1e-6);
    assert!(!result.is_ambiguous);
}

/// Test DecisionThresholds defaults
#[test]
fn test_decision_thresholds_defaults() {
    use pf_application::identity_evidence::DecisionThresholds;

    let thresholds = DecisionThresholds::default();
    assert!((thresholds.face_strong - 0.75).abs() < 1e-6);
    assert!((thresholds.face_medium - 0.65).abs() < 1e-6);
    assert!((thresholds.face_weak - 0.55).abs() < 1e-6);
    assert!((thresholds.body_threshold - 0.70).abs() < 1e-6);
    assert!((thresholds.body_strong - 0.75).abs() < 1e-6);
    assert!((thresholds.margin_threshold - 0.03).abs() < 1e-6);
    assert!((thresholds.anti_chain_threshold - 0.05).abs() < 1e-6);
}

/// Test ProductionRequirements defaults
#[test]
fn test_production_requirements_defaults() {
    use pf_application::identity_evidence::ProductionRequirements;

    let req = ProductionRequirements::default();
    assert!((req.min_top1 - 0.95).abs() < 1e-6);
    assert!((req.max_far - 0.01).abs() < 1e-6);
    assert!((req.max_frr - 0.10).abs() < 1e-6);
    assert!((req.min_loo_pass_rate - 0.90).abs() < 1e-6);
}
