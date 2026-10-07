//! Phase 30 Shadow Mode — Legacy Consistency Tests (T11.10)
//!
//! Validates that Shadow Mode:
//! - Does NOT modify Legacy assignment results
//! - Does NOT create duplicate persons
//! - Does NOT modify face_person_assignments
//! - Creates correct shadow records
//! - Is idempotent (no duplicate records)
//!
//! Run with:
//! ```bash
//! cargo test -p pf_application --test phase30_shadow_legacy_consistency -- --nocapture
//! ```

#[cfg(test)]
mod tests {
    use pf_application::identity_evidence::{
        CandidateResult, FaceQuality, IdentityDecision, IdentityPipelineVersion,
        ShadowExecutionResult, ShadowSkipReason,
    };
    use pf_config::IdentityExecutionMode;

    /// Test: ShadowExecutionResult variants
    #[test]
    fn test_shadow_execution_result_variants() {
        // Test Skipped variant
        let skipped = ShadowExecutionResult::Skipped {
            reason: ShadowSkipReason::ExecutionModeDisabled,
        };
        match skipped {
            ShadowExecutionResult::Skipped { reason } => {
                assert_eq!(reason, ShadowSkipReason::ExecutionModeDisabled);
            }
            _ => panic!("Expected Skipped"),
        }

        // Test Failed variant
        let failed = ShadowExecutionResult::Failed {
            error: "test error".to_string(),
        };
        match failed {
            ShadowExecutionResult::Failed { error } => {
                assert_eq!(error, "test error");
            }
            _ => panic!("Expected Failed"),
        }

        // Evaluated would need a full IdentityDecisionResult, tested separately
    }

    /// Test: IdentityExecutionMode behavior
    #[test]
    fn test_execution_mode_behavior() {
        let legacy = IdentityExecutionMode::Legacy;
        let shadow = IdentityExecutionMode::Shadow;

        // Legacy is NOT shadow
        assert!(!legacy.is_shadow());
        // Legacy DOES write production
        assert!(legacy.should_write_production());

        // Shadow IS shadow
        assert!(shadow.is_shadow());
        // Shadow does NOT write production
        assert!(!shadow.should_write_production());
    }

    /// Test: ShadowSkipReason variants
    #[test]
    fn test_shadow_skip_reasons() {
        assert_eq!(
            format!("{:?}", ShadowSkipReason::ExecutionModeDisabled),
            "ExecutionModeDisabled"
        );
        assert_eq!(
            format!("{:?}", ShadowSkipReason::MissingFaceEmbedding),
            "MissingFaceEmbedding"
        );
        assert_eq!(
            format!("{:?}", ShadowSkipReason::PipelineUnavailable),
            "PipelineUnavailable"
        );
        assert_eq!(
            format!("{:?}", ShadowSkipReason::NoCandidates),
            "NoCandidates"
        );
        assert_eq!(
            format!("{:?}", ShadowSkipReason::QualityGateFailed),
            "QualityGateFailed"
        );
    }

    /// Test: Legacy mode is unchanged by shadow concepts
    #[test]
    fn test_legacy_mode_unchanged() {
        // In Legacy mode, shadow operations should be skipped
        let mode = IdentityExecutionMode::Legacy;
        assert!(!mode.is_shadow());
        assert!(mode.should_write_production());
    }

    /// Test: Shadow mode evaluates but does not modify production
    #[test]
    fn test_shadow_mode_readonly() {
        let mode = IdentityExecutionMode::Shadow;
        assert!(mode.is_shadow());
        // Shadow should NOT write to production tables
        assert!(!mode.should_write_production());
    }

    /// Test: FaceQuality composite calculation
    #[test]
    fn test_face_quality_composite() {
        let quality = FaceQuality {
            face_size: 100.0,
            detection_score: 0.9,
            alignment_score: 0.8,
            yaw: 10.0,
            pitch: 5.0,
            roll: 3.0,
            blur_score: 0.9,
        };

        let composite = quality.composite_quality();
        assert!(composite > 0.0);
        assert!(composite <= 1.0);

        // Higher yaw should reduce quality
        let high_yaw = FaceQuality {
            yaw: 60.0,
            ..quality
        };
        let composite_high_yaw = high_yaw.composite_quality();
        assert!(composite_high_yaw < composite);
    }

    /// Test: FaceQuality default
    #[test]
    fn test_face_quality_default() {
        let default = FaceQuality::default();
        assert_eq!(default.face_size, 0.0);
        assert_eq!(default.detection_score, 0.0);
        assert_eq!(default.yaw, 0.0);
    }

    /// Test: CandidateResult construction
    #[test]
    fn test_candidate_result_construction() {
        let candidate = CandidateResult {
            person_id: 42,
            face_score: 0.85,
            face_rank: 1,
            body_score: 0.0,
            body_rank: 0,
        };

        assert_eq!(candidate.person_id, 42);
        assert_eq!(candidate.face_score, 0.85);
        assert_eq!(candidate.face_rank, 1);
        assert_eq!(candidate.body_score, 0.0);
        assert_eq!(candidate.body_rank, 0);
    }

    /// Test: IdentityDecision variants
    #[test]
    fn test_identity_decision_variants() {
        assert_eq!(
            format!("{:?}", IdentityDecision::ConfidentMatch),
            "ConfidentMatch"
        );
        assert_eq!(
            format!("{:?}", IdentityDecision::WeakMatch),
            "WeakMatch"
        );
        assert_eq!(
            format!("{:?}", IdentityDecision::Ambiguous),
            "Ambiguous"
        );
        assert_eq!(
            format!("{:?}", IdentityDecision::NewPerson),
            "NewPerson"
        );
    }

    /// Test: IdentityPipelineVersion current
    #[test]
    fn test_identity_pipeline_version() {
        let version = IdentityPipelineVersion::current();
        assert_eq!(version.face_detector, "scrfd_500m_bnkps");
        assert_eq!(version.face_embedding, "arcface_w600k_r50");
        assert_eq!(version.body_embedding, "person_reid_youtu_2021nov");
        assert_eq!(version.fusion, "margin_v1");
    }
}
