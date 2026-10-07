//! Shadow Records Repository (Phase 30 T10)
//!
//! Repository for storing shadow pipeline evaluation results.
//! Shadow records track disagreements between Legacy and Shadow pipelines.

use rusqlite::params;

use crate::connection::Transaction;
use crate::error::DatabaseError;

/// Shadow record row (read from DB)
#[derive(Debug, Clone)]
pub struct ShadowRecordRow {
    pub id: i64,
    pub image_id: i64,
    pub face_id: Option<i64>,
    pub body_id: Option<i64>,
    pub legacy_person_id: Option<i64>,
    pub legacy_decision: String,
    pub shadow_person_id: Option<i64>,
    pub shadow_decision: String,
    pub face_score: Option<f32>,
    pub face_margin: Option<f32>,
    pub body_score: Option<f32>,
    pub body_margin: Option<f32>,
    pub confidence: f32,
    pub face_quality: Option<f32>,
    pub body_quality: Option<f32>,
    pub anti_chain_status: String,
    pub pollution_status: String,
    pub quality_gate_status: String,
    pub disagreement_type: String,
    pub pipeline_version: String,
    pub threshold_version: String,
    pub model_version: String,
    pub created_at: String,
    // Phase 31.5: Enhanced fields
    pub legacy_score: Option<f32>,
    pub legacy_margin: Option<f32>,
    pub candidate_count: Option<i32>,
    pub shadow_face_score: Option<f32>,
    pub shadow_face_margin: Option<f32>,
    pub face_size: Option<f32>,
    pub yaw: Option<f32>,
    pub pitch: Option<f32>,
    pub roll: Option<f32>,
    pub blur_score: Option<f32>,
}

/// New shadow record (for insert)
#[derive(Debug, Clone)]
pub struct NewShadowRecord {
    pub image_id: i64,
    pub face_id: Option<i64>,
    pub body_id: Option<i64>,
    pub legacy_person_id: Option<i64>,
    pub legacy_decision: String,
    pub shadow_person_id: Option<i64>,
    pub shadow_decision: String,
    pub face_score: Option<f32>,
    pub face_margin: Option<f32>,
    pub body_score: Option<f32>,
    pub body_margin: Option<f32>,
    pub confidence: f32,
    pub face_quality: Option<f32>,
    pub body_quality: Option<f32>,
    pub anti_chain_status: String,
    pub pollution_status: String,
    pub quality_gate_status: String,
    pub disagreement_type: String,
    pub pipeline_version: String,
    pub threshold_version: String,
    pub model_version: String,
    // Phase 31.5: Enhanced fields for disagreement analysis
    pub legacy_score: Option<f32>,
    pub legacy_margin: Option<f32>,
    pub candidate_count: Option<i32>,
    pub shadow_face_score: Option<f32>,
    pub shadow_face_margin: Option<f32>,
    pub face_size: Option<f32>,
    pub yaw: Option<f32>,
    pub pitch: Option<f32>,
    pub roll: Option<f32>,
    pub blur_score: Option<f32>,
}

/// Shadow Records Repository
pub struct ShadowRecordsRepository<'tx, 'db> {
    pub(crate) tx: &'tx mut Transaction<'db>,
}

impl<'tx, 'db> ShadowRecordsRepository<'tx, 'db> {
    /// Insert a new shadow record.
    /// Uses INSERT OR REPLACE for idempotency (unique constraint on image_id, face_id, pipeline_version, threshold_version).
    pub fn insert(&mut self, record: &NewShadowRecord) -> Result<i64, DatabaseError> {
        self.tx.execute(
            r#"INSERT OR REPLACE INTO identity_shadow_records (
                image_id, face_id, body_id,
                legacy_person_id, legacy_decision,
                shadow_person_id, shadow_decision,
                face_score, face_margin, body_score, body_margin,
                confidence, face_quality, body_quality,
                anti_chain_status, pollution_status, quality_gate_status,
                disagreement_type,
                pipeline_version, threshold_version, model_version,
                created_at,
                legacy_score, legacy_margin, candidate_count,
                shadow_face_score, shadow_face_margin,
                face_size, yaw, pitch, roll, blur_score
            ) VALUES (
                ?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17, ?18, ?19, ?20, ?21,
                strftime('%Y-%m-%dT%H:%M:%fZ','now'),
                ?22, ?23, ?24, ?25, ?26, ?27, ?28, ?29, ?30, ?31
            )"#,
            params![
                record.image_id,
                record.face_id,
                record.body_id,
                record.legacy_person_id,
                record.legacy_decision,
                record.shadow_person_id,
                record.shadow_decision,
                record.face_score,
                record.face_margin,
                record.body_score,
                record.body_margin,
                record.confidence,
                record.face_quality,
                record.body_quality,
                record.anti_chain_status,
                record.pollution_status,
                record.quality_gate_status,
                record.disagreement_type,
                record.pipeline_version,
                record.threshold_version,
                record.model_version,
                record.legacy_score,
                record.legacy_margin,
                record.candidate_count,
                record.shadow_face_score,
                record.shadow_face_margin,
                record.face_size,
                record.yaw,
                record.pitch,
                record.roll,
                record.blur_score,
            ],
        )?;
        Ok(self.tx.last_insert_rowid())
    }

    /// Get shadow record by face_id
    pub fn get_by_face_id(&mut self, face_id: i64) -> Result<Option<ShadowRecordRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            r#"SELECT id, image_id, face_id, body_id,
                      legacy_person_id, legacy_decision,
                      shadow_person_id, shadow_decision,
                      face_score, face_margin, body_score, body_margin,
                      confidence, face_quality, body_quality,
                      anti_chain_status, pollution_status, quality_gate_status,
                      disagreement_type,
                      pipeline_version, threshold_version, model_version,
                      created_at,
                      legacy_score, legacy_margin, candidate_count,
                      shadow_face_score, shadow_face_margin,
                      face_size, yaw, pitch, roll, blur_score
               FROM identity_shadow_records WHERE face_id = ?1"#,
        )?;
        let mut rows = stmt.query(params![face_id])?;
        if let Some(row) = rows.next()? {
            Ok(Some(row_to_shadow_record(row)?))
        } else {
            Ok(None)
        }
    }

    /// Count all shadow records
    pub fn count(&mut self) -> Result<i64, DatabaseError> {
        let count: i64 = self.tx.query_row(
            "SELECT COUNT(*) FROM identity_shadow_records",
            [],
            |r| r.get(0),
        )?;
        Ok(count)
    }

    /// Count disagreements (where legacy_decision != shadow_decision)
    pub fn count_disagreements(&mut self) -> Result<i64, DatabaseError> {
        let count: i64 = self.tx.query_row(
            "SELECT COUNT(*) FROM identity_shadow_records WHERE legacy_decision != shadow_decision",
            [],
            |r| r.get(0),
        )?;
        Ok(count)
    }

    /// List recent shadow records (ordered by created_at DESC)
    pub fn list_recent(&mut self, limit: i64) -> Result<Vec<ShadowRecordRow>, DatabaseError> {
        let mut stmt = self.tx.prepare(
            r#"SELECT id, image_id, face_id, body_id,
                      legacy_person_id, legacy_decision,
                      shadow_person_id, shadow_decision,
                      face_score, face_margin, body_score, body_margin,
                      confidence, face_quality, body_quality,
                      anti_chain_status, pollution_status, quality_gate_status,
                      disagreement_type,
                      pipeline_version, threshold_version, model_version,
                      created_at,
                      legacy_score, legacy_margin, candidate_count,
                      shadow_face_score, shadow_face_margin,
                      face_size, yaw, pitch, roll, blur_score
               FROM identity_shadow_records
               ORDER BY created_at DESC
               LIMIT ?1"#,
        )?;
        let mut rows = stmt.query(params![limit])?;
        let mut out = Vec::new();
        while let Some(row) = rows.next()? {
            out.push(row_to_shadow_record(row)?);
        }
        Ok(out)
    }
}

fn row_to_shadow_record(row: &rusqlite::Row<'_>) -> Result<ShadowRecordRow, DatabaseError> {
    Ok(ShadowRecordRow {
        id: row.get(0)?,
        image_id: row.get(1)?,
        face_id: row.get(2)?,
        body_id: row.get(3)?,
        legacy_person_id: row.get(4)?,
        legacy_decision: row.get(5)?,
        shadow_person_id: row.get(6)?,
        shadow_decision: row.get(7)?,
        face_score: row.get(8)?,
        face_margin: row.get(9)?,
        body_score: row.get(10)?,
        body_margin: row.get(11)?,
        confidence: row.get(12)?,
        face_quality: row.get(13)?,
        body_quality: row.get(14)?,
        anti_chain_status: row.get(15)?,
        pollution_status: row.get(16)?,
        quality_gate_status: row.get(17)?,
        disagreement_type: row.get(18)?,
        pipeline_version: row.get(19)?,
        threshold_version: row.get(20)?,
        model_version: row.get(21)?,
        created_at: row.get(22)?,
        // Phase 31.5: Enhanced fields
        legacy_score: row.get(23)?,
        legacy_margin: row.get(24)?,
        candidate_count: row.get(25)?,
        shadow_face_score: row.get(26)?,
        shadow_face_margin: row.get(27)?,
        face_size: row.get(28)?,
        yaw: row.get(29)?,
        pitch: row.get(30)?,
        roll: row.get(31)?,
        blur_score: row.get(32)?,
    })
}
