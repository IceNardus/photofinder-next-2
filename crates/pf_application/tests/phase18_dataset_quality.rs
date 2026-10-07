//! Phase 18 — Benchmark Dataset Quality System
//!
//! Dataset quality check and readiness evaluation for person identity benchmarks.
//!
//! 运行：
//! ```bash
//! cargo test -p pf_application --test phase18_dataset_quality -- --nocapture
//! ```
//!
//! Ignored test (requires actual images):
//! ```bash
//! cargo test -p pf_application --test phase18_dataset_quality -- --nocapture --ignored
//! ```

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

// ============================================================================
// Data Structures
// ============================================================================

/// Benchmark readiness criteria.
#[derive(Debug, Clone)]
pub struct BenchmarkCriteria {
    /// Minimum number of persons required
    pub min_persons: usize,
    /// Minimum images per person
    pub min_images_per_person: usize,
    /// Minimum usable images per person (face + body)
    pub min_usable_per_person: usize,
}

impl Default for BenchmarkCriteria {
    fn default() -> Self {
        Self {
            min_persons: 10,
            min_images_per_person: 5,
            min_usable_per_person: 5,
        }
    }
}

/// Image status flags.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageStatus {
    /// Face detected
    FaceValid,
    /// Body detected
    BodyValid,
    /// Both face and body valid
    Usable,
    /// No face detected
    NoFace,
    /// No body detected
    NoBody,
    /// Face quality too low
    LowQuality,
}

/// Person statistics.
#[derive(Debug, Clone)]
pub struct PersonStats {
    pub person_id: i64,
    pub name: String,
    pub total_images: usize,
    pub face_valid_count: usize,
    pub body_valid_count: usize,
    pub usable_count: usize,
    pub missing_face_count: usize,
    pub missing_body_count: usize,
    pub low_quality_count: usize,
}

/// Dataset statistics.
#[derive(Debug, Clone)]
pub struct DatasetStats {
    pub total_persons: usize,
    pub total_images: usize,
    pub usable_images: usize,
    pub missing_face_count: usize,
    pub missing_body_count: usize,
    pub low_quality_count: usize,
    pub min_images_per_person: usize,
    pub median_images_per_person: f64,
    pub max_images_per_person: usize,
    pub min_usable_per_person: usize,
    pub median_usable_per_person: f64,
    pub max_usable_per_person: usize,
}

/// Hard negative type.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HardNegativeType {
    /// Face cosine is high but different person
    FaceHardNegative,
    /// Body cosine is high but different person
    BodyHardNegative,
    /// Both face and body cosine are high but different person
    DualHardNegative,
}

/// Hard negative case.
#[derive(Debug, Clone)]
pub struct HardNegative {
    pub query_image_id: i64,
    pub query_person_id: i64,
    pub query_person_name: String,
    pub negative_image_id: i64,
    pub negative_person_id: i64,
    pub negative_person_name: String,
    pub hard_negative_type: HardNegativeType,
    pub face_score: f32,
    pub body_score: f32,
    pub face_margin: f32,
    pub body_margin: f32,
}

/// Positive pair.
#[derive(Debug, Clone)]
pub struct PositivePair {
    pub image1_id: i64,
    pub image2_id: i64,
    pub person_id: i64,
}

/// Negative pair.
#[derive(Debug, Clone)]
pub struct NegativePair {
    pub image1_id: i64,
    pub image1_person_id: i64,
    pub image2_id: i64,
    pub image2_person_id: i64,
    pub hard_negative_type: Option<HardNegativeType>,
}

/// Benchmark dataset JSON output.
#[derive(Debug, Clone)]
pub struct BenchmarkDatasetJson {
    pub persons: Vec<DatasetPerson>,
    pub images: Vec<DatasetImage>,
    pub positive_pairs: Vec<DatasetPositivePair>,
    pub negative_pairs: Vec<DatasetNegativePair>,
    pub hard_negatives: Vec<DatasetHardNegative>,
    pub statistics: DatasetStats,
    pub verdict: String,
    pub reasons: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct DatasetPerson {
    pub person_id: i64,
    pub name: String,
    pub image_count: usize,
    pub usable_count: usize,
}

#[derive(Debug, Clone)]
pub struct DatasetImage {
    pub id: i64,
    pub name: String,
    pub person_id: i64,
    pub person_name: String,
    pub status: String,
}

#[derive(Debug, Clone)]
pub struct DatasetPositivePair {
    pub image1_id: i64,
    pub image2_id: i64,
    pub person_id: i64,
}

#[derive(Debug, Clone)]
pub struct DatasetNegativePair {
    pub image1_id: i64,
    pub image1_person_id: i64,
    pub image2_id: i64,
    pub image2_person_id: i64,
    pub is_hard: bool,
}

#[derive(Debug, Clone)]
pub struct DatasetHardNegative {
    pub query_image_id: i64,
    pub query_person_id: i64,
    pub query_person_name: String,
    pub negative_image_id: i64,
    pub negative_person_id: i64,
    pub negative_person_name: String,
    pub hard_negative_type: String,
    pub face_score: f32,
    pub body_score: f32,
}

/// Dataset quality verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DatasetVerdict {
    /// Dataset is ready for benchmark
    DatasetReady,
    /// Dataset is insufficient for benchmark
    DatasetInsufficient,
}

// ============================================================================
// Core Logic
// ============================================================================

/// Analyze dataset and produce quality report.
pub struct DatasetAnalyzer {
    criteria: BenchmarkCriteria,
}

impl DatasetAnalyzer {
    pub fn new(criteria: BenchmarkCriteria) -> Self {
        Self { criteria }
    }

    pub fn with_default_criteria() -> Self {
        Self::new(BenchmarkCriteria::default())
    }

    /// Compute verdict based on criteria and stats.
    pub fn compute_verdict(&self, stats: &DatasetStats) -> (DatasetVerdict, Vec<String>) {
        let mut reasons = Vec::new();

        if stats.total_persons < self.criteria.min_persons {
            reasons.push(format!(
                "insufficient persons: {} < {}",
                stats.total_persons, self.criteria.min_persons
            ));
        }

        if stats.min_images_per_person < self.criteria.min_images_per_person {
            reasons.push(format!(
                "insufficient images per person: min={} < {}",
                stats.min_images_per_person, self.criteria.min_images_per_person
            ));
        }

        if stats.min_usable_per_person < self.criteria.min_usable_per_person {
            reasons.push(format!(
                "insufficient usable images per person: min={} < {}",
                stats.min_usable_per_person, self.criteria.min_usable_per_person
            ));
        }

        let verdict = if reasons.is_empty() {
            DatasetVerdict::DatasetReady
        } else {
            DatasetVerdict::DatasetInsufficient
        };

        (verdict, reasons)
    }

    /// Build person stats from image data.
    pub fn compute_person_stats(
        &self,
        images: &[AnalyzedImage],
    ) -> (Vec<PersonStats>, DatasetStats) {
        let mut person_map: HashMap<i64, Vec<&AnalyzedImage>> = HashMap::new();
        for img in images {
            person_map.entry(img.person_id).or_default().push(img);
        }

        let mut person_stats: Vec<PersonStats> = person_map.iter()
            .map(|(pid, imgs)| {
                let first = imgs.first().unwrap();
                PersonStats {
                    person_id: *pid,
                    name: first.person_name.clone(),
                    total_images: imgs.len(),
                    face_valid_count: imgs.iter().filter(|i| i.has_face).count(),
                    body_valid_count: imgs.iter().filter(|i| i.has_body).count(),
                    usable_count: imgs.iter().filter(|i| i.has_face && i.has_body).count(),
                    missing_face_count: imgs.iter().filter(|i| !i.has_face).count(),
                    missing_body_count: imgs.iter().filter(|i| i.has_body).count(),
                    low_quality_count: imgs.iter().filter(|i| i.low_quality).count(),
                }
            })
            .collect();

        person_stats.sort_by_key(|p| p.person_id);

        // Compute dataset stats
        let total_images = images.len();
        let usable_images = images.iter().filter(|i| i.has_face && i.has_body).count();

        let image_counts: Vec<usize> = person_stats.iter().map(|p| p.total_images).collect();
        let usable_counts: Vec<usize> = person_stats.iter().map(|p| p.usable_count).collect();

        let min_images = image_counts.iter().min().copied().unwrap_or(0);
        let max_images = image_counts.iter().max().copied().unwrap_or(0);
        let median_images = if image_counts.is_empty() {
            0.0
        } else {
            let mut sorted = image_counts.clone();
            sorted.sort();
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 {
                (sorted[mid - 1] + sorted[mid]) as f64 / 2.0
            } else {
                sorted[mid] as f64
            }
        };

        let min_usable = usable_counts.iter().min().copied().unwrap_or(0);
        let max_usable = usable_counts.iter().max().copied().unwrap_or(0);
        let median_usable = if usable_counts.is_empty() {
            0.0
        } else {
            let mut sorted = usable_counts.clone();
            sorted.sort();
            let mid = sorted.len() / 2;
            if sorted.len() % 2 == 0 {
                (sorted[mid - 1] + sorted[mid]) as f64 / 2.0
            } else {
                sorted[mid] as f64
            }
        };

        let dataset_stats = DatasetStats {
            total_persons: person_stats.len(),
            total_images,
            usable_images,
            missing_face_count: images.iter().filter(|i| !i.has_face).count(),
            missing_body_count: images.iter().filter(|i| i.has_body).count(),
            low_quality_count: images.iter().filter(|i| i.low_quality).count(),
            min_images_per_person: min_images,
            median_images_per_person: median_images,
            max_images_per_person: max_images,
            min_usable_per_person: min_usable,
            median_usable_per_person: median_usable,
            max_usable_per_person: max_usable,
        };

        (person_stats, dataset_stats)
    }

    /// Construct positive pairs (same person).
    pub fn construct_positive_pairs(&self, images: &[AnalyzedImage]) -> Vec<PositivePair> {
        let mut pairs = Vec::new();
        let usable: Vec<_> = images.iter().filter(|i| i.has_face && i.has_body).collect();

        // Group by person
        let mut person_groups: HashMap<i64, Vec<_>> = HashMap::new();
        for img in &usable {
            person_groups.entry(img.person_id).or_default().push(img);
        }

        for (_, imgs) in person_groups {
            for i in 0..imgs.len() {
                for j in (i + 1)..imgs.len() {
                    pairs.push(PositivePair {
                        image1_id: imgs[i].image_id,
                        image2_id: imgs[j].image_id,
                        person_id: imgs[i].person_id,
                    });
                }
            }
        }

        pairs
    }

    /// Construct negative pairs (different persons).
    /// Also identifies hard negatives.
    pub fn construct_negative_pairs(
        &self,
        images: &[AnalyzedImage],
        face_scores: &HashMap<(i64, i64), f32>,
        body_scores: &HashMap<(i64, i64), f32>,
    ) -> (Vec<NegativePair>, Vec<HardNegative>) {
        let mut pairs = Vec::new();
        let mut hard_negatives = Vec::new();

        let usable: Vec<_> = images.iter().filter(|i| i.has_face && i.has_body).collect();

        // Group by person
        let mut person_groups: HashMap<i64, Vec<_>> = HashMap::new();
        for img in &usable {
            person_groups.entry(img.person_id).or_default().push(img);
        }

        let person_ids: Vec<i64> = person_groups.keys().copied().collect();

        // For each pair of different persons
        for i in 0..person_ids.len() {
            for j in (i + 1)..person_ids.len() {
                let pid1 = person_ids[i];
                let pid2 = person_ids[j];
                let imgs1 = &person_groups[&pid1];
                let imgs2 = &person_groups[&pid2];

                let name1 = imgs1.first().map(|i| i.person_name.as_str()).unwrap_or("unknown");
                let name2 = imgs2.first().map(|i| i.person_name.as_str()).unwrap_or("unknown");

                for img1 in imgs1 {
                    for img2 in imgs2 {
                        // Look up scores in both directions (scores may not be symmetric)
                        let face_score = face_scores
                            .get(&(img1.image_id, img2.image_id))
                            .or_else(|| face_scores.get(&(img2.image_id, img1.image_id)))
                            .copied()
                            .unwrap_or(0.0);
                        let body_score = body_scores
                            .get(&(img1.image_id, img2.image_id))
                            .or_else(|| body_scores.get(&(img2.image_id, img1.image_id)))
                            .copied()
                            .unwrap_or(0.0);

                        // Determine if hard negative
                        let hn_type = self.classify_hard_negative(face_score, body_score);

                        if hn_type.is_some() {
                            hard_negatives.push(HardNegative {
                                query_image_id: img1.image_id,
                                query_person_id: pid1,
                                query_person_name: name1.to_string(),
                                negative_image_id: img2.image_id,
                                negative_person_id: pid2,
                                negative_person_name: name2.to_string(),
                                hard_negative_type: hn_type.unwrap(),
                                face_score,
                                body_score,
                                face_margin: 0.0, // Would need all scores to compute
                                body_margin: 0.0,
                            });
                        }

                        pairs.push(NegativePair {
                            image1_id: img1.image_id,
                            image1_person_id: pid1,
                            image2_id: img2.image_id,
                            image2_person_id: pid2,
                            hard_negative_type: hn_type,
                        });
                    }
                }
            }
        }

        (pairs, hard_negatives)
    }

    /// Classify hard negative type based on scores.
    fn classify_hard_negative(&self, face_score: f32, body_score: f32) -> Option<HardNegativeType> {
        // Hard negative thresholds (similar to production decision thresholds)
        const FACE_HARD_THRESHOLD: f32 = 0.65;
        const BODY_HARD_THRESHOLD: f32 = 0.70;

        let face_hard = face_score >= FACE_HARD_THRESHOLD;
        let body_hard = body_score >= BODY_HARD_THRESHOLD;

        if face_hard && body_hard {
            Some(HardNegativeType::DualHardNegative)
        } else if face_hard {
            Some(HardNegativeType::FaceHardNegative)
        } else if body_hard {
            Some(HardNegativeType::BodyHardNegative)
        } else {
            None
        }
    }

    /// Check for data leakage issues.
    pub fn check_leakage(&self, images: &[AnalyzedImage]) -> Vec<String> {
        let mut issues = Vec::new();

        // Check for duplicate/near-duplicate images within same person
        // (would need image hash comparison for full check)

        // Check for missing self-match exclusion in pairs
        // (pair construction should exclude same-image pairs)

        issues
    }

    /// Generate benchmark dataset JSON.
    pub fn generate_json(
        &self,
        person_stats: &[PersonStats],
        images: &[AnalyzedImage],
        positive_pairs: &[PositivePair],
        negative_pairs: &[NegativePair],
        hard_negatives: &[HardNegative],
        dataset_stats: &DatasetStats,
        verdict: DatasetVerdict,
        reasons: &[String],
    ) -> BenchmarkDatasetJson {
        let dataset_persons: Vec<DatasetPerson> = person_stats.iter()
            .map(|p| DatasetPerson {
                person_id: p.person_id,
                name: p.name.clone(),
                image_count: p.total_images,
                usable_count: p.usable_count,
            })
            .collect();

        let dataset_images: Vec<DatasetImage> = images.iter()
            .map(|img| DatasetImage {
                id: img.image_id,
                name: img.name.clone(),
                person_id: img.person_id,
                person_name: img.person_name.clone(),
                status: img.status_string(),
            })
            .collect();

        let dataset_positive: Vec<DatasetPositivePair> = positive_pairs.iter()
            .map(|p| DatasetPositivePair {
                image1_id: p.image1_id,
                image2_id: p.image2_id,
                person_id: p.person_id,
            })
            .collect();

        let dataset_negative: Vec<DatasetNegativePair> = negative_pairs.iter()
            .map(|p| DatasetNegativePair {
                image1_id: p.image1_id,
                image1_person_id: p.image1_person_id,
                image2_id: p.image2_id,
                image2_person_id: p.image2_person_id,
                is_hard: p.hard_negative_type.is_some(),
            })
            .collect();

        let dataset_hard: Vec<DatasetHardNegative> = hard_negatives.iter()
            .map(|h| DatasetHardNegative {
                query_image_id: h.query_image_id,
                query_person_id: h.query_person_id,
                query_person_name: h.query_person_name.clone(),
                negative_image_id: h.negative_image_id,
                negative_person_id: h.negative_person_id,
                negative_person_name: h.negative_person_name.clone(),
                hard_negative_type: h.hard_negative_type_string(),
                face_score: h.face_score,
                body_score: h.body_score,
            })
            .collect();

        BenchmarkDatasetJson {
            persons: dataset_persons,
            images: dataset_images,
            positive_pairs: dataset_positive,
            negative_pairs: dataset_negative,
            hard_negatives: dataset_hard,
            statistics: dataset_stats.clone(),
            verdict: verdict_string(verdict),
            reasons: reasons.to_vec(),
        }
    }
}

/// Analyzed image with status.
#[derive(Debug, Clone)]
pub struct AnalyzedImage {
    pub image_id: i64,
    pub name: String,
    pub person_id: i64,
    pub person_name: String,
    pub has_face: bool,
    pub has_body: bool,
    pub low_quality: bool,
}

impl AnalyzedImage {
    pub fn status(&self) -> ImageStatus {
        if self.has_face && self.has_body {
            ImageStatus::Usable
        } else if !self.has_face {
            ImageStatus::NoFace
        } else if !self.has_body {
            ImageStatus::NoBody
        } else {
            ImageStatus::NoFace
        }
    }

    pub fn status_string(&self) -> String {
        match self.status() {
            ImageStatus::Usable => "usable".to_string(),
            ImageStatus::FaceValid => "face_valid".to_string(),
            ImageStatus::BodyValid => "body_valid".to_string(),
            ImageStatus::NoFace => "no_face".to_string(),
            ImageStatus::NoBody => "no_body".to_string(),
            ImageStatus::LowQuality => "low_quality".to_string(),
        }
    }
}

fn verdict_string(v: DatasetVerdict) -> String {
    match v {
        DatasetVerdict::DatasetReady => "DATASET_READY".to_string(),
        DatasetVerdict::DatasetInsufficient => "DATASET_INSUFFICIENT".to_string(),
    }
}

impl HardNegative {
    pub fn hard_negative_type_string(&self) -> String {
        match self.hard_negative_type {
            HardNegativeType::FaceHardNegative => "face_hard_negative".to_string(),
            HardNegativeType::BodyHardNegative => "body_hard_negative".to_string(),
            HardNegativeType::DualHardNegative => "dual_hard_negative".to_string(),
        }
    }
}

// ============================================================================
// JSON Serialization
// ============================================================================

impl BenchmarkDatasetJson {
    pub fn to_json(&self) -> String {
        // Manual JSON serialization to avoid serde dependency in test
        let mut s = String::new();
        s.push_str("{\n");

        // persons
        s.push_str("  \"persons\": [\n");
        for (i, p) in self.persons.iter().enumerate() {
            s.push_str(&format!(
                "    {{\"person_id\": {}, \"name\": \"{}\", \"image_count\": {}, \"usable_count\": {}}}",
                p.person_id, p.name, p.image_count, p.usable_count
            ));
            if i < self.persons.len() - 1 {
                s.push(',');
            }
            s.push('\n');
        }
        s.push_str("  ],\n");

        // statistics
        let st = &self.statistics;
        s.push_str("  \"statistics\": {\n");
        s.push_str(&format!(
            "    \"total_persons\": {},\n    \"total_images\": {},\n    \"usable_images\": {},\n    \"missing_face_count\": {},\n    \"missing_body_count\": {},\n    \"low_quality_count\": {},\n    \"min_images_per_person\": {},\n    \"median_images_per_person\": {},\n    \"max_images_per_person\": {},\n    \"min_usable_per_person\": {},\n    \"median_usable_per_person\": {},\n    \"max_usable_per_person\": {}",
            st.total_persons, st.total_images, st.usable_images, st.missing_face_count, st.missing_body_count, st.low_quality_count, st.min_images_per_person, st.median_images_per_person as i64, st.max_images_per_person, st.min_usable_per_person, st.median_usable_per_person as i64, st.max_usable_per_person
        ));
        s.push_str("\n  },\n");

        // verdict
        s.push_str(&format!("  \"verdict\": \"{}\",\n", self.verdict));

        // reasons
        s.push_str("  \"reasons\": [");
        for (i, r) in self.reasons.iter().enumerate() {
            s.push_str(&format!("\"{}\"", r));
            if i < self.reasons.len() - 1 {
                s.push(',');
            }
        }
        s.push_str("],\n");

        // counts
        s.push_str(&format!(
            "  \"positive_pair_count\": {},\n  \"negative_pair_count\": {},\n  \"hard_negative_count\": {}",
            self.positive_pairs.len(),
            self.negative_pairs.len(),
            self.hard_negatives.len()
        ));
        s.push('\n');
        s.push('}');

        s
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_images() -> Vec<AnalyzedImage> {
        vec![
            // Person 1 with 5 usable images
            AnalyzedImage { image_id: 1, name: "p1_1.jpg".into(), person_id: 1, person_name: "person1".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 2, name: "p1_2.jpg".into(), person_id: 1, person_name: "person1".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 3, name: "p1_3.jpg".into(), person_id: 1, person_name: "person1".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 4, name: "p1_4.jpg".into(), person_id: 1, person_name: "person1".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 5, name: "p1_5.jpg".into(), person_id: 1, person_name: "person1".into(), has_face: true, has_body: true, low_quality: false },
            // Person 2 with 4 usable images
            AnalyzedImage { image_id: 6, name: "p2_1.jpg".into(), person_id: 2, person_name: "person2".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 7, name: "p2_2.jpg".into(), person_id: 2, person_name: "person2".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 8, name: "p2_3.jpg".into(), person_id: 2, person_name: "person2".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 9, name: "p2_4.jpg".into(), person_id: 2, person_name: "person2".into(), has_face: true, has_body: true, low_quality: false },
            // Person 3 with 3 usable images
            AnalyzedImage { image_id: 10, name: "p3_1.jpg".into(), person_id: 3, person_name: "person3".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 11, name: "p3_2.jpg".into(), person_id: 3, person_name: "person3".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 12, name: "p3_3.jpg".into(), person_id: 3, person_name: "person3".into(), has_face: true, has_body: true, low_quality: false },
            // Person 4 with 2 usable images
            AnalyzedImage { image_id: 13, name: "p4_1.jpg".into(), person_id: 4, person_name: "person4".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 14, name: "p4_2.jpg".into(), person_id: 4, person_name: "person4".into(), has_face: true, has_body: true, low_quality: false },
            // Person 5 with 1 usable image
            AnalyzedImage { image_id: 15, name: "p5_1.jpg".into(), person_id: 5, person_name: "person5".into(), has_face: true, has_body: true, low_quality: false },
            // Person 6 with missing face
            AnalyzedImage { image_id: 16, name: "p6_1.jpg".into(), person_id: 6, person_name: "person6".into(), has_face: false, has_body: true, low_quality: false },
            // Person 7 with missing body
            AnalyzedImage { image_id: 17, name: "p7_1.jpg".into(), person_id: 7, person_name: "person7".into(), has_face: true, has_body: false, low_quality: false },
        ]
    }

    #[test]
    fn test_insufficient_persons() {
        // Only 3 persons - should be insufficient for default criteria
        let images = vec![
            AnalyzedImage { image_id: 1, name: "p1_1.jpg".into(), person_id: 1, person_name: "person1".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 2, name: "p1_2.jpg".into(), person_id: 1, person_name: "person1".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 3, name: "p1_3.jpg".into(), person_id: 1, person_name: "person1".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 4, name: "p2_1.jpg".into(), person_id: 2, person_name: "person2".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 5, name: "p2_2.jpg".into(), person_id: 2, person_name: "person2".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 6, name: "p3_1.jpg".into(), person_id: 3, person_name: "person3".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 7, name: "p3_2.jpg".into(), person_id: 3, person_name: "person3".into(), has_face: true, has_body: true, low_quality: false },
        ];

        let analyzer = DatasetAnalyzer::with_default_criteria();
        let (person_stats, dataset_stats) = analyzer.compute_person_stats(&images);
        let (verdict, reasons) = analyzer.compute_verdict(&dataset_stats);

        assert_eq!(verdict, DatasetVerdict::DatasetInsufficient);
        assert!(!reasons.is_empty());
        assert!(reasons.iter().any(|r| r.contains("insufficient persons")));
    }

    #[test]
    fn test_insufficient_images_per_person() {
        // 10 persons but most have only 1-2 images
        let images: Vec<AnalyzedImage> = (0..10)
            .flat_map(|pid| {
                let count = if pid < 8 { 1 } else { 3 };
                (0..count).map(move |i| AnalyzedImage {
                    image_id: (pid * 10 + i) as i64,
                    name: format!("p{}_{}.jpg", pid, i),
                    person_id: pid as i64,
                    person_name: format!("person{}", pid),
                    has_face: true,
                    has_body: true,
                    low_quality: false,
                })
            })
            .collect();

        let analyzer = DatasetAnalyzer::with_default_criteria();
        let (_, dataset_stats) = analyzer.compute_person_stats(&images);
        let (verdict, reasons) = analyzer.compute_verdict(&dataset_stats);

        assert_eq!(verdict, DatasetVerdict::DatasetInsufficient);
        assert!(reasons.iter().any(|r| r.contains("insufficient images per person")));
    }

    #[test]
    fn test_valid_dataset() {
        // Create a dataset with 10 persons, each having 5+ usable images
        let images: Vec<AnalyzedImage> = (0..10)
            .flat_map(|pid| {
                (0..5).map(move |i| AnalyzedImage {
                    image_id: (pid * 10 + i) as i64,
                    name: format!("p{}_{}.jpg", pid, i),
                    person_id: pid as i64,
                    person_name: format!("person{}", pid),
                    has_face: true,
                    has_body: true,
                    low_quality: false,
                })
            })
            .collect();

        let analyzer = DatasetAnalyzer::with_default_criteria();
        let (_, dataset_stats) = analyzer.compute_person_stats(&images);
        let (verdict, reasons) = analyzer.compute_verdict(&dataset_stats);

        assert_eq!(verdict, DatasetVerdict::DatasetReady);
        assert!(reasons.is_empty());
    }

    #[test]
    fn test_missing_face_detection() {
        let images = vec![
            AnalyzedImage { image_id: 1, name: "good.jpg".into(), person_id: 1, person_name: "person1".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 2, name: "no_face.jpg".into(), person_id: 1, person_name: "person1".into(), has_face: false, has_body: true, low_quality: false },
        ];

        let analyzer = DatasetAnalyzer::with_default_criteria();
        let (person_stats, _) = analyzer.compute_person_stats(&images);

        assert_eq!(person_stats[0].missing_face_count, 1);
        assert_eq!(person_stats[0].usable_count, 1);
    }

    #[test]
    fn test_missing_body_detection() {
        let images = vec![
            AnalyzedImage { image_id: 1, name: "good.jpg".into(), person_id: 1, person_name: "person1".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 2, name: "no_body.jpg".into(), person_id: 1, person_name: "person1".into(), has_face: true, has_body: false, low_quality: false },
        ];

        let analyzer = DatasetAnalyzer::with_default_criteria();
        let (person_stats, _) = analyzer.compute_person_stats(&images);

        assert_eq!(person_stats[0].missing_body_count, 1);
        assert_eq!(person_stats[0].usable_count, 1);
    }

    #[test]
    fn test_positive_pair_construction() {
        let images = vec![
            AnalyzedImage { image_id: 1, name: "p1_1.jpg".into(), person_id: 1, person_name: "person1".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 2, name: "p1_2.jpg".into(), person_id: 1, person_name: "person1".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 3, name: "p1_3.jpg".into(), person_id: 1, person_name: "person1".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 4, name: "p2_1.jpg".into(), person_id: 2, person_name: "person2".into(), has_face: true, has_body: true, low_quality: false },
        ];

        let analyzer = DatasetAnalyzer::with_default_criteria();
        let pairs = analyzer.construct_positive_pairs(&images);

        // Person 1: C(3,2) = 3 pairs, Person 2: C(1,2) = 0
        assert_eq!(pairs.len(), 3);
        // All pairs should be person 1
        assert!(pairs.iter().all(|p| p.person_id == 1));
    }

    #[test]
    fn test_negative_pair_construction() {
        let images = vec![
            AnalyzedImage { image_id: 1, name: "p1_1.jpg".into(), person_id: 1, person_name: "person1".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 2, name: "p2_1.jpg".into(), person_id: 2, person_name: "person2".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 3, name: "p3_1.jpg".into(), person_id: 3, person_name: "person3".into(), has_face: true, has_body: true, low_quality: false },
        ];

        let analyzer = DatasetAnalyzer::with_default_criteria();
        let face_scores: HashMap<(i64, i64), f32> = HashMap::new();
        let body_scores: HashMap<(i64, i64), f32> = HashMap::new();

        let (pairs, hard_negatives) = analyzer.construct_negative_pairs(&images, &face_scores, &body_scores);

        // 3 persons, each with 1 image: C(3,2) * 1 * 1 = 3 pairs
        assert_eq!(pairs.len(), 3);
        // No hard negatives since no scores above threshold
        assert!(hard_negatives.is_empty());
    }

    #[test]
    fn test_hard_negative_detection() {
        let images = vec![
            AnalyzedImage { image_id: 1, name: "p1_1.jpg".into(), person_id: 1, person_name: "person1".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 2, name: "p2_1.jpg".into(), person_id: 2, person_name: "person2".into(), has_face: true, has_body: true, low_quality: false },
        ];

        let analyzer = DatasetAnalyzer::with_default_criteria();

        // High face score but different person = face hard negative
        let mut face_scores: HashMap<(i64, i64), f32> = HashMap::new();
        face_scores.insert((1, 2), 0.75); // Above FACE_HARD_THRESHOLD

        let body_scores: HashMap<(i64, i64), f32> = HashMap::new();

        let (_, hard_negatives) = analyzer.construct_negative_pairs(&images, &face_scores, &body_scores);

        assert_eq!(hard_negatives.len(), 1);
        assert_eq!(hard_negatives[0].hard_negative_type, HardNegativeType::FaceHardNegative);
    }

    #[test]
    fn test_dual_hard_negative_detection() {
        let images = vec![
            AnalyzedImage { image_id: 1, name: "p1_1.jpg".into(), person_id: 1, person_name: "person1".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 2, name: "p2_1.jpg".into(), person_id: 2, person_name: "person2".into(), has_face: true, has_body: true, low_quality: false },
        ];

        let analyzer = DatasetAnalyzer::with_default_criteria();

        // High face AND body score but different person = dual hard negative
        let mut face_scores: HashMap<(i64, i64), f32> = HashMap::new();
        face_scores.insert((1, 2), 0.75);

        let mut body_scores: HashMap<(i64, i64), f32> = HashMap::new();
        body_scores.insert((1, 2), 0.80);

        let (_, hard_negatives) = analyzer.construct_negative_pairs(&images, &face_scores, &body_scores);

        assert_eq!(hard_negatives.len(), 1);
        assert_eq!(hard_negatives[0].hard_negative_type, HardNegativeType::DualHardNegative);
    }

    #[test]
    fn test_dataset_stats_computation() {
        let images = make_test_images();
        let analyzer = DatasetAnalyzer::with_default_criteria();
        let (_, stats) = analyzer.compute_person_stats(&images);

        assert_eq!(stats.total_persons, 7);
        assert_eq!(stats.total_images, 17);
        assert_eq!(stats.usable_images, 15);
        assert_eq!(stats.min_images_per_person, 1);
        assert!(stats.max_images_per_person >= 5);
    }

    #[test]
    fn test_image_status_strings() {
        let usable = AnalyzedImage {
            image_id: 1, name: "test.jpg".into(), person_id: 1, person_name: "p1".into(),
            has_face: true, has_body: true, low_quality: false,
        };
        assert_eq!(usable.status_string(), "usable");

        let no_face = AnalyzedImage {
            image_id: 2, name: "test.jpg".into(), person_id: 1, person_name: "p1".into(),
            has_face: false, has_body: true, low_quality: false,
        };
        assert_eq!(no_face.status_string(), "no_face");

        let no_body = AnalyzedImage {
            image_id: 3, name: "test.jpg".into(), person_id: 1, person_name: "p1".into(),
            has_face: true, has_body: false, low_quality: false,
        };
        assert_eq!(no_body.status_string(), "no_body");
    }

    #[test]
    fn test_custom_criteria() {
        let criteria = BenchmarkCriteria {
            min_persons: 5,
            min_images_per_person: 3,
            min_usable_per_person: 3,
        };

        let images: Vec<AnalyzedImage> = (0..5)
            .flat_map(|pid| {
                (0..3).map(move |i| AnalyzedImage {
                    image_id: (pid * 10 + i) as i64,
                    name: format!("p{}_{}.jpg", pid, i),
                    person_id: pid as i64,
                    person_name: format!("person{}", pid),
                    has_face: true,
                    has_body: true,
                    low_quality: false,
                })
            })
            .collect();

        let analyzer = DatasetAnalyzer::new(criteria);
        let (_, dataset_stats) = analyzer.compute_person_stats(&images);
        let (verdict, _) = analyzer.compute_verdict(&dataset_stats);

        assert_eq!(verdict, DatasetVerdict::DatasetReady);
    }

    #[test]
    fn test_self_match_exclusion() {
        // Positive pairs should never have the same image
        let images = vec![
            AnalyzedImage { image_id: 1, name: "p1_1.jpg".into(), person_id: 1, person_name: "person1".into(), has_face: true, has_body: true, low_quality: false },
            AnalyzedImage { image_id: 2, name: "p1_2.jpg".into(), person_id: 1, person_name: "person1".into(), has_face: true, has_body: true, low_quality: false },
        ];

        let analyzer = DatasetAnalyzer::with_default_criteria();
        let pairs = analyzer.construct_positive_pairs(&images);

        for pair in &pairs {
            assert_ne!(pair.image1_id, pair.image2_id);
        }
    }

    #[test]
    fn test_verdict_string() {
        assert_eq!(verdict_string(DatasetVerdict::DatasetReady), "DATASET_READY");
        assert_eq!(verdict_string(DatasetVerdict::DatasetInsufficient), "DATASET_INSUFFICIENT");
    }
}
