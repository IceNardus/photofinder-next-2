//! iOS Person Clustering - Multi-Prototype Implementation
//!
//! Implements multi-prototype per person (frontal, profile, high_quality)
//! reusing desktop PersonService core algorithm principles.

use crate::ffi::DB;
use pf_ai::face::FaceFeature;
use pf_database::PooledConnection;
use rusqlite::params;
use serde::Serialize;

/// Clustering configuration
#[derive(Debug, Clone)]
pub struct ClusterConfig {
    /// Cosine similarity threshold (>= this means same person)
    pub similarity_threshold: f32,
    /// Anti-chaining margin (top2 - top1 < margin means ambiguous)
    pub chaining_margin: f32,
    /// Maximum prototypes per person
    pub max_prototypes: usize,
    /// Pose bucket yaw threshold (degrees)
    pub pose_yaw_threshold: f32,
    /// Prototype deduplication cosine threshold
    pub dedup_threshold: f32,
    /// Minimum quality score to consider assignment
    pub min_quality_for_match: f32,
    /// Below this quality, face is considered "low quality"
    pub low_quality_threshold: f32,
    /// Extra threshold penalty for low quality faces
    pub low_quality_penalty: f32,
}

impl Default for ClusterConfig {
    fn default() -> Self {
        Self {
            similarity_threshold: 0.30,
            chaining_margin: 0.05,
            max_prototypes: 8,
            pose_yaw_threshold: 15.0,
            dedup_threshold: 0.90,
            // Quality thresholds
            min_quality_for_match: 0.3,  // Minimum quality to consider assignment
            low_quality_threshold: 0.5,  // Below this, face is "low quality"
            low_quality_penalty: 0.1,   // Extra threshold for low quality faces
        }
    }
}

/// Clustering result
#[derive(Debug, Serialize)]
pub struct ClusterResult {
    pub assigned: usize,
    pub created: usize,
    pub failed: usize,
}

/// Person info for listing
#[derive(Debug, Serialize)]
pub struct PersonInfo {
    pub id: i64,
    pub name: String,
    pub face_count: i64,
}

/// Face info for a person
#[derive(Debug, Serialize)]
pub struct PersonFaceInfo {
    pub face_id: i64,
    pub photo_id: i64,
}

/// Face data with embedding
#[derive(Debug)]
#[allow(dead_code)]
struct FaceWithEmbedding {
    face_id: i64,
    photo_id: i64,
    embedding: Vec<f32>,
    quality: f32,
    yaw: Option<f32>,
}

/// Person with their prototypes
#[derive(Debug)]
struct PersonWithPrototypes {
    person_id: i64,
    prototypes: Vec<Prototype>,
}

/// A single prototype
#[derive(Debug)]
#[allow(dead_code)]
struct Prototype {
    embedding: Vec<f32>,
    pose: String,  // "frontal", "left_profile", "right_profile", "high_quality"
    quality: f32,
}

/// Classify yaw into pose bucket
fn classify_pose(yaw: Option<f32>, threshold: f32) -> String {
    match yaw {
        None => "frontal".to_string(),
        Some(y) if y.abs() < threshold => "frontal".to_string(),
        Some(y) if y < 0.0 => "left_profile".to_string(),
        Some(_) => "right_profile".to_string(),
    }
}

/// Compute cosine similarity between two normalized embeddings
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Convert blob bytes to f32 vector
fn blob_to_embedding(blob: &[u8]) -> Vec<f32> {
    blob.chunks(4)
        .filter(|c| c.len() == 4)
        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
        .collect()
}

/// Get pooled connection from DB
fn get_conn() -> Result<PooledConnection, String> {
    let db = DB.lock().unwrap();
    match db.as_ref() {
        Some(db) => db.connection().map_err(|e| e.to_string()),
        None => Err("Database not initialized".to_string()),
    }
}

/// Parse yaw from yaw_pitch_roll string "yaw,pitch,roll"
fn parse_yaw(yaw_str: &str) -> Option<f32> {
    yaw_str.split(',').next().and_then(|s| s.parse().ok())
}

/// Get all unassigned faces with their data
fn get_unassigned_faces(conn: &PooledConnection) -> Result<Vec<FaceWithEmbedding>, String> {
    let mut stmt = conn.prepare(
        "SELECT f.id, f.photo_id, e.vector, f.quality_score, f.yaw_pitch_roll
         FROM faces f
         JOIN face_embeddings e ON e.face_id = f.id
         WHERE f.id NOT IN (SELECT face_id FROM face_person_assignments)
         ORDER BY f.id"
    ).map_err(|e| e.to_string())?;

    let faces: Vec<FaceWithEmbedding> = stmt
        .query_map([], |row| {
            let blob: Vec<u8> = row.get(2)?;
            let embedding = blob_to_embedding(&blob);
            let yaw_str: String = row.get(4)?;
            let yaw = parse_yaw(&yaw_str);
            Ok(FaceWithEmbedding {
                face_id: row.get(0)?,
                photo_id: row.get(1)?,
                embedding,
                quality: row.get::<_, Option<f32>>(3)?.unwrap_or(0.5),
                yaw,
            })
        })
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

    Ok(faces)
}

/// Get all existing persons with their prototypes
fn get_persons_with_prototypes(conn: &PooledConnection) -> Result<Vec<PersonWithPrototypes>, String> {
    let mut stmt = conn.prepare(
        "SELECT p.id, pp.pose, e.vector, f.quality_score
         FROM persons p
         JOIN person_prototypes pp ON pp.person_id = p.id
         JOIN face_embeddings e ON e.face_id = pp.face_id
         JOIN faces f ON f.id = pp.face_id
         ORDER BY p.id"
    ).map_err(|e| e.to_string())?;

    let mut person_map: std::collections::HashMap<i64, PersonWithPrototypes> = std::collections::HashMap::new();

    let mut rows = stmt.query([]).map_err(|e| e.to_string())?;
    while let Some(row_result) = rows.next().map_err(|e| e.to_string())? {
        let row = row_result;
        let pid: i64 = row.get(0).map_err(|e| e.to_string())?;
        let pose: String = row.get(1).map_err(|e| e.to_string())?;
        let blob: Vec<u8> = row.get(2).map_err(|e| e.to_string())?;
        let embedding = blob_to_embedding(&blob);
        let quality: f32 = row.get::<_, Option<f32>>(3).map_err(|e| e.to_string())?.unwrap_or(0.5);

        let proto = Prototype { embedding, pose: pose.clone(), quality };

        person_map.entry(pid).or_insert_with(|| PersonWithPrototypes {
            person_id: pid,
            prototypes: Vec::new(),
        }).prototypes.push(proto);
    }

    Ok(person_map.into_values().collect())
}

/// Insert a new person
fn insert_person(conn: &PooledConnection) -> Result<i64, String> {
    conn.execute(
        "INSERT INTO persons (name, face_count) VALUES ('Unknown', 0)",
        [],
    ).map_err(|e| e.to_string())?;
    Ok(conn.last_insert_rowid())
}

/// Update person's face count
fn update_person_face_count(conn: &PooledConnection, person_id: i64) -> Result<(), String> {
    let count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM face_person_assignments WHERE person_id = ?",
            params![person_id],
            |row| row.get(0),
        )
        .map_err(|e| e.to_string())?;

    conn.execute(
        "UPDATE persons SET face_count = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
        params![count, person_id],
    ).map_err(|e| e.to_string())?;
    Ok(())
}

/// Insert face-person assignment
fn insert_assignment(
    conn: &PooledConnection,
    face_id: i64,
    person_id: i64,
    similarity: f32,
) -> Result<(), String> {
    conn.execute(
        "INSERT OR REPLACE INTO face_person_assignments (face_id, person_id, similarity, confidence, method, status) VALUES (?1, ?2, ?3, ?4, 'cosine', 'active')",
        params![face_id, person_id, similarity, similarity],
    ).map_err(|e| e.to_string())?;

    update_person_face_count(conn, person_id)?;
    Ok(())
}

/// Get all face embeddings with quality for a person
/// This is used for validation - comparing against ALL enrolled faces with quality weighting
fn get_person_all_faces_with_quality(
    conn: &PooledConnection,
    person_id: i64,
) -> Result<Vec<(Vec<f32>, f32)>, String> {
    let mut stmt = conn.prepare(
        "SELECT e.vector, f.quality_score
         FROM face_person_assignments fpa
         JOIN face_embeddings e ON e.face_id = fpa.face_id
         JOIN faces f ON f.id = fpa.face_id
         WHERE fpa.person_id = ?1"
    ).map_err(|e| e.to_string())?;

    let faces: Vec<(Vec<f32>, f32)> = stmt
        .query_map(params![person_id], |row| {
            let blob: Vec<u8> = row.get(0)?;
            let quality: f32 = row.get::<_, Option<f32>>(1)?.unwrap_or(0.5);
            Ok((blob_to_embedding(&blob), quality))
        })
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

    Ok(faces)
}

/// Validate a candidate assignment by checking against ALL enrolled faces
/// Returns quality-weighted cosine similarity
/// Higher quality faces contribute more to the score
fn validate_against_all_faces(
    query: &[f32],
    person_id: i64,
    conn: &PooledConnection,
) -> Result<f32, String> {
    let faces = get_person_all_faces_with_quality(conn, person_id)?;
    if faces.is_empty() {
        return Ok(0.0);
    }

    // Compute quality-weighted max similarity
    // Each face's similarity is weighted by its quality score
    let mut best_weighted_score = 0.0f32;
    let mut best_raw_score = 0.0f32;

    for (embedding, quality) in &faces {
        let sim = cosine_sim(query, embedding);
        // Weighted score: raw_similarity * quality
        let weighted = sim * quality;
        if weighted > best_weighted_score {
            best_weighted_score = weighted;
            best_raw_score = sim;
        }
    }

    // Also consider raw max for threshold comparison
    // Use raw max for threshold check but weighted for ranking
    Ok(best_raw_score)
}

/// Add initial centroid for a new person
fn add_centroid(
    conn: &PooledConnection,
    person_id: i64,
    embedding: &[f32],
) -> Result<(), String> {
    let embedding_blob: Vec<u8> = embedding.iter()
        .flat_map(|&f| f.to_le_bytes().to_vec())
        .collect();

    conn.execute(
        "INSERT INTO person_centroids (person_id, embedding, face_count) VALUES (?1, ?2, 1)",
        params![person_id, embedding_blob],
    ).map_err(|e| e.to_string())?;

    Ok(())
}

/// Merge new embedding into person's centroid
/// centroid = (centroid + new_emb) * 0.5, then normalize
fn merge_centroid(
    conn: &PooledConnection,
    person_id: i64,
    new_embedding: &[f32],
) -> Result<(), String> {
    // Get current centroid
    let current_blob: Option<Vec<u8>> = conn
        .query_row(
            "SELECT embedding FROM person_centroids WHERE person_id = ?",
            params![person_id],
            |row| row.get(0),
        )
        .ok();

    let merged_embedding = if let Some(blob) = current_blob {
        let current = blob_to_embedding(&blob);
        if current.len() != new_embedding.len() {
            return Err("Embedding dimension mismatch".to_string());
        }

        // Merge: (current + new) * 0.5
        let merged: Vec<f32> = current.iter()
            .zip(new_embedding.iter())
            .map(|(c, n)| (*c + *n) * 0.5)
            .collect();

        // Normalize
        let norm: f32 = merged.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
        merged.iter().map(|x| x / norm).collect()
    } else {
        // No existing centroid - use new embedding directly
        new_embedding.to_vec()
    };

    let embedding_blob: Vec<u8> = merged_embedding.iter()
        .flat_map(|&f| f.to_le_bytes().to_vec())
        .collect();

    // Update face count and centroid
    conn.execute(
        "UPDATE person_centroids SET embedding = ?1, face_count = face_count + 1, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE person_id = ?2",
        params![embedding_blob, person_id],
    ).map_err(|e| e.to_string())?;

    Ok(())
}

/// Get all person centroids
fn get_all_centroids(
    conn: &PooledConnection,
) -> Result<Vec<(i64, Vec<f32>)>, String> {
    let mut stmt = conn.prepare(
        "SELECT person_id, embedding FROM person_centroids"
    ).map_err(|e| e.to_string())?;

    let centroids: Vec<(i64, Vec<f32>)> = stmt
        .query_map([], |row| {
            let pid: i64 = row.get(0)?;
            let blob: Vec<u8> = row.get(1)?;
            Ok((pid, blob_to_embedding(&blob)))
        })
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

    Ok(centroids)
}

/// Find best matching person using centroids
fn find_best_by_centroid(
    query: &[f32],
    centroids: &[(i64, Vec<f32>)],
) -> Option<(i64, f32)> {
    centroids.iter()
        .map(|(pid, emb)| (*pid, cosine_sim(query, emb)))
        .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
}

/// Select and insert prototypes for a person based on their faces
fn rebuild_person_prototypes(
    conn: &PooledConnection,
    person_id: i64,
    config: &ClusterConfig,
) -> Result<(), String> {
    // Get all faces for this person with their embeddings
    let mut stmt = conn.prepare(
        "SELECT f.id, e.vector, f.quality_score, f.yaw_pitch_roll
         FROM faces f
         JOIN face_person_assignments fpa ON fpa.face_id = f.id
         JOIN face_embeddings e ON e.face_id = f.id
         WHERE fpa.person_id = ?1"
    ).map_err(|e| e.to_string())?;

    let faces: Vec<FaceWithEmbedding> = stmt
        .query_map(params![person_id], |row| {
            let blob: Vec<u8> = row.get(1)?;
            let yaw_str: String = row.get(3)?;
            Ok(FaceWithEmbedding {
                face_id: row.get(0)?,
                photo_id: 0,  // not needed for prototype selection
                embedding: blob_to_embedding(&blob),
                quality: row.get::<_, Option<f32>>(2)?.unwrap_or(0.5),
                yaw: parse_yaw(&yaw_str),
            })
        })
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

    if faces.is_empty() {
        return Ok(());
    }

    // Clear existing prototypes
    conn.execute(
        "DELETE FROM person_prototypes WHERE person_id = ?",
        params![person_id],
    ).map_err(|e| e.to_string())?;

    // Bucket faces by pose
    let mut frontal: Vec<&FaceWithEmbedding> = Vec::new();
    let mut left_profile: Vec<&FaceWithEmbedding> = Vec::new();
    let mut right_profile: Vec<&FaceWithEmbedding> = Vec::new();
    let mut all_faces: Vec<&FaceWithEmbedding> = Vec::new();

    for face in &faces {
        let pose = classify_pose(face.yaw, config.pose_yaw_threshold);
        all_faces.push(face);
        match pose.as_str() {
            "frontal" => frontal.push(face),
            "left_profile" => left_profile.push(face),
            "right_profile" => right_profile.push(face),
            _ => {}
        }
    }

    // Select best per bucket with deduplication
    let mut selected: Vec<&FaceWithEmbedding> = Vec::new();

    for bucket in &[&frontal, &left_profile, &right_profile] {
        if bucket.is_empty() {
            continue;
        }

        // Sort by quality descending
        let mut sorted = bucket.to_vec();
        sorted.sort_by(|a, b| b.quality.partial_cmp(&a.quality).unwrap());

        // Greedy selection with deduplication
        for face in sorted {
            let is_dup = selected.iter().any(|s| {
                cosine_sim(&s.embedding, &face.embedding) >= config.dedup_threshold
            });
            if !is_dup && selected.len() < config.max_prototypes {
                selected.push(face);
            }
        }
    }

    // Add global best quality if not already selected
    let mut sorted_all = all_faces.to_vec();
    sorted_all.sort_by(|a, b| b.quality.partial_cmp(&a.quality).unwrap());
    for face in sorted_all {
        if selected.len() >= config.max_prototypes {
            break;
        }
        let is_dup = selected.iter().any(|s| {
            s.face_id == face.face_id ||
            cosine_sim(&s.embedding, &face.embedding) >= config.dedup_threshold
        });
        if !is_dup {
            selected.push(face);
        }
    }

    // Insert prototypes
    for face in selected {
        let pose = classify_pose(face.yaw, config.pose_yaw_threshold);
        let embedding_blob: Vec<u8> = face.embedding.iter()
            .flat_map(|&f| f.to_le_bytes().to_vec())
            .collect();

        conn.execute(
            "INSERT INTO person_prototypes (person_id, face_id, embedding, is_primary, pose) VALUES (?1, ?2, ?3, 1, ?4)",
            params![person_id, face.face_id, embedding_blob, pose],
        ).map_err(|e| e.to_string())?;
    }

    Ok(())
}

/// Perform full clustering of all unassigned faces
/// Uses two-phase matching:
/// 1. Fast filter using prototype similarity
/// 2. Validation against ALL enrolled faces
pub fn cluster_all(config: ClusterConfig) -> Result<ClusterResult, String> {
    let conn = get_conn()?;

    eprintln!("[CLUSTER] Starting clustering with config: {:?}", config);

    // Get all unassigned faces
    let unassigned_faces = get_unassigned_faces(&conn)?;
    eprintln!("[CLUSTER] Found {} unassigned faces", unassigned_faces.len());

    if unassigned_faces.is_empty() {
        return Ok(ClusterResult { assigned: 0, created: 0, failed: 0 });
    }

    let mut assigned = 0;
    let mut created = 0;
    let failed = 0;

    // Process each unassigned face
    for face in unassigned_faces {
        // Quality pre-check: skip very low quality faces
        if face.quality < config.min_quality_for_match {
            eprintln!("[CLUSTER] Face {} quality {:.3} < {:.3}, skipped",
                      face.face_id, face.quality, config.min_quality_for_match);
            continue;
        }

        // Calculate effective threshold based on face quality
        // Low quality faces need higher similarity to match (stricter threshold)
        let effective_threshold = if face.quality < config.low_quality_threshold {
            config.similarity_threshold + config.low_quality_penalty
        } else {
            config.similarity_threshold
        };

        // Get current persons with prototypes
        let persons = get_persons_with_prototypes(&conn)?;

        if persons.is_empty() {
            // No existing persons - create first one
            let new_pid = insert_person(&conn)?;
            insert_assignment(&conn, face.face_id, new_pid, 1.0)?;
            rebuild_person_prototypes(&conn, new_pid, &config)?;
            add_centroid(&conn, new_pid, &face.embedding)?;
            created += 1;
            eprintln!("[CLUSTER] Created first person {} for face {}", new_pid, face.face_id);
            continue;
        }

        // Phase 1: Fast filter - find candidates using prototype similarity
        let prototype_scores: Vec<(i64, f32)> = persons.iter()
            .filter(|p| !p.prototypes.is_empty())
            .map(|p| {
                let max_sim = p.prototypes.iter()
                    .map(|proto| cosine_sim(&face.embedding, &proto.embedding))
                    .fold(0.0f32, f32::max);
                (p.person_id, max_sim)
            })
            .filter(|(_, score)| *score >= config.similarity_threshold * 0.8)  // Lower threshold for candidate filter
            .collect();

        if prototype_scores.is_empty() {
            // No candidates - create new person
            let new_pid = insert_person(&conn)?;
            insert_assignment(&conn, face.face_id, new_pid, 1.0)?;
            rebuild_person_prototypes(&conn, new_pid, &config)?;
            add_centroid(&conn, new_pid, &face.embedding)?;
            created += 1;
            eprintln!("[CLUSTER] Face {} no prototype candidates, created new person {}",
                      face.face_id, new_pid);
            continue;
        }

        // Phase 2: Validate candidates against ALL enrolled faces
        let mut validated_scores: Vec<(i64, f32)> = Vec::new();
        for (person_id, proto_score) in prototype_scores {
            // Validate by checking against ALL faces of this person
            match validate_against_all_faces(&face.embedding, person_id, &conn) {
                Ok(all_score) => {
                    eprintln!("[CLUSTER] Face {}: person {} proto_score={:.3} all_score={:.3}",
                              face.face_id, person_id, proto_score, all_score);
                    validated_scores.push((person_id, all_score));
                }
                Err(e) => {
                    eprintln!("[CLUSTER] Face {}: person {} validation error: {}",
                              face.face_id, person_id, e);
                }
            }
        }

        if validated_scores.is_empty() {
            // No valid candidates - create new person
            let new_pid = insert_person(&conn)?;
            insert_assignment(&conn, face.face_id, new_pid, 1.0)?;
            rebuild_person_prototypes(&conn, new_pid, &config)?;
            add_centroid(&conn, new_pid, &face.embedding)?;
            created += 1;
            eprintln!("[CLUSTER] Face {} no valid candidates after validation, created new person {}",
                      face.face_id, new_pid);
            continue;
        }

        // Sort by validation score
        validated_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        let best = validated_scores.first().copied();
        let second_best = validated_scores.get(1).map(|&(_, s)| s);

        match best {
            Some((person_id, score)) if score >= effective_threshold => {
                // Check for ambiguity (chaining prevention)
                let is_ambiguous = second_best
                    .map(|s2| score - s2 < config.chaining_margin)
                    .unwrap_or(false);

                if is_ambiguous {
                    // Ambiguous - create new person
                    let new_pid = insert_person(&conn)?;
                    insert_assignment(&conn, face.face_id, new_pid, 1.0)?;
                    rebuild_person_prototypes(&conn, new_pid, &config)?;
            add_centroid(&conn, new_pid, &face.embedding)?;
                    created += 1;
                    eprintln!("[CLUSTER] Face {} ambiguous (top={:.3}, second={:.3}), created new person {}",
                              face.face_id, score, second_best.unwrap_or(0.0), new_pid);
                } else {
                    // Assign to best match
                    insert_assignment(&conn, face.face_id, person_id, score)?;
                    merge_centroid(&conn, person_id, &face.embedding)?;
                    rebuild_person_prototypes(&conn, person_id, &config)?;
                    assigned += 1;
                    eprintln!("[CLUSTER] Face {} assigned to person {} (validated_score={:.3})",
                              face.face_id, person_id, score);
                }
            }
            _ => {
                // Score below threshold - create new person
                let new_pid = insert_person(&conn)?;
                insert_assignment(&conn, face.face_id, new_pid, 1.0)?;
                rebuild_person_prototypes(&conn, new_pid, &config)?;
            add_centroid(&conn, new_pid, &face.embedding)?;
                created += 1;
                eprintln!("[CLUSTER] Face {} best score {:.3} below threshold, created new person {}",
                          face.face_id, best.map(|(_, s)| s).unwrap_or(0.0), new_pid);
            }
        }
    }

    eprintln!("[CLUSTER] Done: assigned={}, created={}, failed={}", assigned, created, failed);
    Ok(ClusterResult { assigned, created, failed })
}

/// List all persons
pub fn list_persons() -> Result<Vec<PersonInfo>, String> {
    let conn = get_conn()?;

    let mut stmt = conn.prepare(
        "SELECT id, name, face_count FROM persons ORDER BY face_count DESC, id"
    ).map_err(|e| e.to_string())?;

    let persons: Vec<PersonInfo> = stmt
        .query_map([], |row| {
            Ok(PersonInfo {
                id: row.get(0)?,
                name: row.get(1)?,
                face_count: row.get(2)?,
            })
        })
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

    Ok(persons)
}

/// Rename a person
pub fn rename_person(person_id: i64, new_name: &str) -> Result<(), String> {
    let conn = get_conn()?;

    let rows = conn.execute(
        "UPDATE persons SET name = ?, updated_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now') WHERE id = ?",
        params![new_name, person_id],
    ).map_err(|e| e.to_string())?;

    if rows == 0 {
        return Err(format!("Person {} not found", person_id));
    }

    Ok(())
}

/// Get all faces for a person
pub fn get_person_faces(person_id: i64) -> Result<Vec<PersonFaceInfo>, String> {
    let conn = get_conn()?;

    let mut stmt = conn.prepare(
        "SELECT f.id, f.photo_id
         FROM faces f
         JOIN face_person_assignments fpa ON fpa.face_id = f.id
         WHERE fpa.person_id = ?
         ORDER BY fpa.assigned_at DESC"
    ).map_err(|e| e.to_string())?;

    let faces: Vec<PersonFaceInfo> = stmt
        .query_map(params![person_id], |row| {
            Ok(PersonFaceInfo {
                face_id: row.get(0)?,
                photo_id: row.get(1)?,
            })
        })
        .map_err(|e| e.to_string())?
        .filter_map(|r| r.ok())
        .collect();

    Ok(faces)
}

/// 多脸融合 - 根据质量加权平均多个人脸 embedding
/// 与桌面版 pf_application/src/search.rs::fuse_face_embeddings 对齐
pub fn fuse_face_embeddings(features: &[FaceFeature]) -> Vec<f32> {
    if features.is_empty() {
        return Vec::new();
    }

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
        return features[0].embedding.as_slice().to_vec();
    }

    let mut fused = vec![0.0f32; dim];
    for (f, w) in features.iter().zip(weights.iter()) {
        let weight = w / total;
        for (out, v) in fused.iter_mut().zip(f.embedding.as_slice().iter()) {
            *out += v * weight;
        }
    }

    // L2 normalize
    let norm = fused.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for v in &mut fused {
            *v /= norm;
        }
    }

    fused
}
