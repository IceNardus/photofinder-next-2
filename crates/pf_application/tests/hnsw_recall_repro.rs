//! 临时诊断（非生产）：复现"图库内查询图自匹配不召回"。
//!
//! 根因已定位并修复（hnsw_index.rs save/load 文件名一致性 + desktop_setup 启动 guard）：
//! hnsw_rs 0.3.4 的 `load_hnsw` 会把 `datamap_opt` 置为 true,使第二次 `save()` 写
//! `face-<rand>.hnsw.data` 随机后缀文件,而 `load()` 只读规范名 `face.hnsw.data`
//! → 规范文件停留在首次 dump(stale),搜索索引缺大部分 DB face → 查询图搜不到自己。
//!
//! 本文件保留两个诊断：
//!   1) live 索引状态 + 修复路径端到端验证（用真实 512-d embeddings）
//!   2) hnsw_rs 在 7 节点小图上的 self-recall（排除 HNSW 随机召回假设）

use std::path::PathBuf;
use std::sync::Arc;

use pf_database::{builtin_migrations, Database, FaceStatus};
use pf_vector::{HnswIndex, VectorIndex};

const DB_PATH: &str = "/Users/mac/Library/Application Support/PhotoFinderNext/photofinder.db";

fn tmpdir(prefix: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pf-recall-{}-{}-{}",
        prefix,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&p).unwrap();
    p
}

fn load_db_items() -> Vec<(i64, Vec<f32>)> {
    let db = Database::open(std::path::Path::new(DB_PATH), builtin_migrations()).expect("open db");
    let mut items: Vec<(i64, Vec<f32>)> = Vec::new();
    db.transaction(|tx| {
        let faces = tx.faces().list_by_status(FaceStatus::Indexed)?;
        for f in &faces {
            let vid = f.vector_id.expect("indexed face should have vector_id");
            let emb = tx
                .faces()
                .get_embedding(f.id, "arcface-w600k-r50")?
                .expect("indexed face should have embedding");
            items.push((vid, emb));
        }
        Ok::<_, pf_database::DatabaseError>(())
    })
    .unwrap();
    items
}

#[test]
fn live_index_state_and_fix_path() {
    let items = load_db_items();
    eprintln!("db indexed faces: {}", items.len());

    // 1) 诊断：live 规范文件当前是否含全部 DB face。
    //    修复后 app 首次启动会因 guard(len < db_indexed) 触发 rebuild + save()
    //    归一到规范名;此处只报告,不做硬断言(取决于 app 是否已跑过)。
    let index_dir = std::path::Path::new(
        "/Users/mac/Library/Application Support/PhotoFinderNext/index",
    );
    let idx = HnswIndex::new(512, index_dir, "face");
    if idx.load().is_ok() {
        let mut found = 0usize;
        let mut missing: Vec<i64> = Vec::new();
        for (vid, emb) in &items {
            let hits = idx.search(emb, 50).unwrap();
            if hits.iter().any(|h| h.id == *vid) {
                found += 1;
            } else {
                missing.push(*vid);
            }
        }
        eprintln!(
            "live face.hnsw: found={found}/{} missing={missing:?}",
            items.len()
        );
    } else {
        eprintln!("live face.hnsw: load failed (no canonical file yet)");
    }

    // 2) 修复路径端到端验证：
    //    seed(写规范文件) → load(datamap_opt=true) → 插入其余 face →
    //    save(随机后缀→归一到规范名) → reload 必须全量读回。
    //    无 save() 归一修复时,reload 只会读到 seed 的 1 个 face → 断言失败。
    let dir = tmpdir("fix");
    let seed = HnswIndex::new(512, &dir, "face");
    seed.insert(items[0].0, &items[0].1).unwrap();
    seed.save().expect("seed save");
    drop(seed);

    let full = HnswIndex::new(512, &dir, "face");
    full.load().expect("seed load");
    for (vid, emb) in items.iter().skip(1) {
        full.insert(*vid, emb).unwrap();
    }
    full.save().expect("full save");

    let loaded = HnswIndex::new(512, &dir, "face");
    loaded.load().expect("reload");
    assert_eq!(loaded.len(), items.len(), "save/load lost faces");
    for (vid, emb) in &items {
        let hits = loaded.search(emb, 50).unwrap();
        assert!(
            hits.iter().any(|h| h.id == *vid),
            "face {vid} missing after save/load"
        );
    }
    eprintln!("fix-path roundtrip: all {} faces survive save/load", items.len());
}

#[test]
fn hnsw_self_recall_on_real_library() {
    let items = load_db_items();
    eprintln!("loaded {} indexed faces", items.len());
    assert!(!items.is_empty(), "live DB should have indexed faces");

    // 反复建图测召回(与 cluster_v2 flaky 同源假设:hnsw_rs 小图随机召回)。
    let trials = 300usize;
    let mut self_miss = 0usize;
    let mut miss_faces: Vec<Vec<i64>> = Vec::new();
    for t in 0..trials {
        let dir = tmpdir(&format!("t{t}"));
        let idx = Arc::new(HnswIndex::new(512, dir, "face"));
        for (vid, emb) in &items {
            idx.insert(*vid, emb).unwrap();
        }
        let mut missed_this_graph: Vec<i64> = Vec::new();
        for (vid, emb) in &items {
            // 生产: fetch_k = max(top_k*4, 4)；top_k=30 → 120。
            let hits = idx.search(emb, 120).unwrap();
            if !hits.iter().any(|h| h.id == *vid) {
                self_miss += 1;
                missed_this_graph.push(*vid);
            }
        }
        if !missed_this_graph.is_empty() {
            miss_faces.push(missed_this_graph);
        }
    }
    eprintln!(
        "trials={trials} self_miss={self_miss} ({:.2}%)  graphs_with_miss={}/{}",
        self_miss as f32 * 100.0 / (trials as f32 * items.len() as f32),
        miss_faces.len(),
        trials
    );
    if let Some(first) = miss_faces.first() {
        eprintln!("example missed vids (graph 0): {first:?}");
    }
    // hnsw_rs 在 ~14 节点小图上偶发随机漏节点(实测 ~0.02-0.05%/查询,raise ef 无效,
    // 属建图连通性随机性)。断言目标是"不系统性漏"(miss 率 << 1%),而非 0。
    let miss_rate = self_miss as f32 / (trials as f32 * items.len() as f32);
    assert!(
        miss_rate < 0.01,
        "HNSW self-recall miss rate too high: {:.4}%",
        miss_rate * 100.0
    );
}
