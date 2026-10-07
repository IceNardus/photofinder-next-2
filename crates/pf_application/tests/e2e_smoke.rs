//! 端到端 smoke: 真实 SCRFD + ArcFace 模型 + ScanService → IndexImage executor → SearchService。
//!
//! 用本地 `/Users/mac/Downloads` 下 11 张 Pexels 人像照, 隔离到 tmpdir 跑完整闭环,
//! 不污染真实 data_dir(`~/Library/Application Support/PhotoFinderNext/`)。
//!
//! 运行:
//! ```bash
//! cargo test --release -p pf_application --test e2e_smoke -- --ignored --nocapture
//! ```
//!
//! 验证重构后管线(模型加载、scan 入库、scheduler/worker/IndexImageExecutor、
//! HNSW 插入、SearchService 多脸融合检索)在真实模型上端到端可用。

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use pf_ai::{
    ArcFaceEmbedder, FacePipeline, QualityFilter, ScrfdDetector, SimpleAligner,
};
use pf_application::executors::register_executors;
use pf_application::index::IndexService;
use pf_application::person::{ClusterPolicy, PersonService};
use pf_application::prototype_service::PrototypeService;
use pf_application::scan::ScanService;
use pf_application::search::SearchService;
use pf_config::Config;
use pf_database::{builtin_migrations, Database};
use pf_platform::{FileSystemPhotoProvider, PhotoProvider};
use pf_task::{ExecutorRegistry, SqliteTaskScheduler, TaskScheduler};
use pf_vector::{HnswIndex, VectorIndex};

const DOWNLOADS: &str = "/Users/mac/Downloads";

/// 11 张 Pexels 人像照(face_debug.rs 已用过 3 张 jaqor 验证 pipeline)。
const PEXELS: &[&str] = &[
    "pexels-cottonbro-5900525.jpg",
    "pexels-daria-voronkov-381938591-14723650.jpg",
    "pexels-daria-voronkov-381938591-14723672.jpg",
    "pexels-denniz-futalan-339724-3378435.jpg",
    "pexels-jaqor-33601811.jpg",
    "pexels-jaqor-33601831.jpg",
    "pexels-jaqor-33601835.jpg",
    "pexels-joelle-s-2162497381-38263248.jpg",
    "pexels-peterdanthy-33692605.jpg",
    "pexels-soc-nang-d-ng-2150345854-38142867.jpg",
    "pexels-yi-ren-57040649-33026322.jpg",
];

/// 解析模型路径: 先仓库 `models/`, 后 Application Support, 后旧 Cache。
fn resolve_model(name: &str) -> PathBuf {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let candidates = [
        workspace.join("models").join(name),
        PathBuf::from(
            "/Users/mac/Library/Application Support/PhotoFinderNext/resources/models",
        )
        .join(name),
        PathBuf::from("/Users/mac/Library/Application Support/PhotoFinderNext/models")
            .join(name),
    ];
    candidates
        .into_iter()
        .find(|p| p.exists())
        .unwrap_or_else(|| workspace.join("models").join(name))
}

fn tmpdir(prefix: &str) -> PathBuf {
    let mut p = std::env::temp_dir();
    p.push(format!(
        "pf-e2e-{}-{}-{}",
        prefix,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    fs::create_dir_all(&p).unwrap();
    p
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore]
async fn e2e_scan_index_search_on_local_pexels() {
    // ----- 1) 解析模型路径 -----
    let scrfd_path = resolve_model("scrfd_500m_bnkps.onnx");
    let arcface_path = resolve_model("w600k_r50.onnx");
    eprintln!("SCRFD   = {}", scrfd_path.display());
    eprintln!("ArcFace = {}", arcface_path.display());
    assert!(scrfd_path.exists(), "SCRFD model not found: {}", scrfd_path.display());
    assert!(arcface_path.exists(), "ArcFace model not found: {}", arcface_path.display());

    // ----- 2) tmpdir + photos -----
    let tmp = tmpdir("smoke");
    let photos_dir = tmp.join("photos");
    fs::create_dir(&photos_dir).unwrap();
    let downloads = PathBuf::from(DOWNLOADS);
    let mut copied = 0usize;
    for name in PEXELS {
        let src = downloads.join(name);
        if !src.exists() {
            eprintln!("WARN 缺图: {}", src.display());
            continue;
        }
        fs::copy(&src, photos_dir.join(name)).unwrap();
        copied += 1;
    }
    eprintln!("复制 {} / {} 张图到 {}", copied, PEXELS.len(), photos_dir.display());
    assert!(copied >= 8, "至少复制 8 张 Pexels 图");

    // ----- 3) 真实模型(face_debug.rs 范本) -----
    let detector = ScrfdDetector::load(&scrfd_path).expect("load SCRFD");
    let aligner = Arc::new(SimpleAligner::new());
    let embedder = ArcFaceEmbedder::load(&arcface_path).expect("load ArcFace");
    let qf = QualityFilter::from_config(0.30, 20, 0.45, 30.0);
    let pipeline = Arc::new(FacePipeline::new(detector, aligner, embedder, qf));

    // ----- 4) DB + HNSW(tmpdir 隔离) -----
    let db_dir = tmp.join("db");
    fs::create_dir(&db_dir).unwrap();
    let db_path = db_dir.join("smoke.db");
    let db = Arc::new(Database::open(&db_path, builtin_migrations()).unwrap());

    let hnsw_dir = tmp.join("hnsw");
    fs::create_dir(&hnsw_dir).unwrap();
    let face_index: Arc<dyn VectorIndex> = Arc::new(HnswIndex::new(512, hnsw_dir.clone(), "face"));

    // ----- 5) Config + PhotoProvider -----
    let config = Arc::new(Config::default());
    let photo_provider: Arc<dyn PhotoProvider> =
        Arc::new(FileSystemPhotoProvider::new(vec![photos_dir.clone()]));

    // ----- 6) Scheduler + services(镜像 bootstrap::assemble) -----
    let registry = Arc::new(ExecutorRegistry::new());
    let scheduler = Arc::new(SqliteTaskScheduler::new((*db).clone(), registry.clone()));
    let scan = Arc::new(ScanService::new(
        db.clone(),
        scheduler.clone() as Arc<dyn TaskScheduler>,
        photo_provider.clone(),
        false,
    ));
    let index = Arc::new(IndexService::new(
        db.clone(),
        pipeline.clone(),
        face_index.clone(),
        None, // body_pipeline
        None, // body_index
        None, // object_embedder
        None, // object_index
        photo_provider.clone(),
        None, // patch_extractor
        None, // patch_index
    ));
    let search = Arc::new(
        SearchService::new(
            db.clone(),
            config.clone(),
            pipeline.clone(),
            face_index.clone(),
            None, // body_index
            None, // object_embedder
            None, // object_index
            photo_provider.clone(),
            None, // patch_index
            None, // patch_search
            None, // category_search
            None, // superpoint
            None, // lightglue
        )
        .with_prototype_service(Arc::new(PrototypeService::new(db.clone()))),
    );
    let person = Arc::new(
        PersonService::new(db.clone(), face_index.clone(), ClusterPolicy::default())
            .with_prototype_service(Arc::new(PrototypeService::new(db.clone()))),
    );
    register_executors(&registry, db.clone(), index.clone(), person.clone());

    // ----- 7) Scan -----
    eprintln!("=== ScanService::scan_folder({}) ===", photos_dir.display());
    let summary = scan.scan_folder(&photos_dir).await.unwrap();
    eprintln!(
        "scan: candidate={} inserted={} skipped={} filtered={} filtered_bpp={} queued_face_tasks={}",
        summary.candidate_count,
        summary.inserted,
        summary.skipped,
        summary.filtered,
        summary.filtered_bpp,
        summary.queued_face_tasks
    );
    assert!(summary.inserted >= 8, "至少 8 张图入库");

    // ----- 8) 等 IndexImage 任务排干 -----
    eprintln!("=== 等 IndexImage 任务 ===");
    let start = Instant::now();
    loop {
        let s = scheduler.status().await.unwrap();
        let busy = s.pending + s.running;
        eprintln!(
            "  t={:>6?}  pending={} running={} completed={} failed={} cancelled={}",
            start.elapsed(),
            s.pending,
            s.running,
            s.completed,
            s.failed,
            s.cancelled
        );
        if busy == 0 && s.failed == 0 {
            break;
        }
        if s.failed > 0 {
            eprintln!("有任务失败");
            break;
        }
        if start.elapsed() > Duration::from_secs(300) {
            panic!("索引超时(300s)");
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
    }
    let final_stats = scheduler.status().await.unwrap();
    eprintln!(
        "终态: completed={} failed={} cancelled={}",
        final_stats.completed, final_stats.failed, final_stats.cancelled
    );
    assert_eq!(final_stats.failed, 0, "索引任务不应失败");

    // ----- 9) DB / HNSW 统计 -----
    let (n_img, n_face): (i64, i64) = db
        .transaction(|tx| Ok((tx.images().count()?, tx.faces().count()?)))
        .unwrap();
    let hnsw_len = face_index.len();
    eprintln!("DB: images={n_img} faces={n_face}");
    eprintln!("HNSW: len={hnsw_len}");
    assert!(n_img > 0, "DB 应有 image");
    assert!(n_face > 0, "至少检测到一张人脸");
    assert!(hnsw_len > 0, "HNSW 应有向量");

    // ----- 10) Search -----
    // 选 jaqor 系列(face_debug 验证过 pipeline), 用第 1 张检索 top-5。
    let query_name = "pexels-jaqor-33601831.jpg";
    let query_bytes = fs::read(photos_dir.join(query_name)).expect("read query");
    eprintln!("=== SearchService::search_by_face_image({query_name}) ===");
    let results = search.search_by_face_image(&query_bytes, 5).await.unwrap();
    eprintln!("返回 {} 个结果:", results.len());
    for r in &results {
        eprintln!(
            "  rank={} image_id={} target_id={} score={:.4}",
            r.rank, r.image_id, r.target_id, r.score
        );
    }
    assert!(!results.is_empty(), "search 应返回结果");

    // ----- 11) 总结 -----
    eprintln!("\n=== E2E SMOKE PASSED ===");
    eprintln!(
        "  images={} faces={} hnsw={} search_results={}",
        n_img,
        n_face,
        hnsw_len,
        results.len()
    );
}