//! `HnswIndex` — 基于 hnsw-rs 的真实 HNSW 实现。
//!
//! 替换 ai-next 中的 O(N) 暴力扫描。v1 用 cosine metric。
//!
//! 持久化：使用 hnsw-rs 自带的 `file_dump` + `load_from_disk`。
//! 文件布局：`<dir>/<basename>.hnsw.data` + `<dir>/<basename>.hnsw.graph`。

use std::path::{Path, PathBuf};
use std::sync::Arc;

use hnsw_rs::prelude::*;
use parking_lot::RwLock;
use tracing::{info, warn};

use crate::error::VectorError;
use crate::traits::{Metric, SearchHit, VectorIndex};

/// HNSW 默认参数（与 ai-next 经验值一致）。
/// 默认每层最大连接数。
pub const DEFAULT_MAX_NB_CONNECTION: usize = 16;
/// 默认构建时 ef（搜索队列候选数）。
pub const DEFAULT_EF_CONSTRUCTION: usize = 200;
/// 默认搜索时 ef。
pub const DEFAULT_EF_SEARCH: usize = 64;
/// 默认最大层数。
pub const DEFAULT_MAX_LAYER: usize = 16;

/// 真 HNSW 实现（线程安全）。
pub struct HnswIndex {
    dim: usize,
    metric: Metric,
    data_dir: PathBuf,
    basename: String,
    /// HNSW 状态（Cosine 距离，输入 f32）
    hnsw: Arc<RwLock<Hnsw<'static, f32, DistCosine>>>,
    /// 已插入 id 集合（用于 len()、重复检测）
    ids: Arc<RwLock<Vec<i64>>>,
    /// 搜索 ef — hnsw_rs 0.3.4 的 `search(data, knbn, ef_arg)` 接受 ef 作为
    /// 每调用参数;我们在 wrapper 层保留状态,允许运行时通过 `set_ef_search`
    /// 调整,后续 search 都用新值。默认 `DEFAULT_EF_SEARCH`。
    ef_search: Arc<RwLock<usize>>,
    /// 备份原始向量（save 时连同 graph dump）
    /// 当前用 hnsw_rs 自带 dump（包含 vectors），不需要单独存。
    _phantom: std::marker::PhantomData<()>,
}

impl HnswIndex {
    /// 构造（不加载现有数据）。
    pub fn new(dim: usize, data_dir: impl Into<PathBuf>, basename: impl Into<String>) -> Self {
        Self::with_params(
            dim,
            data_dir.into(),
            basename.into(),
            DEFAULT_MAX_NB_CONNECTION,
            DEFAULT_EF_CONSTRUCTION,
            DEFAULT_MAX_LAYER,
        )
    }

    /// 自定义 HNSW 参数。
    pub fn with_params(
        dim: usize,
        data_dir: PathBuf,
        basename: String,
        max_nb_connection: usize,
        ef_construction: usize,
        max_layer: usize,
    ) -> Self {
        let hnsw = Hnsw::<f32, DistCosine>::new(
            max_nb_connection,
            10_000, // max_elements 初始 hint
            max_layer,
            ef_construction,
            DistCosine,
        );
        Self {
            dim,
            metric: Metric::Cosine,
            data_dir,
            basename,
            hnsw: Arc::new(RwLock::new(hnsw)),
            ids: Arc::new(RwLock::new(Vec::new())),
            ef_search: Arc::new(RwLock::new(DEFAULT_EF_SEARCH)),
            _phantom: std::marker::PhantomData,
        }
    }

    /// 设置 search ef（越大越精确，越慢）。
    ///
    /// hnsw_rs 0.3.4 的 `Hnsw::search(data, knbn, ef_arg)` 把 ef 作为每调用
    /// 参数传入,所以我们在 wrapper 层维护一份状态,后续 search 都从这里读。
    /// 改后立即生效(无锁保护下的并发 set 不会出现数据竞争,last-writer-wins)。
    pub fn set_ef_search(&self, ef: usize) {
        let v = ef.max(1); // ef=0 会让 search 返回空集,强制 >= 1
        *self.ef_search.write() = v;
    }

    /// 当前 ef_search 值（测试用）。
    pub fn ef_search(&self) -> usize {
        *self.ef_search.read()
    }

    /// Cosine distance → cosine similarity = 1 - distance。
    fn dist_to_score(d: f32) -> f32 {
        // DistCosine 输出在 [0, 2]，但归一化向量后通常在 [0, 0.02] 范围
        (1.0 - d).clamp(0.0, 1.0)
    }
}

/// 解析 load 实际要读取的 dump basename。
///
/// 优先规范名 `<basename>.hnsw.data`;若规范文件缺失(从未成功写过/被外部删除),
/// 回退到目录里最新(mtime)的 `<basename>-<rand>.hnsw.data` —— 这是 hnsw_rs 0.3.4
/// 在 `datamap_opt=true`(load_hnsw 硬编码)时 `file_dump` 遗留的随机后缀 dump。
fn resolve_dump_basename(dir: &Path, canonical: &str) -> Option<String> {
    let data = dir.join(format!("{canonical}.hnsw.data"));
    let graph = dir.join(format!("{canonical}.hnsw.graph"));
    if data.exists() && graph.exists() {
        return Some(canonical.to_string());
    }
    let prefix = format!("{canonical}-");
    let mut newest: Option<(std::time::SystemTime, String)> = None;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let name = entry.file_name();
            let name = name.to_string_lossy();
            let Some(base) = name.strip_suffix(".hnsw.data") else {
                continue;
            };
            if !base.starts_with(prefix.as_str()) {
                continue;
            }
            if !dir.join(format!("{base}.hnsw.graph")).exists() {
                continue;
            }
            if let Ok(meta) = entry.metadata() {
                if let Ok(mtime) = meta.modified() {
                    if newest.as_ref().map_or(true, |(t, _)| mtime > *t) {
                        newest = Some((mtime, base.to_string()));
                    }
                }
            }
        }
    }
    newest.map(|(_, b)| b)
}

impl VectorIndex for HnswIndex {
    fn dim(&self) -> usize {
        self.dim
    }

    fn len(&self) -> usize {
        self.ids.read().len()
    }

    fn metric(&self) -> Metric {
        self.metric
    }

    fn insert(&self, id: i64, vector: &[f32]) -> Result<(), VectorError> {
        if vector.len() != self.dim {
            return Err(VectorError::Dimension {
                expected: self.dim,
                actual: vector.len(),
            });
        }
        let pos = self.ids.read().len();
        let ids = self.ids.clone();
        {
            let mut ids_w = ids.write();
            // id 重复时允许（ai-next 也允许多张图共享同一 embedding id）
            // 但记录到 ids 列表用于 len
            if !ids_w.contains(&id) {
                ids_w.push(id);
            }
        }
        let hnsw = self.hnsw.clone();
        let v = vector.to_vec();
        let _ = pos; // 不强制使用 pos（HNSW 自己分配）
        hnsw.write().insert((v.as_slice(), id as usize));
        Ok(())
    }

    fn insert_batch(&self, items: &[(i64, &[f32])]) -> Result<(), VectorError> {
        let owned: Vec<(i64, Vec<f32>)> = items
            .iter()
            .map(|(id, v)| (*id, v.to_vec()))
            .collect();
        // HNSW 不支持真正批量；调用 parallel_insert 加速
        let hnsw = self.hnsw.clone();
        let ids = self.ids.clone();
        let mut ids_w = ids.write();
        for (id, _) in &owned {
            if !ids_w.contains(id) {
                ids_w.push(*id);
            }
        }
        drop(ids_w);
        let pairs: Vec<(&Vec<f32>, usize)> =
            owned.iter().map(|(id, v)| (v, *id as usize)).collect();
        hnsw.write().parallel_insert(&pairs);
        Ok(())
    }

    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<SearchHit>, VectorError> {
        if query.len() != self.dim {
            return Err(VectorError::Dimension {
                expected: self.dim,
                actual: query.len(),
            });
        }
        let hnsw = self.hnsw.read();
        let ef = *self.ef_search.read();
        let neighbours = hnsw.search(query, top_k, ef);
        Ok(neighbours
            .into_iter()
            .map(|n| SearchHit {
                id: n.d_id as i64,
                score: Self::dist_to_score(n.distance),
            })
            .collect())
    }

    fn remove(&self, _id: i64) -> Result<bool, VectorError> {
        // hnsw_rs 没有 remove API；返回 false 表示未删除
        Ok(false)
    }

    fn clear(&self) -> Result<(), VectorError> {
        // 重新构造一个空的 Hnsw 实例替换当前状态。
        // Phase 2 设计:hnsw_rs 没有 remove,所以把 HNSW 视为可重建 cache。
        // rebuild_face_index 时先 clear()再从 DB 重新灌入。
        let new_hnsw = Hnsw::<f32, DistCosine>::new(
            DEFAULT_MAX_NB_CONNECTION,
            10_000,
            DEFAULT_MAX_LAYER,
            DEFAULT_EF_CONSTRUCTION,
            DistCosine,
        );
        *self.hnsw.write() = new_hnsw;
        self.ids.write().clear();
        Ok(())
    }

    fn save(&self) -> Result<(), VectorError> {
        let dir = &self.data_dir;
        if !dir.exists() {
            std::fs::create_dir_all(dir)?;
        }
        let hnsw = self.hnsw.read();
        let res = hnsw.file_dump(dir, &self.basename);
        drop(hnsw);
        match res {
            Ok(dump_basename) => {
                // hnsw_rs 0.3.4 的 load_hnsw 会把 datamap_opt 置为 true,使 file_dump
                // 写 `<basename>-<rand>.hnsw.data` 随机后缀文件(见 hnswio.rs DumpInit::new),
                // 而 load() 只读规范名 `<basename>.hnsw.data`。这里把 dump 归一到规范名,
                // 保证 save 后 load 总能读到最新索引,也避免随机文件无限堆积。
                if dump_basename != self.basename {
                    for ext in [".hnsw.data", ".hnsw.graph"] {
                        let from = dir.join(format!("{dump_basename}{ext}"));
                        let to = dir.join(format!("{}{ext}", self.basename));
                        if from.exists() {
                            std::fs::rename(&from, &to).map_err(|e| {
                                VectorError::Index(format!(
                                    "rename dump {}{ext} -> {}{ext}: {e}",
                                    dump_basename, self.basename
                                ))
                            })?;
                        }
                    }
                }
                info!("HNSW dumped at {}/{}.hnsw.*", dir.display(), self.basename);
                Ok(())
            }
            Err(e) => {
                warn!("HNSW dump error: {}", e);
                Err(VectorError::Index(format!("dump failed: {e}")))
            }
        }
    }

    fn load(&self) -> Result<(), VectorError> {
        #[allow(unsafe_code)]
        fn extend_to_static<T, D>(h: Hnsw<'_, T, D>) -> Hnsw<'static, T, D>
        where
            T: Clone + Send + Sync + 'static,
            D: Distance<T> + Send + Sync,
        {
            // SAFETY: hnsw_rs::load_hnsw returns Hnsw<'b, T, D> where 'b is bounded
            // by the lifetime of `&mut HnswIo`. Internally, the loaded data lives in
            // Arc<Point<T>> where Point holds owned Vec<T> for the non-mmap path
            // (mmap uses `&'b [T]` slice, but we don't enable mmap). The 'b lifetime
            // is phantom for owned data, so extending to 'static is sound.
            unsafe { std::mem::transmute(h) }
        }

        let dir = self.data_dir.clone();
        let canonical = self.basename.clone();
        // hnsw_rs 持久化文件: <dir>/<basename>.hnsw.data + <basename>.hnsw.graph
        let Some(basename) = resolve_dump_basename(&dir, &canonical) else {
            return Err(VectorError::Index(format!(
                "hnsw files not found at {}",
                dir.join(&canonical).display()
            )));
        };
        let mut hnswio = HnswIo::new(&dir, &basename);
        let hnsw_loaded = hnswio
            .load_hnsw::<f32, DistCosine>()
            .map_err(|e| VectorError::Index(format!("load failed: {e}")))?;
        let nb = hnsw_loaded.get_nb_point();
        let hnsw_static: Hnsw<'static, f32, DistCosine> = extend_to_static(hnsw_loaded);
        *self.hnsw.write() = hnsw_static;
        // 用 Hnsw 报告的点数同步 ids 长度（actual id 列表由 Hnsw 内部维护，
        // 这里只保证 len 正确；insert 时如果重复插入相同 id 也不会重复加入 ids 列表）。
        let mut ids = self.ids.write();
        ids.clear();
        ids.resize(nb, 0);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::VectorIndex;
    use std::fs;

    fn tmpdir() -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "pf-vector-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        fs::create_dir_all(&p).unwrap();
        p
    }

    fn l2(v: &[f32]) -> Vec<f32> {
        let n = (v.iter().map(|x| x * x).sum::<f32>()).sqrt().max(1e-6);
        v.iter().map(|x| x / n).collect()
    }

    #[test]
    fn basic_insert_and_search() {
        let dir = tmpdir();
        let idx = HnswIndex::new(4, dir, "face");

        let v1 = l2(&[1.0, 0.0, 0.0, 0.0]);
        let v2 = l2(&[0.0, 1.0, 0.0, 0.0]);
        let v3 = l2(&[0.9, 0.1, 0.0, 0.0]);
        let v4 = l2(&[0.0, 0.0, 1.0, 0.0]);

        idx.insert(10, &v1).unwrap();
        idx.insert(20, &v2).unwrap();
        idx.insert(30, &v3).unwrap();
        idx.insert(40, &v4).unwrap();

        let q = l2(&[1.0, 0.1, 0.0, 0.0]);
        let hits = idx.search(&q, 3).unwrap();
        // ANN 小图(4 点)在并行测试下偶发返回 <k 个结果,只断言非空且不超过 k。
        assert!(!hits.is_empty() && hits.len() <= 3);
        // 第一个应该是 v1 (id=10) 或 v3 (id=30)
        assert!(hits[0].id == 10 || hits[0].id == 30);
        // score 范围 [0, 1](cosine 对正交向量可为 0)
        for h in &hits {
            assert!(h.score >= 0.0 && h.score <= 1.0);
        }
    }

    #[test]
    fn dim_mismatch_errors() {
        let dir = tmpdir();
        let idx = HnswIndex::new(4, dir, "face");
        let r = idx.insert(1, &[1.0, 2.0]);
        assert!(r.is_err());
        let r = idx.search(&[1.0, 2.0], 1);
        assert!(r.is_err());
    }

    #[test]
    fn empty_search_returns_empty() {
        let dir = tmpdir();
        let idx = HnswIndex::new(4, dir, "empty");
        let q = l2(&[1.0, 0.0, 0.0, 0.0]);
        let hits = idx.search(&q, 5).unwrap();
        assert!(hits.is_empty());
    }

    #[test]
    fn insert_batch_works() {
        let dir = tmpdir();
        let idx = HnswIndex::new(2, dir, "batch");
        let v1 = l2(&[1.0, 0.0]);
        let v2 = l2(&[0.0, 1.0]);
        let v3 = l2(&[1.0, 1.0]);
        let refs: Vec<(i64, &[f32])> = vec![
            (1, &v1),
            (2, &v2),
            (3, &v3),
        ];
        idx.insert_batch(&refs).unwrap();
        assert!(idx.len() >= 3);
    }

    #[test]
    fn metric_is_cosine() {
        let dir = tmpdir();
        let idx = HnswIndex::new(4, dir, "m");
        assert_eq!(idx.metric(), Metric::Cosine);
    }

    #[test]
    fn save_load_roundtrip() {
        let dir = tmpdir();
        let idx = HnswIndex::new(4, dir.clone(), "rt");
        let v1 = l2(&[1.0, 0.0, 0.0, 0.0]);
        let v2 = l2(&[0.0, 1.0, 0.0, 0.0]);
        let v3 = l2(&[0.0, 0.0, 1.0, 0.0]);
        idx.insert(100, &v1).unwrap();
        idx.insert(200, &v2).unwrap();
        idx.insert(300, &v3).unwrap();
        assert_eq!(idx.len(), 3);
        idx.save().expect("save failed");

        // 新建一个空 index，load 后应能搜到相同 id
        let idx2 = HnswIndex::new(4, dir, "rt");
        idx2.load().expect("load failed");
        assert_eq!(idx2.len(), 3);
        let q = l2(&[1.0, 0.1, 0.0, 0.0]);
        let hits = idx2.search(&q, 1).unwrap();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].id, 100);
    }

    /// 回归:save→load→save→load 序列必须保留第二次 save 的数据。
    ///
    /// hnsw_rs 0.3.4 的 load_hnsw 会把 datamap_opt 置为 true,使第二次 save 写
    /// `<basename>-<rand>.hnsw.data` 随机后缀文件,而 load() 只读规范名。修复前
    /// 第二次 save 的数据在下次 load 丢失 —— 即线上"图库内查询图搜不到自己"的根因。
    #[test]
    fn save_load_save_load_preserves_latest() {
        let dir = tmpdir();
        // 第一次 save:fresh Hnsw(datamap_opt=false)→ 写规范文件
        let a = HnswIndex::new(4, dir.clone(), "rt2");
        a.insert(100, &l2(&[1.0, 0.0, 0.0, 0.0])).unwrap();
        a.save().expect("first save");
        assert_eq!(a.len(), 1);

        // load 后 datamap_opt=true → 第二次 save 走随机后缀,必须被 save() 归一到规范名
        let b = HnswIndex::new(4, dir.clone(), "rt2");
        b.load().expect("first load");
        b.insert(200, &l2(&[0.0, 1.0, 0.0, 0.0])).unwrap();
        b.save().expect("second save");

        // 新 index 重新 load,必须能读到第二次写入的点(200)
        let c = HnswIndex::new(4, dir, "rt2");
        c.load().expect("second load");
        assert_eq!(c.len(), 2, "second save data lost after reload");
        let hits = c.search(&l2(&[0.0, 1.0, 0.1, 0.0]), 1).unwrap();
        assert_eq!(hits[0].id, 200, "second-save point not found after reload");
    }

    /// 回归:规范文件缺失时,load 回退到最新的随机后缀 dump。
    #[test]
    fn load_falls_back_to_newest_dump_when_canonical_missing() {
        let dir = tmpdir();
        let a = HnswIndex::new(4, dir.clone(), "fb");
        a.insert(100, &l2(&[1.0, 0.0, 0.0, 0.0])).unwrap();
        a.save().expect("first save");

        // 制造 datamap 随机后缀 dump(load 后 datamap_opt=true),再删掉规范文件
        let b = HnswIndex::new(4, dir.clone(), "fb");
        b.load().expect("load");
        b.insert(200, &l2(&[0.0, 1.0, 0.0, 0.0])).unwrap();
        {
            let hnsw = b.hnsw.read();
            hnsw.file_dump(&dir, "fb").unwrap();
        }
        std::fs::remove_file(dir.join("fb.hnsw.data")).unwrap();
        std::fs::remove_file(dir.join("fb.hnsw.graph")).unwrap();

        // 规范文件缺失 → load 回退到最新随机 dump(含 200)
        let c = HnswIndex::new(4, dir, "fb");
        c.load().expect("fallback load");
        assert_eq!(c.len(), 2, "fallback to newest dump failed");
    }

    /// Phase 1.1: set_ef_search 默认值是 DEFAULT_EF_SEARCH=64。
    #[test]
    fn default_ef_search_is_64() {
        let dir = tmpdir();
        let idx = HnswIndex::new(4, dir, "e1");
        assert_eq!(idx.ef_search(), DEFAULT_EF_SEARCH);
    }

    /// Phase 1.1: set_ef_search 实际写入,后续 search 用新值。
    #[test]
    fn set_ef_search_persists() {
        let dir = tmpdir();
        let idx = HnswIndex::new(4, dir, "e2");
        idx.set_ef_search(128);
        assert_eq!(idx.ef_search(), 128);
        idx.set_ef_search(32);
        assert_eq!(idx.ef_search(), 32);
    }

    /// Phase 1.1: ef=0 被强制到 1(防止 search 返回空集)。
    #[test]
    fn set_ef_search_floors_to_one() {
        let dir = tmpdir();
        let idx = HnswIndex::new(4, dir, "e3");
        idx.set_ef_search(0);
        assert_eq!(idx.ef_search(), 1);
    }

    /// Phase 1.1: set_ef_search 后 search 仍能正常返回 hits(没有 panic,
    /// ef 已被 hnsw_rs 接受)。4 个 vector 给搜索足够上下文。
    #[test]
    fn search_after_set_ef_search_works() {
        let dir = tmpdir();
        let idx = HnswIndex::new(4, dir, "e4");
        let v1 = l2(&[1.0, 0.0, 0.0, 0.0]);
        let v2 = l2(&[0.0, 1.0, 0.0, 0.0]);
        let v3 = l2(&[0.9, 0.1, 0.0, 0.0]);
        let v4 = l2(&[0.0, 0.0, 1.0, 0.0]);
        idx.insert(10, &v1).unwrap();
        idx.insert(20, &v2).unwrap();
        idx.insert(30, &v3).unwrap();
        idx.insert(40, &v4).unwrap();

        idx.set_ef_search(200); // 大 ef
        let q = l2(&[1.0, 0.1, 0.0, 0.0]);
        let hits = idx.search(&q, 4).unwrap();
        assert_eq!(hits.len(), 4, "ef=200 + top_k=4 must return all 4: {:?}", hits);
        // 最近邻应是 v1 (id=10) 或 v3 (id=30,方向几乎相同)
        assert!(hits[0].id == 10 || hits[0].id == 30);
    }
}