//! Body embedding index using HNSW.
//!
//! Wraps `pf_vector::HnswIndex` for body embeddings (768D YouTu Re-ID).
//! Files stored at `<data_dir>/index/body.hnsw.data` + `.hnsw.graph`.

use std::path::PathBuf;
use std::sync::Arc;

use pf_vector::HnswIndex;
use tracing::info;

/// Body index using HNSW for 768D body embeddings.
pub struct BodyIndex {
    inner: Arc<HnswIndex>,
}

impl BodyIndex {
    /// Create a new BodyIndex.
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        let inner = HnswIndex::new(768, data_dir, "body");
        info!("BodyIndex created (768D, YouTu Re-ID)");
        Self { inner: Arc::new(inner) }
    }

    /// Get inner HnswIndex for insertion/search.
    pub fn inner(&self) -> Arc<HnswIndex> {
        self.inner.clone()
    }

    /// Get embedding dimension.
    pub fn dim(&self) -> usize {
        768
    }
}

impl From<Arc<HnswIndex>> for BodyIndex {
    fn from(inner: Arc<HnswIndex>) -> Self {
        Self { inner }
    }
}
