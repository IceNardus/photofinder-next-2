//! Tauri `State` 容器：持有所有 service 句柄。
//!
//! 通过 `pf_application::Bootstrapped` 构造。

use std::sync::Arc;

use parking_lot::Mutex;

use pf_application::Bootstrapped;
use pf_database::Database;
use pf_vector::VectorIndex;

/// Tauri 共享状态（用 `Arc` 包以便跨线程）。
///
/// 字段全部 `Arc<...>`，方便 `#[tauri::command]` 通过 `State<'_, Arc<AppState>>`
/// 拿到 `&AppState` 的子字段引用。
pub struct AppState {
    /// pf_application 装配结果
    pub bootstrapped: Bootstrapped,

    /// 扫描开关（用于 stop_scan / is_scanning）
    pub is_scanning: Arc<Mutex<bool>>,

    /// 上一次 scan 的累计统计
    pub last_scan_stats: Arc<Mutex<LastScanStats>>,
}

impl AppState {
    /// 从 `Bootstrapped` 构造。
    pub fn from(boot: Bootstrapped) -> Self {
        Self {
            bootstrapped: boot,
            is_scanning: Arc::new(Mutex::new(false)),
            last_scan_stats: Arc::new(Mutex::new(LastScanStats::default())),
        }
    }

    /// 数据库快捷访问。
    pub fn db(&self) -> &Arc<Database> {
        &self.bootstrapped.ctx.database
    }

    /// 人脸向量索引快捷访问。
    pub fn face_index(&self) -> &Arc<dyn VectorIndex> {
        &self.bootstrapped.ctx.face_index
    }
}

/// 最近一次扫描统计（用于前端实时显示进度）。
#[derive(Debug, Default, Clone)]
pub struct LastScanStats {
    /// 候选总数
    pub total: usize,
    /// 已扫描
    pub scanned: usize,
    /// 当前文件 basename
    pub current_file: String,
    /// 上次 scan path
    pub path: String,
}
