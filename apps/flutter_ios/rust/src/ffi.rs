//! FFI exports for Flutter iOS
//! All functions use C ABI and are exported with #[no_mangle]

use std::ffi::CString;
use std::os::raw::c_char;

fn null_str(s: &str) -> *mut c_char {
    CString::new(s).expect("CString::new failed").into_raw()
}

fn json_ok<T: serde::Serialize>(data: &T) -> *mut c_char {
    match serde_json::to_string(data) {
        Ok(s) => null_str(&s),
        Err(e) => null_str(&format!("{{\"error\":\"{}\"}}", e)),
    }
}

/// Initialize the Rust backend
/// Returns: 0 on success, -1 on failure
#[no_mangle]
pub extern "C" fn pf_init() -> i32 {
    // TODO: Call bootstrap
    0
}

/// Get version string
#[no_mangle]
pub extern "C" fn pf_get_version() -> *mut c_char {
    null_str("PhotoFinder 2.0")
}

/// Start scanning photos
/// Returns: task_id on success, -1 on not implemented
#[no_mangle]
pub extern "C" fn pf_scan_start() -> i32 {
    -1  // Not implemented yet
}

/// Get scan progress as JSON
#[no_mangle]
pub extern "C" fn pf_scan_progress() -> *mut c_char {
    json_ok(&serde_json::json!({
        "processed": 0,
        "total": 0,
        "current_file": ""
    }))
}

/// Stop scan
/// Returns: 0 on success, -1 on not implemented
#[no_mangle]
pub extern "C" fn pf_scan_stop() -> i32 {
    -1
}

/// Get database statistics
#[no_mangle]
pub extern "C" fn pf_get_statistics() -> *mut c_char {
    json_ok(&serde_json::json!({
        "image_count": 0,
        "face_count": 0
    }))
}

/// Clear database
#[no_mangle]
pub extern "C" fn pf_clear_database() -> i32 {
    0
}

/// Search faces by image bytes
#[no_mangle]
pub extern "C" fn pf_search_face(
    _image_bytes: *const u8,
    _len: usize,
    _top_k: i32,
) -> *mut c_char {
    json_ok(&serde_json::json!([]))
}

/// Get thumbnail as base64
#[no_mangle]
pub extern "C" fn pf_get_thumbnail(_image_id: i64, _max_size: i32) -> *mut c_char {
    null_str("")
}

/// Free string allocated by Rust
/// # Safety
/// ptr must be a pointer returned by a Rust FFI function, and must not be used after freeing
#[no_mangle]
pub unsafe extern "C" fn pf_free_string(ptr: *mut c_char) {
    if !ptr.is_null() {
        let _ = CString::from_raw(ptr);
    }
}
