/*
 * iOS-only C shim providing the pf_ios_* C-ABI symbols the Rust bridge
 * (platform/src/ios.rs) calls. iOS staticlibs do not allow undefined
 * symbols at link time, so the crate itself must not link with unresolved
 * references.
 *
 * The real implementation lives in PhotoKitBridge.swift (part of the Xcode
 * app target). The Swift object defines these symbols directly and links
 * AFTER this staticlib in the Frameworks build phase, so without a weak
 * attribute the static archive's strong definition wins and every call
 * returns NULL — which is exactly what was happening before this fix.
 *
 * Marking every pf_ios_* symbol `weak` lets the Rust staticlib link still
 * succeed (a weak symbol satisfies the resolver), while the strong
 * Swift @_cdecl definition overrides it at the final app link. If the Swift
 * counterpart ever disappears (e.g. someone strips PhotoKitBridge.swift
 * from the Xcode target), the weak NULL stub is what runs — same behavior
 * as before, just safer.
 *
 * Function results are freed by Rust via libc::free, so NULL here just means
 * "nothing available" (the bridge treats NULL as an error / empty result).
 *
 * Symbol signatures must match the Rust `unsafe extern "C"` block in
 * pf_platform/src/ios.rs exactly. If you add a new extern "C" function in
 * Rust, add a matching weak stub here.
 */

#include <stdint.h>

#define PF_IOS_WEAK __attribute__((weak))

/* photo permission */
PF_IOS_WEAK int32_t pf_ios_photo_permission_status(void) { return 0; }

PF_IOS_WEAK int32_t pf_ios_request_photo_permission(void) { return 0; }

/* list + lookup */
PF_IOS_WEAK char *pf_ios_list_photo_items(void) { return 0; }

PF_IOS_WEAK char *pf_ios_photo_item_by_uri(const char *uri) { return 0; }

/* export to disk */
PF_IOS_WEAK char *pf_ios_export_to_temp_file(const char *uri) { return 0; }

/* export thumb (uri + max_size match Rust signature) */
PF_IOS_WEAK char *pf_ios_export_thumbnail_to_temp_file(const char *uri, uint32_t max_size) { return 0; }

/* read raw bytes (base64 in current implementation) */
PF_IOS_WEAK char *pf_ios_read_bytes(const char *uri) { return 0; }

/* background task scheduling */
PF_IOS_WEAK int32_t pf_ios_schedule_bg_task(const char *id, int64_t wait_secs) { return 0; }
