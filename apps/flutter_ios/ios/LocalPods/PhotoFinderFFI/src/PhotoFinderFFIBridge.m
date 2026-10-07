// PhotoFinderFFI Bridge - Forces linker to include Rust symbols
// This file creates compile-time references to Rust functions

#import <Foundation/Foundation.h>

// Declare Rust FFI functions
extern int _pf_init(void);
extern int _pf_set_models_dir(const char* path);
extern int _pf_add_photo(const uint8_t* bytes, int len, int64_t photo_id);
extern int _pf_start_scan(void);
extern int _pf_stop_scan(void);
extern int _pf_clear_database(void);
extern char* _pf_get_statistics(void);
extern char* _pf_get_status(void);
extern char* _pf_search_face(const uint8_t* bytes, int len, int top_k);
extern char* _pf_search_object(const uint8_t* bytes, int len, int top_k);
extern char* _pf_get_version(void);
extern void _pf_free_string(char* ptr);

// Force linker to include ALL Rust object files
// by creating a function that references them
__attribute__((used))
static void photofinder_force_link_anchor(void) {
    // These calls prevent dead stripping
    // We use the results to prevent the compiler from optimizing away
    int result __attribute__((unused));

    result = _pf_init();
    result = _pf_set_models_dir("");
    result = _pf_add_photo(NULL, 0, 0);
    result = _pf_start_scan();
    result = _pf_stop_scan();
    result = _pf_clear_database();

    char* str __attribute__((unused));
    str = _pf_get_statistics();
    str = _pf_get_status();
    str = _pf_search_face(NULL, 0, 0);
    str = _pf_search_object(NULL, 0, 0);
    str = _pf_get_version();
    _pf_free_string(NULL);

    // Additional references to ensure symbols are not stripped
    (void)&_pf_init;
    (void)&_pf_set_models_dir;
    (void)&_pf_add_photo;
    (void)&_pf_start_scan;
    (void)&_pf_stop_scan;
    (void)&_pf_clear_database;
    (void)&_pf_get_statistics;
    (void)&_pf_get_status;
    (void)&_pf_search_face;
    (void)&_pf_search_object;
    (void)&_pf_get_version;
    (void)&_pf_free_string;
}

// Plugin entry point - called by Flutter
// This is a dummy export that ensures the linker includes this object file
__attribute__((visibility("default")))
__attribute__((used))
int photofinderFFIBridgeRegister(void) {
    photofinder_force_link_anchor();
    return 0;
}
