Pod::Spec.new do |s|
  s.name         = 'PhotoFinderFFI'
  s.version      = '0.1.0'
  s.summary      = 'PhotoFinder Rust FFI Library'
  s.description  = 'Rust static library providing photo scanning functionality'
  s.homepage     = 'https://github.com/photofinder/photofinder-next-2'
  s.license      = { :type => 'MIT' }
  s.author       = { 'PhotoFinder' => 'dev@photofinder.app' }
  s.platform     = :ios, '15.0'
  s.source       = { :path => '.' }
  s.source_files = 'src/**/*.{h,m,c}'
  s.public_header_files = 'src/**/*.h'
  s.vendored_libraries = 'lib/*.a'
  s.libraries    = 'sqlite3', 'z'
  s.static_framework = true

  # Library search path for vendored Rust library
  s.pod_target_xcconfig = {
    'LIBRARY_SEARCH_PATHS' => '"$(PODS_TARGET_SRCROOT)/lib"',
  }
end
