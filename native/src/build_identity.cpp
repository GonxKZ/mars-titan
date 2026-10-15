#include "mars_titan/build_identity.hpp"

// CMake define estas macros solo para esta unidad (CMakeLists.txt y cmake/BuildIdentity.cmake).
namespace mars_titan::build {

const char* native_version() noexcept { return MARS_TITAN_NATIVE_VERSION; }
const char* native_source_sha256() noexcept { return MARS_TITAN_NATIVE_SOURCE_SHA256; }
const char* native_build_sha256() noexcept { return MARS_TITAN_NATIVE_BUILD_SHA256; }
const char* compiler_id() noexcept { return MARS_TITAN_NATIVE_COMPILER_ID; }
const char* compiler_version() noexcept { return MARS_TITAN_NATIVE_COMPILER_VERSION; }
const char* build_type() noexcept { return MARS_TITAN_NATIVE_BUILD_TYPE; }

} // namespace mars_titan::build
