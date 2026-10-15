#ifndef MARS_TITAN_BUILD_IDENTITY_HPP
#define MARS_TITAN_BUILD_IDENTITY_HPP

namespace mars_titan::build {

// Identidad de la compilación nativa que se escribe en los recibos. Los valores se definen solo
// en build_identity.cpp. Así un cambio de fuentes, de configuración o de worktree recompila esa
// unidad y no todas las que enlazan mars_titan_financial, y ccache puede reutilizar las demás.
[[nodiscard]] const char* native_version() noexcept;
[[nodiscard]] const char* native_source_sha256() noexcept;
[[nodiscard]] const char* native_build_sha256() noexcept;
[[nodiscard]] const char* compiler_id() noexcept;
[[nodiscard]] const char* compiler_version() noexcept;
[[nodiscard]] const char* build_type() noexcept;

} // namespace mars_titan::build

#endif
