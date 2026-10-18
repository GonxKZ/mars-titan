#ifndef MARS_TITAN_CUDA_LIFETIME_OPTIONS_HPP
#define MARS_TITAN_CUDA_LIFETIME_OPTIONS_HPP

#include <charconv>
#include <memory>
#include <span>
#include <stdexcept>
#include <string_view>

namespace mars_titan::diagnostics {
inline constexpr int max_lifetime_repetitions = 300;

struct LifetimeOptions {
    std::string_view workload;
    int repetitions;
    bool release_workspaces;
};

inline LifetimeOptions lifetime_options(std::span<const std::string_view, 3> arguments) {
    std::string_view selected;
    // La opción devuelta referencia un literal estático, no el almacenamiento del llamante.
    for (const auto workload : {"empty", "gemm", "gru", "candidate"}) {
        if (arguments[0] == workload) {
            selected = workload;
        }
    }
    int count = 0;
    const auto number = arguments[1];
    if (number.empty()) {
        throw std::invalid_argument("Las repeticiones no pueden estar vacías");
    }
    const auto first = std::to_address(number.begin());
    const auto end = std::to_address(number.end());
    const auto parsed = std::from_chars(first, end, count);
    if (selected.empty() || parsed.ec != std::errc{} || parsed.ptr != end ||
        count < 1 || count > max_lifetime_repetitions ||
        (arguments[2] != "keep" && arguments[2] != "release-workspaces")) {
        throw std::invalid_argument(
            "Uso: carga empty|gemm|gru|candidate, repeticiones 1..300, keep|release-workspaces");
    }
    return {selected, count, arguments[2] == "release-workspaces"};
}
} // namespace mars_titan::diagnostics
#endif
