#ifndef MARS_TITAN_SIMULATION_FILES_HPP
#define MARS_TITAN_SIMULATION_FILES_HPP

#include "mars_titan/financial_session.hpp"

#include <nlohmann/json.hpp>

#include <cstddef>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>

namespace mars_titan::simulation {

inline constexpr std::size_t bytes_per_kibibyte = 1024;
inline constexpr std::size_t bytes_per_mebibyte = bytes_per_kibibyte * bytes_per_kibibyte;
inline constexpr std::size_t maximum_manifest_bytes = 4 * bytes_per_mebibyte;
inline constexpr std::size_t maximum_market_bytes = 256 * bytes_per_mebibyte;
inline constexpr std::size_t maximum_checkpoint_bytes = 16 * bytes_per_mebibyte;
inline constexpr std::size_t default_checkpoint_steps = 64;

struct RunOptions {
    std::filesystem::path output;
    Parameters parameters;
    ReferencePolicy policy = ReferencePolicy::cash;
    std::size_t checkpoint_steps = default_checkpoint_steps;
    std::optional<std::size_t> stop_after;
    bool resume = false;
    bool diagnostic = false;
};

struct ComparisonOptions {
    RunOptions run;
    std::size_t workers = 1;
};

void require_safe_path(const std::filesystem::path& path);
[[nodiscard]] std::string read_bounded_file(const std::filesystem::path& path, std::size_t maximum);
[[nodiscard]] std::string content_sha256(std::string_view bytes);
[[nodiscard]] nlohmann::json parse_bounded_json(std::string_view bytes);
[[nodiscard]] int64_t read_json_int64(const nlohmann::json& value);
void atomic_json_file(const std::filesystem::path& path, const nlohmann::json& value,
                      std::size_t maximum = maximum_manifest_bytes);
[[nodiscard]] std::shared_ptr<const MarketTape>
load_market_tape(const std::filesystem::path& directory);
[[nodiscard]] nlohmann::json snapshot_json(const SessionSnapshot& snapshot);
[[nodiscard]] SessionSnapshot read_snapshot(const nlohmann::json& value);
[[nodiscard]] nlohmann::json run_reference(std::shared_ptr<const MarketTape> tape,
                                           const RunOptions& options,
                                           const std::function<bool()>& stop_requested);
[[nodiscard]] nlohmann::json run_comparison(std::shared_ptr<const MarketTape> tape,
                                            const ComparisonOptions& options,
                                            const std::function<bool()>& stop_requested);

} // namespace mars_titan::simulation

#endif
