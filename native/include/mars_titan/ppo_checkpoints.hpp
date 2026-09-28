#ifndef MARS_TITAN_PPO_CHECKPOINTS_HPP
#define MARS_TITAN_PPO_CHECKPOINTS_HPP

#include <nlohmann/json.hpp>

#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <string_view>

namespace mars_titan::learning {

inline constexpr std::size_t maximum_ppo_archive_bytes = std::size_t{128} * 1024 * 1024;
inline constexpr std::size_t maximum_ppo_metadata_bytes = std::size_t{32} * 1024 * 1024;
inline constexpr std::size_t maximum_ppo_bundle_bytes =
    2 * maximum_ppo_archive_bytes + maximum_ppo_metadata_bytes;
inline constexpr std::size_t maximum_ppo_retained_payload_bytes = 3 * maximum_ppo_bundle_bytes;

struct PpoCheckpointBundle {
    nlohmann::json metadata;
    std::string policy_archive;
    std::string rollout_archive;
    nlohmann::json receipt;
};

// Conserva dos estados recientes y el mejor seleccionado. Cada estado admite
// 288 MiB de contenido y la retención suma como máximo 864 MiB. La publicación
// admite un cuarto estado y un temporal de hasta 128 MiB al recuperar un corte.
class PpoCheckpointStore {
  public:
    PpoCheckpointStore(const std::filesystem::path& output, const nlohmann::json& identity,
                       bool resume = false);
    PpoCheckpointStore(const PpoCheckpointStore&) = delete;
    PpoCheckpointStore& operator=(const PpoCheckpointStore&) = delete;
    PpoCheckpointStore(PpoCheckpointStore&&) = delete;
    PpoCheckpointStore& operator=(PpoCheckpointStore&&) = delete;
    ~PpoCheckpointStore();

    [[nodiscard]] nlohmann::json save(const nlohmann::json& metadata,
                                      std::string_view policy_archive,
                                      std::string_view rollout_archive, bool select_best = false);
    [[nodiscard]] PpoCheckpointBundle load_latest();
    [[nodiscard]] PpoCheckpointBundle load_best();

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mars_titan::learning

#endif
