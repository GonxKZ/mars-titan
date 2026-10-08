#ifndef MARS_TITAN_KLPO_EPISODES_HPP
#define MARS_TITAN_KLPO_EPISODES_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace mars_titan::learning {
inline constexpr std::size_t maximum_klpo_episodes = 128;
inline constexpr std::size_t maximum_klpo_steps = 256;
inline constexpr std::size_t klpo_record_action_count = 6;
inline constexpr std::size_t maximum_klpo_record_bytes = std::size_t{128} * 1024 * 1024;
inline constexpr std::string_view klpo_episode_contract = "klpo_complete_episodes_v1";

enum class KlpoEpisodeStatus : uint8_t { open, horizon, ruin, missing_valuation };

struct KlpoEpisodeSpec {
    std::string id;
    std::string source_sha256;
    std::string context_sha256;
    std::string partition = "train";
    std::vector<int64_t> close_times;
    std::size_t forced_prefix = 0;
};

struct KlpoEpisodeStep {
    std::size_t cursor = 0;
    int64_t decision_at = 0;
    int64_t outcome_at = 0;
    std::vector<float> observation;
    std::array<float, klpo_record_action_count> behavior{};
    uint8_t action = 1;
    bool sampled = false;
    double reward = 0;
    // Procede del motor. No combina valoración con elegibilidad de aprendizaje.
    bool valuation_valid = true;
    bool terminated = false;
    bool truncated = false;
};

struct KlpoEpisodeRecord {
    KlpoEpisodeSpec spec;
    std::vector<KlpoEpisodeStep> steps;
};

struct KlpoEpisodeBatch {
    std::string fold;
    std::string reference_sha256;
    double beta = 1;
    double gamma = 1;
    std::size_t observation_width = 0;
    std::size_t max_bytes = maximum_klpo_record_bytes;
    std::vector<KlpoEpisodeRecord> episodes;
};

// Valida el presupuesto de toda la oleada prevista, incluso si está incompleta.
void validate_klpo_batch(const KlpoEpisodeBatch& batch, bool require_complete = false);
[[nodiscard]] KlpoEpisodeStatus klpo_episode_status(const KlpoEpisodeRecord& episode);
[[nodiscard]] std::vector<double> klpo_terminal_returns(const KlpoEpisodeBatch& batch);
[[nodiscard]] std::string serialize_klpo_batch(const KlpoEpisodeBatch& batch);
[[nodiscard]] KlpoEpisodeBatch deserialize_klpo_batch(
    std::string_view bytes, std::size_t max_bytes = maximum_klpo_record_bytes);
}
#endif
