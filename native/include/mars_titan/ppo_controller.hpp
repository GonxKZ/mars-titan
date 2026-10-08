#ifndef MARS_TITAN_PPO_CONTROLLER_HPP
#define MARS_TITAN_PPO_CONTROLLER_HPP

#include <cstdint>
#include <optional>
#include <string_view>

namespace mars_titan::learning {
enum class PpoObjectiveKind : uint8_t { legacy_clip, clip_full_kl, kl_penalty_adaptive, clip_kl_epoch_stop };

inline constexpr double ppo_default_target_kl = 0.01;
inline constexpr double ppo_default_beta_min = 1e-6;
inline constexpr double ppo_default_beta_max = 1e6;
inline constexpr double ppo_kl_band = 1.5;
inline constexpr int64_t ppo_maximum_rollout_rows = 16384;
inline constexpr std::string_view ppo_sampler_contract = "categorical_fp32_weights_normalized_fp64_v1";

struct PpoObjectiveConfig {
    PpoObjectiveKind kind = PpoObjectiveKind::legacy_clip;
    double target_kl = ppo_default_target_kl;
    double beta_initial = 1;
    double beta_min = ppo_default_beta_min;
    double beta_max = ppo_default_beta_max;
    void validate() const;
    [[nodiscard]] bool enabled() const noexcept { return kind != PpoObjectiveKind::legacy_clip; }
    bool operator==(const PpoObjectiveConfig&) const = default;
};

struct PpoEpochDecision {
    bool threshold_exceeded = false;
    bool stop_remaining_epochs = false;
};

struct PpoControllerState {
    double beta = 1;
    double last_beta = 1;
    int64_t completed_rollouts = 0;
    int64_t optimizer_steps = 0;
    int64_t valid_rows = 0;
    int64_t completed_epochs = 0;
    int64_t skipped_epochs = 0;
    std::optional<double> full_kl;
    bool threshold_exceeded = false;
    void validate(const PpoObjectiveConfig& config, int64_t adam_steps, int64_t epochs,
                  int64_t rollout_limit = ppo_maximum_rollout_rows) const;
    bool operator==(const PpoControllerState&) const = default;
};

[[nodiscard]] std::string_view objective_id(PpoObjectiveKind kind);
[[nodiscard]] PpoObjectiveKind objective_kind(std::string_view id);
[[nodiscard]] PpoEpochDecision epoch_decision(const PpoObjectiveConfig& config,
    int64_t completed_epochs, int64_t configured_epochs, int64_t valid_rows,
    std::optional<double> mean_kl);
[[nodiscard]] double next_beta(const PpoObjectiveConfig& config, double current_beta,
                               std::optional<double> final_kl, int64_t valid_rows);
} // namespace mars_titan::learning
#endif
