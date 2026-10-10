#ifndef MARS_TITAN_PPO_TRAINING_HPP
#define MARS_TITAN_PPO_TRAINING_HPP

#include "mars_titan/financial_batch.hpp"
#include "mars_titan/ppo_policy.hpp"
#include "mars_titan/policy_context.hpp"
#include "mars_titan/decision_trace.hpp"
#include "mars_titan/learning_replay.hpp"

#include <limits>
#include <string_view>

namespace mars_titan::learning {
using PpoDecisionObserver = std::function<void(std::span<const DecisionRecord>)>;

inline constexpr std::size_t ppo_default_total_transitions = 8192;
inline constexpr std::size_t ppo_default_rollout_transitions = 1024;
inline constexpr uint64_t ppo_default_training_seed = 42;
inline constexpr std::size_t ppo_default_rollout_bytes = std::size_t{128} * 1024 * 1024;

struct PpoTrainingConfig {
    std::size_t total_transitions = ppo_default_total_transitions;
    std::size_t rollout_transitions = ppo_default_rollout_transitions;
    std::size_t workers = 1;
    uint64_t seed = ppo_default_training_seed;
    std::size_t rollout_bytes = ppo_default_rollout_bytes;
    bool operator==(const PpoTrainingConfig&) const = default;
};

struct PpoTrainingState {
    PpoTrainingConfig config;
    PpoHyperparameters hyperparameters;
    std::string device;
    bool diagnostic = true;
    std::size_t transitions = 0;
    std::size_t optimizer_steps = 0;
    std::size_t invalid_transitions = 0;
    std::size_t episodes = 0;
    std::vector<std::size_t> reset_lanes;
    simulation::BatchSnapshot environment;
    PpoRollout rollout;
    std::string policy_archive;
    PpoLearningOptions learning;
    std::vector<std::size_t> source_indices;
    std::size_t next_source = 0;
    std::size_t observed_transitions = 0;
    std::string adaptive_archive;
    PpoObjectiveConfig objective;
    PpoControllerState controller;
};

class PpoTrainer {
public:
    PpoTrainer(std::vector<simulation::BatchInput> inputs, PpoTrainingConfig config,
               PpoHyperparameters hyperparameters, std::string device, bool diagnostic,
               PpoLearningOptions learning = {}, PpoObjectiveConfig objective = {});
    PpoTrainer(const PpoTrainer&) = delete;
    PpoTrainer& operator=(const PpoTrainer&) = delete;
    PpoTrainer(PpoTrainer&&) = delete;
    PpoTrainer& operator=(PpoTrainer&&) = delete;
    ~PpoTrainer() = default;

    // Una llamada confirma N decisiones. La pausa se solicita entre llamadas.
    bool advance(const PpoDecisionObserver& observer = {});
    [[nodiscard]] PpoTrainingState snapshot() const;
    void restore(const PpoTrainingState& state);
    [[nodiscard]] const PpoPolicy& policy() const noexcept;
    [[nodiscard]] std::size_t transitions() const noexcept;
    [[nodiscard]] std::size_t optimizer_steps() const noexcept;
    [[nodiscard]] std::size_t invalid_transitions() const noexcept;
    [[nodiscard]] std::size_t partial_ticks() const noexcept;
    [[nodiscard]] const PpoUpdateStats& last_update() const noexcept;
    [[nodiscard]] std::size_t observed_transitions() const noexcept;
    [[nodiscard]] std::size_t auxiliary_samples() const noexcept;

private:
    std::vector<simulation::BatchInput> inputs_;
    PpoTrainingConfig config_;
    PpoHyperparameters hyperparameters_;
    std::string device_;
    bool diagnostic_;
    std::unique_ptr<simulation::FinancialBatch> batch_;
    std::unique_ptr<PpoPolicy> policy_;
    PpoRollout buffers_;
    std::vector<uint8_t> actions_;
    std::vector<std::size_t> reset_lanes_;
    std::size_t ticks_ = 0;
    std::size_t transitions_ = 0;
    std::size_t optimizer_steps_ = 0;
    std::size_t invalid_transitions_ = 0;
    std::size_t episodes_ = 0;
    bool failed_ = false;
    PpoUpdateStats last_update_;
    PpoLearningOptions learning_;
    std::vector<simulation::BatchInput> active_inputs_;
    std::vector<std::size_t> source_indices_;
    std::size_t next_source_ = 0;
    std::size_t observed_transitions_ = 0;
    std::size_t collector_ticks_ = 0;
    std::unique_ptr<PolicyContext> context_;
    at::Tensor hidden_;
    at::Tensor history_;
    at::Tensor history_lengths_;
    std::unique_ptr<LearningReplay> replay_;
    std::size_t auxiliary_samples_ = 0;
    PpoObjectiveConfig objective_;
    // Forward del bootstrap del paso anterior en PPO MLP sin contexto. El paso siguiente lo
    // reutiliza para muestrear si su observación es idéntica y los pesos no han cambiado.
    // No forma parte del estado recuperable.
    struct BootstrapCache {
        at::Tensor observation;
        PpoInference output;
        std::size_t optimizer_steps = 0;
    };
    std::optional<BootstrapCache> bootstrap_;

    void reset_pending();
    void rebuild_hidden();
};

struct PpoEvaluation {
    bool paused = false;
    std::size_t episodes = 0;
    std::size_t incomplete = 0;
    std::size_t ruined = 0;
    // Una evaluación pausada, vacía o incompleta no tiene puntuación y conserva NaN.
    double mean_log_growth = std::numeric_limits<double>::quiet_NaN();
    // La misma media con la cartera final vendida al último cierre y su coste.
    double mean_liquidated_log_growth = std::numeric_limits<double>::quiet_NaN();
    std::vector<simulation::FinancialMetrics> metrics;
};

[[nodiscard]] PpoEvaluation evaluate_policy(const PpoPolicy& policy,
                                            std::vector<simulation::BatchInput> inputs,
                                            std::size_t workers,
                                            const std::function<bool()>& stop = {},
                                            const PpoLearningOptions& learning = {},
                                            const PpoDecisionObserver& observer = {},
                                            bool memory_sensitivity = false);
[[nodiscard]] std::string serialize_rollout(const PpoRollout& rollout);
[[nodiscard]] PpoRollout deserialize_rollout(std::string_view bytes);
[[nodiscard]] std::string serialize_training_buffer(const PpoTrainingState& state);
void restore_training_buffer(std::string_view bytes, PpoTrainingState& state);
}
#endif
