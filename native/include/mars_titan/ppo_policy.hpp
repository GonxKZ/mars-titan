#ifndef MARS_TITAN_PPO_POLICY_HPP
#define MARS_TITAN_PPO_POLICY_HPP

#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>

namespace mars_titan::learning {

inline constexpr int64_t ppo_action_count = 6;
inline constexpr std::size_t default_ppo_memory_bytes = std::size_t{512} * 1024 * 1024;
inline constexpr double default_ppo_learning_rate = 3e-4;
inline constexpr double default_ppo_gamma = 0.99;
inline constexpr double default_ppo_gae_lambda = 0.95;
inline constexpr double default_ppo_clip = 0.2;
inline constexpr double default_ppo_entropy = 0.01;
inline constexpr double default_ppo_value_weight = 0.5;
inline constexpr int64_t default_ppo_minibatch_size = 64;

struct PpoHyperparameters {
    double learning_rate = default_ppo_learning_rate;
    double gamma = default_ppo_gamma;
    double gae_lambda = default_ppo_gae_lambda;
    double clip = default_ppo_clip;
    double entropy = default_ppo_entropy;
    double value_weight = default_ppo_value_weight;
    double gradient_norm = 1.0;
    int64_t epochs = 4;
    int64_t minibatch_size = default_ppo_minibatch_size;
    void validate() const;
    bool operator==(const PpoHyperparameters&) const = default;
};

struct PpoForward {
    at::Tensor logits;
    at::Tensor values;
};

struct PpoRandomState {
    at::Tensor sampling;
    at::Tensor shuffle;
};

// Observaciones FP32 [T,N,D], acciones int64 y máscaras bool [T,N].
// Las magnitudes restantes aceptan FP32 o FP64 y no conservan gradientes.
struct PpoRollout {
    at::Tensor observations;
    at::Tensor actions;
    at::Tensor old_log_probabilities;
    at::Tensor old_values;
    at::Tensor rewards;
    at::Tensor next_values;
    at::Tensor reward_valid;
    at::Tensor terminated;
    at::Tensor truncated;
};

struct PpoAdvantages {
    at::Tensor advantages;
    at::Tensor returns;
};

struct PpoUpdateStats {
    int64_t valid_transitions = 0;
    int64_t minibatches = 0;
    double policy_loss = 0;
    double value_loss = 0;
    double entropy = 0;
    double approximate_kl = 0;
    double clip_fraction = 0;
    double gradient_norm = 0;
};

[[nodiscard]] at::Tensor ppo_clipped_objective(const at::Tensor& log_probabilities,
                                              const at::Tensor& old_log_probabilities,
                                              const at::Tensor& advantages, double clip);
[[nodiscard]] PpoAdvantages ppo_gae(const PpoRollout& rollout,
                                   const PpoHyperparameters& parameters);

// Un controlador por política. La recuperación exige el mismo dispositivo.
class PpoPolicy {
public:
    explicit PpoPolicy(std::size_t observation_width, const PpoHyperparameters& parameters,
                       uint64_t seed, std::string_view device = "cpu",
                       std::size_t memory_budget = default_ppo_memory_bytes);
    PpoPolicy(const PpoPolicy&) = delete;
    PpoPolicy& operator=(const PpoPolicy&) = delete;
    PpoPolicy(PpoPolicy&&) noexcept;
    PpoPolicy& operator=(PpoPolicy&&) noexcept;
    ~PpoPolicy();

    [[nodiscard]] PpoForward forward(const at::Tensor& observations) const;
    // FP64 [N,3] con acción, logaritmo de su probabilidad y valor, en el dispositivo.
    [[nodiscard]] at::Tensor act(const at::Tensor& observations, bool deterministic = false);
    [[nodiscard]] at::Tensor values(const at::Tensor& observations) const;
    // Normaliza las ventajas válidas con desviación poblacional y suelo de 1e-8.
    // Un fallo durante la optimización exige recuperar un checkpoint confirmado.
    [[nodiscard]] PpoUpdateStats update(const PpoRollout& rollout);
    void save(std::ostream& destination) const;
    [[nodiscard]] static PpoPolicy load(std::istream& source, std::string_view device = "cpu");
    [[nodiscard]] PpoRandomState random_state() const;
    void restore_random_state(const PpoRandomState& state);

    [[nodiscard]] std::size_t observation_width() const noexcept;
    [[nodiscard]] const PpoHyperparameters& hyperparameters() const noexcept;
    [[nodiscard]] const std::string& device() const noexcept;
    [[nodiscard]] uint64_t seed() const;
    [[nodiscard]] std::size_t optimizer_steps() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
#endif
