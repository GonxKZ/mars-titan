#ifndef MARS_TITAN_PPO_POLICY_HPP
#define MARS_TITAN_PPO_POLICY_HPP

#include "mars_titan/ppo_controller.hpp"

#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

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
inline constexpr int64_t default_ppo_hidden_width = 64;
inline constexpr int64_t ppo_sequence_length = 16;
inline constexpr int64_t ppo_maximum_history = 256;
inline constexpr int64_t ppo_auxiliary_batch_size = 64;
inline constexpr std::size_t default_dqn_target_interval = 256;
inline constexpr std::size_t dqn_warmup_steps = 256;

enum class PpoNetworkKind : uint8_t { mlp, gru };

struct PpoArchitecture {
    PpoNetworkKind kind = PpoNetworkKind::mlp;
    int64_t hidden_width = default_ppo_hidden_width;
    bool auxiliary = false;
    bool double_dqn = false;
    // QR-DQN: cero conserva la cabeza escalar. Con N > 0 la cabeza da N cuantiles por acción y
    // la acción se elige con la media de los risk_alpha*N primeros niveles (1 = media, CVaR si < 1).
    int64_t quantiles = 0;
    double risk_alpha = 1;
    void validate() const;
    bool operator==(const PpoArchitecture&) const = default;
};

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

struct PpoTerminalAdamOptions {
    double learning_rate = 0;
    // Cero desactiva el clipping. El resto del contrato Adam permanece cerrado.
    double gradient_norm = 0;
    void validate() const;
    bool operator==(const PpoTerminalAdamOptions&) const noexcept = default;
};

struct PpoInference {
    at::Tensor logits;
    at::Tensor values;
    at::Tensor next_state;
};

struct PpoAction {
    at::Tensor packed;
    at::Tensor next_state;
    at::Tensor probabilities;
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
    // GRU: reinicios [T,N], prefijo FP32 [P,N,D] y longitudes int64 [N].
    // El prefijo contiene toda la historia anterior al rollout desde el último reinicio.
    at::Tensor episode_starts = {};
    at::Tensor prefix_observations = {};
    at::Tensor prefix_lengths = {};
    // Solo objetivos explícitos: los seis pesos FP32 usados al muestrear [T,N,6].
    at::Tensor old_action_weights = {};
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
    std::optional<double> full_kl;
    int64_t completed_epochs = 0;
    int64_t skipped_epochs = 0;
    bool threshold_exceeded = false;
};

struct PpoAuxiliaryStats {
    int64_t samples = 0;
    int64_t steps = 0;
    double mae_before = 0;
    double mae_after = 0;
    double gradient_norm = 0;
};

struct DqnBatch {
    // Observaciones FP32 [B,D], acciones int64 y máscaras bool [B].
    // Una truncación conserva la observación final y no activa terminated.
    at::Tensor observations;
    at::Tensor next_observations;
    at::Tensor actions;
    at::Tensor rewards;
    at::Tensor terminated;
    at::Tensor reward_valid;
};

// Pérdida de Double DQN o QR-DQN sobre las filas válidas, con su grafo y sin paso del optimizador.
struct DqnLoss {
    at::Tensor loss;
    int64_t valid_transitions = 0;
};

struct DqnUpdateStats {
    int64_t valid_transitions = 0;
    int64_t updates = 0;
    double loss = 0;
    double gradient_norm = 0;
    bool target_synchronized = false;
};

[[nodiscard]] at::Tensor ppo_clipped_objective(const at::Tensor& log_probabilities,
                                              const at::Tensor& old_log_probabilities,
                                              const at::Tensor& advantages, double clip);
[[nodiscard]] PpoAdvantages ppo_gae(const PpoRollout& rollout,
                                   const PpoHyperparameters& parameters);
[[nodiscard]] at::Tensor double_dqn_targets(const at::Tensor& rewards, const at::Tensor& terminated,
                                            const at::Tensor& online_next, const at::Tensor& target_next,
                                            double gamma);

// Un controlador por política. La recuperación exige el mismo dispositivo.
class PpoPolicy {
public:
    explicit PpoPolicy(std::size_t observation_width, const PpoHyperparameters& parameters,
                       uint64_t seed, std::string_view device = "cpu",
                       std::size_t memory_budget = default_ppo_memory_bytes,
                       PpoArchitecture architecture = {}, PpoObjectiveConfig objective = {});
    PpoPolicy(const PpoPolicy&) = delete;
    PpoPolicy& operator=(const PpoPolicy&) = delete;
    PpoPolicy(PpoPolicy&&) noexcept;
    PpoPolicy& operator=(PpoPolicy&&) noexcept;
    ~PpoPolicy();

    [[nodiscard]] PpoForward forward(const at::Tensor& observations) const;
    // FP64 [N,3] con acción, logaritmo de su probabilidad y valor, en el dispositivo.
    [[nodiscard]] at::Tensor act(const at::Tensor& observations, bool deterministic = false);
    [[nodiscard]] at::Tensor values(const at::Tensor& observations) const;
    [[nodiscard]] at::Tensor initial_state(std::size_t batch_size) const;
    [[nodiscard]] at::Tensor state_from_history(const at::Tensor& observations,
                                               const at::Tensor& lengths) const;
    [[nodiscard]] PpoInference infer(const at::Tensor& observations, const at::Tensor& state = {},
                                     const at::Tensor& episode_starts = {}) const;
    // Historia completa [H,N,D], longitudes [N]. Conserva el modo autograd del llamante.
    // No utiliza hidden histórico, no desacopla prefijos y no muestrea ni optimiza.
    [[nodiscard]] PpoForward terminal_forward(const at::Tensor& history,
                                             const at::Tensor& lengths) const;
    [[nodiscard]] PpoAction act_recurrent(const at::Tensor& observations, const at::Tensor& state = {},
                                          const at::Tensor& episode_starts = {}, bool deterministic = false);
    // Muestreo de act_recurrent sobre una inferencia ya calculada por esta política con los
    // pesos actuales. Consume el mismo generador y no vuelve a ejecutar la red.
    [[nodiscard]] PpoAction sample(const PpoInference& output, bool deterministic = false);
    // Normaliza las ventajas válidas con desviación poblacional y suelo de 1e-8.
    // Un fallo durante la optimización exige recuperar un checkpoint confirmado.
    [[nodiscard]] PpoUpdateStats update(const PpoRollout& rollout);
    // PPO MLP con rollout íntegro en CPU y GAE FP64 antes del traslado de filas válidas.
    // La conversión a FP32 y la normalización usan el dispositivo de la política.
    // Rechaza GRU y Double DQN.
    [[nodiscard]] PpoUpdateStats update_from_cpu(const PpoRollout& rollout);
    // Diagnóstico completo con los parámetros actuales, sin gradiente ni cambios de RNG.
    [[nodiscard]] std::optional<double> full_kl(const PpoRollout& rollout) const;
    // Solo MLP64 auxiliar. Admite entrada CPU y transfiere las filas elegidas al dispositivo.
    // Muestrea hasta 64 filas maduras por paso, con RNG independiente.
    [[nodiscard]] PpoAuxiliaryStats consolidate(const at::Tensor& observations,
                                                const at::Tensor& matured_rewards, int64_t steps = 1);
    [[nodiscard]] PpoAction act_double_dqn(const at::Tensor& observations, double epsilon);
    // Cuantiles [N,6,Q] de QR-DQN en el dispositivo de la política. Conserva el modo autograd.
    [[nodiscard]] at::Tensor action_quantiles(const at::Tensor& observations) const;
    // Pérdida SmoothL1 de Double DQN o cuantílica de QR-DQN, sin backward ni pasos de Adam.
    [[nodiscard]] DqnLoss double_dqn_loss(const DqnBatch& batch) const;
    // Un fallo después de Adam exige recuperar el último checkpoint confirmado.
    [[nodiscard]] DqnUpdateStats update_double_dqn(const DqnBatch& batch, std::size_t environment_step,
                                                   std::size_t target_interval = default_dqn_target_interval);
    void save(std::ostream& destination) const;
    [[nodiscard]] static PpoPolicy load(std::istream& source, std::string_view device = "cpu");
    [[nodiscard]] PpoRandomState random_state() const;
    void restore_random_state(const PpoRandomState& state);

    [[nodiscard]] std::size_t observation_width() const noexcept;
    [[nodiscard]] std::size_t memory_budget() const noexcept;
    [[nodiscard]] std::string critic_fingerprint() const;
    [[nodiscard]] const PpoHyperparameters& hyperparameters() const noexcept;
    [[nodiscard]] const std::string& device() const noexcept;
    [[nodiscard]] uint64_t seed() const;
    [[nodiscard]] std::size_t optimizer_steps() const;
    [[nodiscard]] std::size_t auxiliary_steps() const;
    [[nodiscard]] const PpoObjectiveConfig& objective() const noexcept;
    [[nodiscard]] const PpoControllerState& controller_state() const noexcept;
    [[nodiscard]] std::size_t dqn_environment_step() const noexcept;
    [[nodiscard]] std::size_t target_sync_step() const noexcept;
    [[nodiscard]] const PpoArchitecture& architecture() const noexcept;
    [[nodiscard]] std::size_t parameter_count() const noexcept;
    // Parámetros FP32 y geometría, sin RNG, gradientes, Adam ni direcciones de memoria.
    [[nodiscard]] std::string parameter_fingerprint() const;

    // Ruta terminal optativa. No reutiliza el optimizador de PPO.
    void enable_terminal_adam(PpoTerminalAdamOptions options);
    [[nodiscard]] bool terminal_adam_enabled() const noexcept;
    [[nodiscard]] PpoTerminalAdamOptions terminal_adam_options() const;
    void terminal_zero_grad();
    [[nodiscard]] std::vector<at::Tensor> terminal_gradients() const;
    [[nodiscard]] double validate_terminal_gradients() const;
    // Un fallo desde el inicio de Adam invalida el objeto hasta recuperar otro checkpoint.
    [[nodiscard]] double terminal_step();
    [[nodiscard]] PpoPolicy frozen_reference(const PpoRandomState& initial_rng) const;
    [[nodiscard]] static PpoPolicy load_terminal(std::istream& source,
                                                std::string_view device = "cpu");

private:
    [[nodiscard]] static PpoPolicy load_mode(std::istream& source, std::string_view device,
                                            bool terminal);
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
#endif
