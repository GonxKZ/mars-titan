#ifndef MARS_TITAN_KLPO_LEARNING_HPP
#define MARS_TITAN_KLPO_LEARNING_HPP

#include "mars_titan/klpo_collection.hpp"

namespace mars_titan::learning {
inline constexpr std::size_t default_klpo_gradient_block = 8;

struct KlpoLearningConfig {
    KlpoCollectionOptions collection;
    PpoTerminalAdamOptions adam;
    std::size_t confirmed_updates_per_reference = 2;
    std::size_t gradient_block_episodes = default_klpo_gradient_block;
    void validate() const;
};

struct KlpoLearningCounters {
    uint64_t consumed_waves = 0;
    uint64_t confirmed_updates = 0;
    uint64_t reference_version = 0;
    uint64_t updates_since_reference = 0;
    void validate(std::size_t cadence) const;
    bool operator==(const KlpoLearningCounters&) const noexcept = default;
};

// Transiciones puras de contadores. No acreditan por sí solas una actualización.
[[nodiscard]] KlpoLearningCounters klpo_after_consumption(KlpoLearningCounters previous,
                                                          std::size_t cadence, bool updated);
[[nodiscard]] KlpoLearningCounters klpo_before_collection(KlpoLearningCounters previous,
                                                          std::size_t cadence);

enum class KlpoLearningPhase : uint8_t { collecting, ready, blocked, consumed };
enum class KlpoLearningBoundary : uint8_t {
    before_update,
    after_update,
    before_publish,
    confirmed
};

struct KlpoGradientSummary {
    std::size_t episodes = 0;
    std::size_t sampled_decisions = 0;
    std::size_t blocks = 0;
    double surrogate = 0;
    double gradient_norm = 0;
    bool no_policy_decisions = true;
};

class KlpoLearningController {
  public:
    KlpoLearningController(std::vector<simulation::BatchInput> inputs, KlpoLearningConfig config);
    ~KlpoLearningController();
    KlpoLearningController(const KlpoLearningController&) = delete;
    KlpoLearningController& operator=(const KlpoLearningController&) = delete;
    KlpoLearningController(KlpoLearningController&&) = delete;
    KlpoLearningController& operator=(KlpoLearningController&&) = delete;

    bool collect_tick();
    [[nodiscard]] KlpoLearningPhase phase() const;
    [[nodiscard]] KlpoLearningCounters counters() const;
    [[nodiscard]] KlpoGradientSummary backward_ready();
    [[nodiscard]] std::vector<at::Tensor> gradient_snapshot() const;
    [[nodiscard]] std::string actor_fingerprint() const;
    [[nodiscard]] std::string reference_fingerprint() const;
    [[nodiscard]] nlohmann::json identity() const;
    [[nodiscard]] PpoCheckpointBundle snapshot() const;
    [[nodiscard]] nlohmann::json save(PpoCheckpointStore& store) const;
    void restore(const PpoCheckpointBundle& state);

    void consume_without_update(PpoCheckpointStore& store,
                                const std::function<void(KlpoLearningBoundary)>& failure = {});
    void start_next_wave(PpoCheckpointStore& store,
                         const std::function<void(KlpoLearningBoundary)>& failure = {});
    // Ruta real de Adam. Su ejecución permanece pendiente bajo el bloqueo de aprendizaje.
    [[nodiscard]] KlpoGradientSummary
    update_ready(PpoCheckpointStore& store,
                 const std::function<void(KlpoLearningBoundary)>& failure = {});

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace mars_titan::learning
#endif
