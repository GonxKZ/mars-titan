#ifndef MARS_TITAN_KLPO_COLLECTION_HPP
#define MARS_TITAN_KLPO_COLLECTION_HPP

#include "mars_titan/klpo_episodes.hpp"
#include "mars_titan/klpo_terminal.hpp"
#include "mars_titan/policy_context.hpp"
#include "mars_titan/ppo_checkpoints.hpp"
#include "mars_titan/ppo_policy.hpp"

#include <functional>
#include <memory>

namespace mars_titan::learning {
inline constexpr uint64_t default_klpo_collection_seed = 42;
struct KlpoEpisodeObjective {
    at::Tensor per_episode;
    at::Tensor mean;
    std::size_t sampled_decisions = 0;
    bool no_policy_decisions = true;
};

// Solo consume oleadas completas. El denominador incluye episodios sin decisiones.
// Devuelve un sustituto de retropropagación, no una métrica financiera.
// Construye el grafo del actor, sin backward ni pasos de optimizador.
[[nodiscard]] KlpoEpisodeObjective klpo_episode_objective(const PpoPolicy& policy,
                                                          const KlpoEpisodeBatch& batch,
                                                          KlpoTerminalBudget budget = {});

enum class KlpoCollectionPhase : uint8_t { collecting, ready, blocked };
enum class KlpoCollectionBoundary : uint8_t { before_commit, committed };

struct KlpoCollectionOptions {
    std::string fold = "experimental";
    double beta = 1;
    double gamma = 1;
    uint64_t seed = default_klpo_collection_seed;
    std::string device = "cpu";
    std::size_t workers = 1;
    std::size_t record_bytes = maximum_klpo_record_bytes / 2;
    PpoArchitecture architecture;
    PpoLearningOptions context;
};

// Una oleada fija, sin reemplazar episodios ni actualizar al actor.
class KlpoTerminalCollector {
  public:
    KlpoTerminalCollector(std::vector<simulation::BatchInput> inputs,
                          KlpoCollectionOptions options);
    ~KlpoTerminalCollector();
    KlpoTerminalCollector(const KlpoTerminalCollector&) = delete;
    KlpoTerminalCollector& operator=(const KlpoTerminalCollector&) = delete;
    KlpoTerminalCollector(KlpoTerminalCollector&&) = delete;
    KlpoTerminalCollector& operator=(KlpoTerminalCollector&&) = delete;
    bool collect_tick(const std::function<void(KlpoCollectionBoundary)>& failure = {});
    [[nodiscard]] KlpoCollectionPhase phase() const;
    [[nodiscard]] const KlpoEpisodeBatch& records() const&;
    const KlpoEpisodeBatch& records() const&& = delete;
    [[nodiscard]] const PpoPolicy& policy() const&;
    const PpoPolicy& policy() const&& = delete;
    [[nodiscard]] KlpoEpisodeObjective objective(KlpoTerminalBudget budget = {}) const;
    [[nodiscard]] nlohmann::json identity() const;
    [[nodiscard]] PpoCheckpointBundle snapshot() const;
    void restore(const PpoCheckpointBundle& state);
    [[nodiscard]] nlohmann::json save(PpoCheckpointStore& store) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace mars_titan::learning
#endif
