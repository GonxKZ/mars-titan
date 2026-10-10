#ifndef MARS_TITAN_POLICY_CONTEXT_HPP
#define MARS_TITAN_POLICY_CONTEXT_HPP

#include "mars_titan/episodic_memory.hpp"
#include "mars_titan/financial_batch.hpp"
#include "mars_titan/markov_filter.hpp"

namespace mars_titan::learning {
inline constexpr std::size_t policy_window = 16;
inline constexpr uint64_t policy_projection_seed = 1729;

// Variantes que aprenden valores de acción con Double DQN. qr_dqn y qr_dqn_cvar cambian el valor
// escalar por cuantiles y comparten con double_dqn la representación, la réplica y la recogida.
[[nodiscard]] bool value_variant(std::string_view variant) noexcept;

struct PpoLearningOptions {
    bool enabled = false;
    std::string variant = "ppo";
    std::string fold = "experimental";
    std::string representation = "fixed_projection_1729_v1";
    std::size_t environments = policy_window;
    std::optional<std::size_t> trading_field;
    std::optional<simulation::MarkovParameters> markov;
    std::vector<std::size_t> markov_fields;
    bool operator==(const PpoLearningOptions&) const = default;
};

struct PolicyContextSnapshot {
    std::string archive;
};

// El contexto confirma memoria, feedback y filtro junto al paso contable.
class PolicyContext {
  public:
    PolicyContext(const std::vector<simulation::BatchInput>& active,
                  const PpoLearningOptions& options, uint64_t seed,
                  std::span<const float> observations);
    ~PolicyContext();
    PolicyContext(const PolicyContext&) = delete;
    PolicyContext& operator=(const PolicyContext&) = delete;
    PolicyContext(PolicyContext&&) noexcept;
    PolicyContext& operator=(PolicyContext&&) noexcept;

    [[nodiscard]] const at::Tensor& observations() const;
    [[nodiscard]] at::Tensor without_retrieved_memory() const;
    [[nodiscard]] std::vector<uint8_t> training_mask() const;
    [[nodiscard]] const std::vector<MemoryQuery>& retrieved() const;
    [[nodiscard]] const at::Tensor& prepare(std::span<const float> next_observations,
                                            std::span<const uint8_t> actions,
                                            const simulation::BatchTransition& transition,
                                            std::span<const uint8_t> active = {});
    void commit();
    void cancel() noexcept;
    void reset(std::size_t lane, const simulation::BatchInput& input,
               std::span<const float> observation);
    [[nodiscard]] PolicyContextSnapshot snapshot() const;
    void restore(const PolicyContextSnapshot& snapshot);
    [[nodiscard]] std::size_t observation_width() const noexcept;
    [[nodiscard]] std::size_t cursor(std::size_t lane) const;
    [[nodiscard]] MemoryScope memory_scope(std::size_t lane) const;
    [[nodiscard]] uint64_t episode(std::size_t lane) const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace mars_titan::learning
#endif
