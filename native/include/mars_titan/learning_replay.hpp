#ifndef MARS_TITAN_LEARNING_REPLAY_HPP
#define MARS_TITAN_LEARNING_REPLAY_HPP

#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace mars_titan::learning {
inline constexpr std::size_t learning_replay_batch = 64;
inline constexpr std::size_t auxiliary_replay_capacity = 8192;
inline constexpr std::size_t dqn_replay_capacity = 4096;
inline constexpr std::size_t dqn_learning_warmup = 256;
inline constexpr std::size_t auxiliary_replay_bytes = std::size_t{32} * 1024 * 1024;
inline constexpr std::size_t dqn_replay_bytes = std::size_t{128} * 1024 * 1024;
enum class ReplayMode : uint8_t { recent, reservoir };

struct ReplaySample {
    at::Tensor observations;
    at::Tensor next_observations;
    at::Tensor actions;
    at::Tensor rewards;
    at::Tensor terminated;
};
struct ReplaySnapshot {
    std::size_t width = 0;
    std::size_t capacity = 0;
    std::size_t max_bytes = 0;
    ReplayMode mode = ReplayMode::recent;
    bool store_next = false;
    uint64_t seed = 0;
    uint64_t seen = 0;
    std::size_t size = 0;
    std::size_t position = 0;
    ReplaySample data;
    at::Tensor sampling_rng;
    at::Tensor selection_rng;
};

// Un controlador. El collector aporta solo recompensas maduras de entrenamiento.
class LearningReplay {
  public:
    LearningReplay(std::size_t width, std::size_t capacity, ReplayMode mode, bool store_next,
                   uint64_t seed, std::size_t max_bytes);
    LearningReplay(const LearningReplay&) = delete;
    LearningReplay& operator=(const LearningReplay&) = delete;
    LearningReplay(LearningReplay&&) noexcept;
    LearningReplay& operator=(LearningReplay&&) noexcept;
    ~LearningReplay();

    void add(const at::Tensor& observation, int64_t action, double reward, bool terminated,
             const at::Tensor& next_observation = {}, bool reward_valid = true);
    [[nodiscard]] ReplaySample sample(std::size_t count = learning_replay_batch);
    // Vistas prestadas del prefijo válido. El consumidor no debe modificarlas.
    [[nodiscard]] at::Tensor observations() const;
    [[nodiscard]] at::Tensor rewards() const;
    [[nodiscard]] ReplaySnapshot snapshot() const;
    void restore(const ReplaySnapshot& state);
    [[nodiscard]] uint64_t seen() const noexcept;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t capacity() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] std::string serialize_replay(const ReplaySnapshot& snapshot);
[[nodiscard]] ReplaySnapshot deserialize_replay(std::string_view archive);
} // namespace mars_titan::learning
#endif
