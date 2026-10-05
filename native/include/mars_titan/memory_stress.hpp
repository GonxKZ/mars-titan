#ifndef MARS_TITAN_MEMORY_STRESS_HPP
#define MARS_TITAN_MEMORY_STRESS_HPP

#include "mars_titan/episodic_memory.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>

namespace mars_titan::stress {
enum class Retention : uint8_t { uniform, recent, selective };
enum class Scenario : uint8_t { recurrence, persistent, noise, outliers };

inline constexpr std::size_t default_stress_steps = 8192;
inline constexpr std::size_t default_regime_length = 2048;
inline constexpr std::size_t default_feedback_delay = 8;
inline constexpr uint64_t default_data_seed = 42;
inline constexpr uint64_t default_retention_seed = 73;
inline constexpr double default_observation_noise = 0.25;
inline constexpr std::size_t default_invalid_period = 97;

struct Config {
    std::size_t steps = default_stress_steps;
    std::size_t capacity = learning::episodic_memory_capacity;
    std::size_t regime_length = default_regime_length;
    std::size_t delay = default_feedback_delay;
    uint64_t data_seed = default_data_seed;
    uint64_t retention_seed = default_retention_seed;
    double noise = default_observation_noise;
    std::size_t invalid_every = default_invalid_period;
    Scenario scenario = Scenario::recurrence;
    bool operator==(const Config&) const = default;
};

// Todos los candidatos maduros entran. Solo cambia qué fila se expulsa.
class RetentionBank {
  public:
    RetentionBank(Retention policy, std::size_t capacity, uint64_t seed);
    ~RetentionBank();
    RetentionBank(const RetentionBank&) = delete;
    RetentionBank& operator=(const RetentionBank&) = delete;
    RetentionBank(RetentionBank&&) = delete;
    RetentionBank& operator=(RetentionBank&&) = delete;
    void write(const learning::MemoryRecord& record, int64_t confirmed_at, double priority);
    [[nodiscard]] learning::MemoryQuery query(const learning::MemoryVector& key,
                                              int64_t cutoff) const;
    [[nodiscard]] std::string checkpoint() const;
    void restore(std::string_view bytes);
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] uint64_t writes() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class Experiment {
  public:
    explicit Experiment(Config config);
    ~Experiment();
    Experiment(const Experiment&) = delete;
    Experiment& operator=(const Experiment&) = delete;
    Experiment(Experiment&&) = delete;
    Experiment& operator=(Experiment&&) = delete;
    void run_until(std::size_t cursor);
    [[nodiscard]] std::string report() const;
    [[nodiscard]] std::string checkpoint() const;
    void restore(std::string_view bytes);
    [[nodiscard]] const Config& config() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] Config read_config(std::string_view checkpoint);
[[nodiscard]] Scenario scenario_from_name(std::string_view name);
[[nodiscard]] std::string_view scenario_name(Scenario scenario);
inline constexpr std::size_t maximum_stress_archive_bytes = std::size_t{32} * 1024 * 1024;
} // namespace mars_titan::stress
#endif
