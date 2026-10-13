#ifndef MARS_TITAN_DECISION_TRACE_HPP
#define MARS_TITAN_DECISION_TRACE_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>

namespace mars_titan::learning {
inline constexpr std::size_t trace_action_count = 6;
inline constexpr std::size_t trace_memory_count = 4;
inline constexpr std::size_t trace_shard_records = 128;
inline constexpr std::size_t default_trace_bytes = std::size_t{512} * 1024 * 1024;

struct MemorySensitivity {
    std::array<float, trace_action_count> probabilities{};
    uint8_t action = 0;
    double probability_l1 = 0;
    bool action_changed = false;
};

struct DecisionRecord {
    uint64_t decision_id = 0;
    uint32_t lane = 0;
    std::string world_sha256;
    std::string context_sha256;
    uint64_t episode = 0;
    uint64_t cursor = 0;
    uint64_t optimizer_step = 0;
    int64_t decision_at = 0;
    int64_t outcome_at = 0;
    uint8_t action = 0;
    std::string mode = "sampled";
    bool learning_allowed = true;
    std::array<float, trace_action_count> probabilities{};
    // Salidas de la red antes de elegir la acción: logits en PPO y KLPO, valores Q en Double
    // DQN. Solo la evaluación congelada las rellena, para comparar dispositivos y empates.
    std::array<float, trace_action_count> logits{};
    double critic = 0;
    uint8_t retrieved_count = 0;
    std::array<uint64_t, trace_memory_count> ids{};
    std::array<double, trace_memory_count> similarities{};
    std::array<int64_t, trace_memory_count> matured_at{};
    double reward = 0;
    bool reward_valid = false;
    bool terminated = false;
    bool truncated = false;
    std::optional<double> costs_delta;
    std::optional<MemorySensitivity> memory_sensitivity;
};

class TraceCapacityError : public std::runtime_error {
  public:
    using std::runtime_error::runtime_error;
};

class TraceWriter {
  public:
    // IDs consecutivos desde uno. cursor() es el último ID admitido, cero si no hay filas.
    TraceWriter(const std::filesystem::path& directory, std::string identity_sha256,
                bool resume = false, uint64_t confirmed_cursor = 0,
                std::size_t budget_bytes = default_trace_bytes);
    ~TraceWriter();
    TraceWriter(const TraceWriter&) = delete;
    TraceWriter& operator=(const TraceWriter&) = delete;
    TraceWriter(TraceWriter&&) = delete;
    TraceWriter& operator=(TraceWriter&&) = delete;
    // El lote queda admitido entero o se conserva el estado anterior.
    void append(std::span<const DecisionRecord> records);
    // Publica el índice de todos los bloques admitidos antes de confirmar el checkpoint.
    void flush();
    [[nodiscard]] uint64_t cursor() const noexcept;
    // Incluye una reserva de 1 MiB para que el próximo flush no exceda el presupuesto.
    [[nodiscard]] std::size_t bytes() const noexcept;
    void rewind_to(uint64_t confirmed_cursor);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace mars_titan::learning
#endif
