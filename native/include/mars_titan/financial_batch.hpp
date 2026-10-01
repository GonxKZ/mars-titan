#ifndef MARS_TITAN_FINANCIAL_BATCH_HPP
#define MARS_TITAN_FINANCIAL_BATCH_HPP

#include "mars_titan/financial_session.hpp"

#include <condition_variable>
#include <exception>
#include <functional>
#include <mutex>
#include <stop_token>
#include <thread>

namespace mars_titan::simulation {

inline constexpr std::size_t maximum_environments = 4096;
inline constexpr std::size_t maximum_context_fields = 512;
inline constexpr std::size_t default_batch_bytes = std::size_t{512} * 1024 * 1024;
inline constexpr std::size_t maximum_batch_workers = 8;

struct ContextField {
    std::string name;
    std::string unit;
    bool operator==(const ContextField&) const = default;
};

struct ContextValue {
    float value = 0;
    bool present = false;
    int64_t available_at = 0;
};

struct ContextTape {
    std::string source_sha256;
    std::vector<ContextField> fields;
    std::vector<ContextValue> values;
    void validate(const MarketTape& market) const;
};

struct BatchInput {
    std::shared_ptr<const MarketTape> tape;
    Parameters parameters;
    std::optional<ContextTape> context;
};

struct BatchTransition {
    std::vector<double> rewards;
    std::vector<uint8_t> reward_valid;
    std::vector<uint8_t> terminated;
    std::vector<uint8_t> truncated;
    std::vector<double> costs = {};
};

struct BatchSnapshot {
    std::vector<SessionSnapshot> sessions;
    std::vector<std::string> context_sources;
};

using BatchValidator = std::function<void(std::span<const float>, const BatchTransition&)>;

// Un único controlador por lote. Los trabajadores solo preparan estados independientes.
class FinancialBatch {
public:
    explicit FinancialBatch(std::vector<BatchInput> inputs, std::size_t workers = 1,
                            std::size_t memory_budget = default_batch_bytes);
    FinancialBatch(const FinancialBatch&) = delete;
    FinancialBatch& operator=(const FinancialBatch&) = delete;
    FinancialBatch(FinancialBatch&&) = delete;
    FinancialBatch& operator=(FinancialBatch&&) = delete;
    ~FinancialBatch();

    [[nodiscard]] const BatchTransition& step(std::span<const uint8_t> actions);
    [[nodiscard]] const BatchTransition& step_active(std::span<const uint8_t> actions,
                                                     std::span<const uint8_t> active);
    // El validador ve el resultado provisional y no puede modificar este lote.
    [[nodiscard]] const BatchTransition& step_checked(std::span<const uint8_t> actions,
                                                      std::span<const uint8_t> active,
                                                      const BatchValidator& validate);
    [[nodiscard]] std::span<const float> observations() const & noexcept;
    std::span<const float> observations() const && = delete;
    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] std::size_t observation_width() const noexcept;
    [[nodiscard]] std::size_t workers() const noexcept;
    [[nodiscard]] std::size_t reserved_payload_bytes() const noexcept;
    [[nodiscard]] BatchSnapshot snapshot() const;
    [[nodiscard]] FinancialMetrics metrics(std::size_t lane) const;
    [[nodiscard]] std::size_t cursor(std::size_t lane) const;
    void restore(const BatchSnapshot& state);
    void reset(std::span<const std::size_t> indices);

private:
    void observe(std::size_t lane, const FinancialSession& session,
                 const SessionSnapshot& state, std::span<float> destination) const;
    void prepare_range(std::size_t worker);
    void worker_loop(const std::stop_token& stop, std::size_t worker);
    void replace(std::span<const std::size_t> indices,
                 const std::vector<SessionSnapshot>* snapshots);

    std::vector<BatchInput> inputs_;
    std::vector<FinancialSession> sessions_;
    std::size_t width_ = 0;
    std::size_t workers_ = 1;
    std::size_t payload_bytes_ = 0;
    std::vector<float> observations_;
    std::vector<float> staged_observations_;
    BatchTransition transition_;
    BatchTransition staged_transition_;
    std::span<const uint8_t> actions_;
    std::vector<uint8_t> active_;
    bool stepping_ = false;
    std::vector<std::exception_ptr> errors_;
    std::mutex mutex_;
    std::condition_variable_any work_;
    std::condition_variable completion_;
    std::size_t generation_ = 0;
    std::size_t completed_ = 0;
    // Se destruyen primero para que los hilos terminen antes que sus buffers y bloqueos.
    std::vector<std::jthread> threads_;
};

}
#endif
