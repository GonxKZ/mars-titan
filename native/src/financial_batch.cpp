#include "mars_titan/financial_batch.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <unordered_set>

namespace mars_titan::simulation {
namespace {
constexpr std::size_t financial_fields = 6;
constexpr std::size_t account_fields = 2;
constexpr std::size_t context_components = 3;
constexpr std::size_t maximum_identifier = 256;
constexpr std::size_t digest_length = 64;
constexpr uint8_t action_count = 6;
constexpr double microseconds_per_day = 86'400'000'000.0;

bool valid_digest(const std::string& digest) {
    return digest.size() == digest_length &&
           std::all_of(digest.begin(), digest.end(), [](char value) {
               return (value >= '0' && value <= '9') || (value >= 'a' && value <= 'f');
           });
}

void account_bytes(std::size_t count, std::size_t width, std::size_t budget,
                   std::size_t& total) {
    if (total > budget || count > (budget - total) / width) {
        throw std::invalid_argument("El lote supera el presupuesto de buffers y estados");
    }
    total += count * width;
}

void resize_transition(BatchTransition& transition, std::size_t size) {
    transition.rewards.resize(size);
    transition.reward_valid.resize(size);
    transition.terminated.resize(size);
    transition.truncated.resize(size);
}

std::string context_source(const BatchInput& input) {
    if (input.context.has_value()) {
        return input.context->source_sha256;
    }
    return {};
}

class StepGuard {
public:
    explicit StepGuard(bool& active) : active_(active) {
        if (active_) {
            throw std::logic_error("El lote no admite una operación reentrante");
        }
        active_ = true;
    }
    StepGuard(const StepGuard&) = delete;
    StepGuard& operator=(const StepGuard&) = delete;
    StepGuard(StepGuard&&) = delete;
    StepGuard& operator=(StepGuard&&) = delete;
    ~StepGuard() { active_ = false; }
private:
    bool& active_;
};
}

void ContextTape::validate(const MarketTape& market) const {
    if (!valid_digest(source_sha256) || fields.empty() ||
        fields.size() > maximum_context_fields ||
        market.close_times.size() > maximum_sessions ||
        values.size() != fields.size() * market.close_times.size()) {
        throw std::invalid_argument("La identidad o las dimensiones del contexto no son válidas");
    }
    std::unordered_set<std::string> names;
    for (const auto& field : fields) {
        if (field.name.empty() || field.name.size() > maximum_identifier ||
            field.unit.empty() || field.unit.size() > maximum_identifier ||
            !names.insert(field.name).second) {
            throw std::invalid_argument("El contexto necesita nombres únicos y unidades explícitas");
        }
    }
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto& value = values[index];
        const auto decision = market.close_times[index / fields.size()];
        if (!std::isfinite(value.value) || value.available_at < 0 ||
            (value.present && value.available_at > decision) ||
            (!value.present && (value.value != 0 || value.available_at != 0))) {
            throw std::invalid_argument("El contexto contiene futuro, valores inválidos o ausencias ambiguas");
        }
    }
}

FinancialBatch::FinancialBatch(std::vector<BatchInput> inputs, std::size_t workers,
                               std::size_t memory_budget)
    : inputs_(std::move(inputs)) {
    if (inputs_.empty() || inputs_.size() > maximum_environments ||
        (workers != 1 && workers != 2 && workers != 4 && workers != maximum_batch_workers) ||
        memory_budget == 0 || memory_budget > default_batch_bytes) {
        throw std::invalid_argument("El número de entornos, trabajadores o presupuesto no está admitido");
    }
    workers_ = std::min(workers, inputs_.size());
    const auto& first = inputs_.front();
    if (!first.tape) {
        throw std::invalid_argument("El lote necesita cintas de mercado");
    }
    const auto context_width = first.context ? first.context->fields.size() : 0;
    if (first.tape->assets.size() > maximum_assets || context_width > maximum_context_fields) {
        throw std::invalid_argument("La observación supera las dimensiones admitidas");
    }
    width_ = first.tape->assets.size() * financial_fields + account_fields +
             context_components * context_width;
    account_bytes(size(), 2 * width_ * sizeof(float), memory_budget, payload_bytes_);
    account_bytes(size(), 2 * (sizeof(double) + 3 * sizeof(uint8_t)), memory_budget, payload_bytes_);
    account_bytes(size(), sizeof(uint8_t), memory_budget, payload_bytes_);
    std::unordered_set<const MarketTape*> validated_tapes;
    for (const auto& input : inputs_) {
        if (!input.tape || input.tape->assets != first.tape->assets ||
            input.tape->partition != first.tape->partition ||
            input.tape->currency != first.tape->currency ||
            input.context.has_value() != first.context.has_value() ||
            (input.context && input.context->fields != first.context->fields)) {
            throw std::invalid_argument("Los entornos deben compartir esquema, moneda y partición");
        }
        if (validated_tapes.insert(input.tape.get()).second) {
            input.tape->validate();
        }
        if (input.context) {
            input.context->validate(*input.tape);
            account_bytes(input.context->values.size(), sizeof(ContextValue), memory_budget,
                          payload_bytes_);
        }
        // Incluye el estado confirmado, preparación y sustitución transaccional de un reset.
        constexpr std::size_t state_copies = 4;
        account_bytes(input.tape->assets.size(),
                      state_copies * (sizeof(mt_position_v1) + sizeof(mt_trade_v1) +
                                      sizeof(std::size_t) + sizeof(double) + sizeof(uint32_t) + 1),
                      memory_budget, payload_bytes_);
        account_bytes(input.tape->close_times.size(),
                      2 * sizeof(std::vector<std::size_t>), memory_budget, payload_bytes_);
        account_bytes(input.tape->actions.size(),
                      state_copies * (sizeof(Receivable) + sizeof(std::size_t) + 1),
                      memory_budget, payload_bytes_);
    }
    sessions_.reserve(size());
    for (const auto& input : inputs_) {
        sessions_.push_back(FinancialSession(input.tape, input.parameters,
                                             FinancialSession::ValidatedTape{}));
    }
    observations_.resize(size() * width_);
    active_.assign(size(), 1);
    staged_observations_.resize(observations_.size());
    resize_transition(transition_, size());
    resize_transition(staged_transition_, size());
    for (std::size_t lane = 0; lane < size(); ++lane) {
        observe(lane, sessions_[lane], sessions_[lane].state_,
                std::span<float>(observations_).subspan(lane * width_, width_));
    }
    errors_.resize(workers_);
    if (workers_ > 1) {
        threads_.reserve(workers_);
        for (std::size_t worker = 0; worker < workers_; ++worker) {
            threads_.emplace_back([this, worker](const std::stop_token& stop) {
                worker_loop(stop, worker);
            });
        }
    }
}

FinancialBatch::~FinancialBatch() = default;

void FinancialBatch::observe(std::size_t lane, const FinancialSession& session,
                             const SessionSnapshot& state, std::span<float> destination) const {
    const auto& input = inputs_[lane];
    const auto base_width = input.tape->assets.size() * financial_fields + account_fields;
    session.observe_state(state, destination.first(base_width));
    if (!input.context) {
        return;
    }
    const auto& context = *input.context;
    for (std::size_t field = 0; field < context.fields.size(); ++field) {
        const auto& value = context.values[state.cursor * context.fields.size() + field];
        const auto offset = base_width + field * context_components;
        destination[offset] = value.value;
        destination[offset + 1] = value.present ? 1.0F : 0.0F;
        destination[offset + 2] = value.present
            ? static_cast<float>(static_cast<double>(input.tape->close_times[state.cursor] -
                                                     value.available_at) / microseconds_per_day)
            : 0.0F;
    }
}

void FinancialBatch::prepare_range(std::size_t worker) {
    const auto begin = size() * worker / workers_;
    const auto end = size() * (worker + 1) / workers_;
    for (std::size_t lane = begin; lane < end; ++lane) {
        auto& session = sessions_[lane];
        if (active_[lane] == 0) {
            std::copy_n(observations_.begin() + static_cast<std::ptrdiff_t>(lane * width_), width_,
                        staged_observations_.begin() + static_cast<std::ptrdiff_t>(lane * width_));
            staged_transition_.rewards[lane] = 0;
            staged_transition_.reward_valid[lane] = 0;
            staged_transition_.terminated[lane] = 0;
            staged_transition_.truncated[lane] = 0;
            continue;
        }
        const auto outcome = session.prepare_step(actions_[lane]);
        observe(lane, session, session.staged_,
                std::span<float>(staged_observations_).subspan(lane * width_, width_));
        staged_transition_.rewards[lane] = outcome.reward;
        staged_transition_.reward_valid[lane] = static_cast<uint8_t>(outcome.reward_valid);
        staged_transition_.terminated[lane] = static_cast<uint8_t>(outcome.terminated);
        staged_transition_.truncated[lane] = static_cast<uint8_t>(outcome.truncated);
    }
}

void FinancialBatch::worker_loop(const std::stop_token& stop, std::size_t worker) {
    std::size_t observed_generation = 0;
    while (!stop.stop_requested()) {
        {
            std::unique_lock lock(mutex_);
            if (!work_.wait(lock, stop, [&] { return generation_ != observed_generation; })) {
                return;
            }
            observed_generation = generation_;
        }
        try {
            prepare_range(worker);
        } catch (...) {
            errors_[worker] = std::current_exception();
        }
        {
            const std::lock_guard lock(mutex_);
            ++completed_;
        }
        completion_.notify_one();
    }
}

const BatchTransition& FinancialBatch::step(std::span<const uint8_t> actions) {
    return step_checked(actions, {}, {});
}

const BatchTransition& FinancialBatch::step_active(std::span<const uint8_t> actions,
                                                   std::span<const uint8_t> active) {
    return step_checked(actions, active, {});
}

const BatchTransition& FinancialBatch::step_checked(std::span<const uint8_t> actions,
                                                    std::span<const uint8_t> active,
                                                    const BatchValidator& validate) {
    const StepGuard guard(stepping_);
    if (actions.size() != size() || (!active.empty() && active.size() != size()) ||
        std::any_of(active.begin(), active.end(), [](uint8_t value) { return value > 1; })) {
        throw std::invalid_argument("El lote de acciones no corresponde a los entornos");
    }
    if (active.empty()) {
        std::fill(active_.begin(), active_.end(), 1);
    } else {
        std::copy(active.begin(), active.end(), active_.begin());
    }
    for (std::size_t lane = 0; lane < size(); ++lane) {
        if (active_[lane] != 0 &&
            (actions[lane] >= action_count || sessions_[lane].done())) {
            throw std::invalid_argument("La acción no está admitida o falta reiniciar un entorno finalizado");
        }
    }
    actions_ = actions;
    if (workers_ == 1) {
        prepare_range(0);
    } else {
        {
            const std::lock_guard lock(mutex_);
            if (generation_ == std::numeric_limits<std::size_t>::max()) {
                throw std::overflow_error("El contador del lote no puede avanzar");
            }
            completed_ = 0;
            std::fill(errors_.begin(), errors_.end(), nullptr);
            ++generation_;
        }
        work_.notify_all();
        {
            std::unique_lock lock(mutex_);
            completion_.wait(lock, [&] { return completed_ == workers_; });
        }
        for (const auto& error : errors_) {
            if (error) {
                std::rethrow_exception(error);
            }
        }
    }
    if (validate) {
        validate(staged_observations_, staged_transition_);
    }
    for (std::size_t lane = 0; lane < size(); ++lane) {
        if (active_[lane] != 0) {
            sessions_[lane].commit_step();
        }
    }
    observations_.swap(staged_observations_);
    std::swap(transition_, staged_transition_);
    return transition_;
}

std::span<const float> FinancialBatch::observations() const & noexcept { return observations_; }
std::size_t FinancialBatch::size() const noexcept { return inputs_.size(); }
std::size_t FinancialBatch::observation_width() const noexcept { return width_; }
std::size_t FinancialBatch::workers() const noexcept { return workers_; }
std::size_t FinancialBatch::reserved_payload_bytes() const noexcept { return payload_bytes_; }

BatchSnapshot FinancialBatch::snapshot() const {
    BatchSnapshot result;
    result.sessions.reserve(size());
    result.context_sources.reserve(size());
    for (std::size_t lane = 0; lane < size(); ++lane) {
        result.sessions.push_back(sessions_[lane].snapshot());
        result.context_sources.push_back(context_source(inputs_[lane]));
    }
    return result;
}

FinancialMetrics FinancialBatch::metrics(std::size_t lane) const {
    return sessions_.at(lane).metrics();
}

void FinancialBatch::replace(std::span<const std::size_t> indices,
                             const std::vector<SessionSnapshot>* snapshots) {
    if (stepping_) {
        throw std::logic_error("El lote no admite una operación reentrante");
    }
    std::unordered_set<std::size_t> unique;
    for (const auto lane : indices) {
        if (lane >= size() || !unique.insert(lane).second) {
            throw std::invalid_argument("El reinicio contiene entornos inexistentes o repetidos");
        }
    }
    std::vector<FinancialSession> candidates;
    candidates.reserve(indices.size());
    for (const auto lane : indices) {
        candidates.push_back(FinancialSession(inputs_[lane].tape, inputs_[lane].parameters,
                                               FinancialSession::ValidatedTape{}));
        if (snapshots != nullptr) {
            candidates.back().restore((*snapshots)[lane]);
        }
        observe(lane, candidates.back(), candidates.back().state_,
                std::span<float>(staged_observations_).subspan(lane * width_, width_));
    }
    for (std::size_t index = 0; index < indices.size(); ++index) {
        const auto lane = indices[index];
        sessions_[lane] = std::move(candidates[index]);
        std::copy_n(staged_observations_.begin() + static_cast<std::ptrdiff_t>(lane * width_),
                    width_, observations_.begin() + static_cast<std::ptrdiff_t>(lane * width_));
    }
}

void FinancialBatch::restore(const BatchSnapshot& state) {
    if (state.sessions.size() != size() || state.context_sources.size() != size()) {
        throw std::invalid_argument("La recuperación no corresponde al tamaño del lote");
    }
    std::vector<std::size_t> indices(size());
    std::iota(indices.begin(), indices.end(), 0);
    for (std::size_t lane = 0; lane < size(); ++lane) {
        const auto expected = context_source(inputs_[lane]);
        if (state.context_sources[lane] != expected) {
            throw std::invalid_argument("La recuperación utiliza otro contexto externo");
        }
    }
    replace(indices, &state.sessions);
}

void FinancialBatch::reset(std::span<const std::size_t> indices) { replace(indices, nullptr); }

}
