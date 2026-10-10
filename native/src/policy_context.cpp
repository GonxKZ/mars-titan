#include "mars_titan/policy_context.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/grad_mode.h>
#include <torch/serialize/archive.h>
#include <torch/version.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <iomanip>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace mars_titan::learning {
namespace {
constexpr std::size_t feedback_width = 8;
constexpr std::size_t action_count = 6;
constexpr std::size_t context_extra_width = feedback_width + episodic_memory_width + 4;
constexpr std::size_t financial_fields = 6;
constexpr std::size_t account_fields = 2;
constexpr std::size_t context_components = 3;
constexpr std::size_t maximum_context_bytes = std::size_t{64} * 1024 * 1024;
constexpr std::size_t maximum_observation_width = 32768;
constexpr int64_t archive_version = 1;
using Feedback = std::array<float, feedback_width>;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::invalid_argument(message);
}
template <class Value> const Value& checked_optional(const std::optional<Value>& value) {
    if (!value.has_value())
        throw std::invalid_argument("Falta un componente opcional requerido del contexto");
    return value.value();
}
template <class Value> Value& checked_optional(std::optional<Value>& value) {
    if (!value.has_value())
        throw std::invalid_argument("Falta un componente opcional requerido del contexto");
    return value.value();
}
bool memory_variant(std::string_view variant) {
    return variant == "ppo_episodic" || variant == "ppo_episodic_hmm" ||
           variant == "ppo_recent_aux" || variant == "ppo_replay_aux";
}
bool markov_variant(std::string_view variant) {
    return variant == "ppo_hmm" || variant == "ppo_episodic_hmm" || variant == "ppo_recent_aux" ||
           variant == "ppo_replay_aux";
}
void validate_options(const PpoLearningOptions& options) {
    const std::unordered_set<std::string> variants{
        "ppo",          "double_dqn",       "ppo_window",     "ppo_gru",        "ppo_hmm",
        "ppo_episodic", "ppo_episodic_hmm", "ppo_recent_aux", "ppo_replay_aux", "qr_dqn",
        "qr_dqn_cvar"};
    require(variants.contains(options.variant) && options.environments > 0 &&
                options.environments <= simulation::maximum_environments,
            "La variante o el número de entornos del contexto no están admitidos");
    if (options.enabled && markov_variant(options.variant)) {
        require(options.markov && checked_optional(options.markov).states == 2 &&
                    options.markov_fields.size() == checked_optional(options.markov).dimensions,
                "El contexto HMM necesita dos estados y sus campos de entrada");
        const simulation::MarkovFilter checked(checked_optional(options.markov));
        static_cast<void>(checked);
    }
    const EpisodicMemory scope_check({"scope", "train", options.fold, options.representation, 0}, 0,
                                     1);
    static_cast<void>(scope_check);
}
std::size_t raw_width(const simulation::BatchInput& input) {
    require(input.tape && !input.tape->assets.empty() &&
                input.tape->assets.size() <= simulation::maximum_assets &&
                (!input.context || checked_optional(input.context).fields.size() <=
                                       simulation::maximum_context_fields),
            "El contexto necesita una cinta de mercado con dimensiones acotadas");
    return financial_fields * input.tape->assets.size() + account_fields +
           context_components * (input.context ? checked_optional(input.context).fields.size() : 0);
}
std::string context_identity(const simulation::BatchInput& input) {
    return input.context ? checked_optional(input.context).source_sha256 : std::string{};
}
void validate_input(const simulation::BatchInput& input, const simulation::BatchInput& first,
                    const PpoLearningOptions& options) {
    require(input.tape && first.tape && input.tape->assets == first.tape->assets &&
                input.tape->partition == first.tape->partition &&
                input.tape->currency == first.tape->currency &&
                simulation::same_policy_origin(*input.tape, *first.tape) &&
                input.context.has_value() == first.context.has_value() &&
                (!input.context || checked_optional(input.context).fields == first.context->fields),
            "El contexto mezcla esquemas, particiones o predictores");
    input.tape->validate();
    if (input.context)
        checked_optional(input.context).validate(*input.tape);
    if (options.trading_field) {
        require(input.context && checked_optional(options.trading_field) <
                                     checked_optional(input.context).fields.size(),
                "Falta el campo declarado para la máscara de entrenamiento");
        for (std::size_t cursor = 0; cursor < input.tape->close_times.size(); ++cursor) {
            const auto& value =
                checked_optional(input.context)
                    .values.at(cursor * checked_optional(input.context).fields.size() +
                               checked_optional(options.trading_field));
            require(!value.present || value.value == 0 || value.value == 1,
                    "El campo de entrenamiento solo admite cero, uno o ausencia");
        }
    }
    if (options.enabled && markov_variant(options.variant)) {
        std::unordered_set<std::size_t> seen;
        for (const auto field : options.markov_fields) {
            require(input.context && field < checked_optional(input.context).fields.size() &&
                        seen.insert(field).second,
                    "El filtro no conserva campos presentes y distintos");
        }
    }
}
at::Tensor observations_tensor(std::span<const float> values, std::size_t lanes,
                               std::size_t width) {
    require(values.size() == lanes * width &&
                std::all_of(values.begin(), values.end(),
                            [](float value) { return std::isfinite(value); }),
            "Las observaciones del contexto no conservan dimensión o finitud");
    return at::tensor(at::ArrayRef<float>(values.data(), values.size()), at::kFloat)
        .view({static_cast<int64_t>(lanes), static_cast<int64_t>(width)});
}
void check_tensor(const at::Tensor& tensor, at::ScalarType dtype, at::IntArrayRef shape) {
    require(tensor.defined() && tensor.device().is_cpu() && tensor.layout() == at::kStrided &&
                tensor.scalar_type() == dtype && tensor.sizes() == shape &&
                tensor.is_contiguous() && !tensor.requires_grad(),
            "El archivo de contexto no conserva un tensor CPU del esquema");
    if (tensor.is_floating_point())
        require(at::isfinite(tensor).all().item<bool>(),
                "El contexto guardado contiene NaN o infinito");
}
MemoryVector projected_vector(const at::Tensor& projected, std::size_t lane, bool key) {
    MemoryVector result{};
    const auto row = projected.select(0, static_cast<int64_t>(lane));
    std::copy_n(row.const_data_ptr<float>(), episodic_memory_width, result.begin());
    if (key) {
        result[0] += 1;
        if (std::all_of(result.begin(), result.end(), [](float value) { return value == 0; }))
            result[0] = 1;
    }
    return result;
}
void save_filter(torch::serialize::OutputArchive& archive,
                 const simulation::MarkovSnapshot& state) {
    archive.write("log_probabilities", at::tensor(state.log_probabilities, at::kDouble), true);
    archive.write("cursor", c10::IValue(static_cast<int64_t>(state.cursor)));
    archive.write("last_decision_at", c10::IValue(state.last_decision_at.value_or(-1)));
    archive.write("log_likelihood", c10::IValue(state.log_likelihood));
    archive.write("change_probability", c10::IValue(state.change_probability));
}
c10::IValue read_value(torch::serialize::InputArchive& archive, const char* name) {
    c10::IValue value;
    archive.read(name, value);
    return value;
}
int64_t read_integer(torch::serialize::InputArchive& archive, const char* name) {
    const auto value = read_value(archive, name);
    require(value.isInt(), "El contexto guardado necesita un entero");
    return value.toInt();
}
std::string read_text(torch::serialize::InputArchive& archive, const char* name) {
    const auto value = read_value(archive, name);
    require(value.isString(), "El contexto guardado necesita una identidad de texto");
    return value.toStringRef();
}
double read_number(torch::serialize::InputArchive& archive, const char* name) {
    const auto value = read_value(archive, name);
    require(value.isDouble() && std::isfinite(value.toDouble()),
            "El filtro guardado necesita un número finito");
    return value.toDouble();
}
} // namespace

bool value_variant(std::string_view variant) noexcept {
    return variant == "double_dqn" || variant == "qr_dqn" || variant == "qr_dqn_cvar";
}

struct PolicyContext::Impl {
    struct Lane {
        std::size_t cursor = 0;
        uint64_t nonce = 0;
        bool done = false;
        Feedback feedback{};
        std::optional<simulation::MarkovFilter> filter;
    };
    struct Stage {
        std::vector<Lane> lanes;
        std::vector<MemoryQuery> retrieved;
        std::vector<std::optional<PreparedMemoryWrite>> writes;
        at::Tensor raw;
        at::Tensor projected;
        at::Tensor history;
        at::Tensor observations;
    };

    struct Shape {
        std::size_t width;
        std::size_t window;
        std::size_t frame_width;
    };
    static Shape validate_shape(const std::vector<simulation::BatchInput>& active,
                                const PpoLearningOptions& options) {
        validate_options(options);
        require(!active.empty() && active.size() <= options.environments,
                "El contexto supera el número de lanes admitidas");
        const auto width = raw_width(active.front());
        const auto window = options.enabled && options.variant == "ppo_window" ? policy_window : 1;
        const auto frame_width = width + (options.enabled ? context_extra_width : 0);
        require(frame_width <= maximum_observation_width / window &&
                    active.size() <= maximum_context_bytes /
                                         (frame_width * window * sizeof(float) * 3 +
                                          ((options.enabled && memory_variant(options.variant))
                                               ? maximum_episodic_archive_bytes
                                               : 1)),
                "El contexto supera su presupuesto de dimensiones o memoria");
        std::size_t input_bytes = 0;
        for (const auto& input : active) {
            validate_input(input, active.front(), options);
            if (input.context) {
                const auto bytes = checked_optional(input.context).values.size() *
                                   sizeof(simulation::ContextValue);
                require(bytes <= maximum_context_bytes - input_bytes,
                        "Las cintas de contexto superan 64 MiB");
                input_bytes += bytes;
            }
        }
        return {width, window, frame_width};
    }
    Impl(const std::vector<simulation::BatchInput>& active,
         const PpoLearningOptions& initial_options, uint64_t initial_seed,
         std::span<const float> observations)
        : Impl(active, initial_options, initial_seed, observations,
               validate_shape(active, initial_options)) {}
    Impl(const std::vector<simulation::BatchInput>& active,
         const PpoLearningOptions& initial_options, uint64_t initial_seed,
         std::span<const float> observations, Shape shape)
        : options(initial_options), seed(initial_seed), width(shape.width), window(shape.window),
          frame_width(shape.frame_width), inputs(active) {
        current.raw = observations_tensor(observations, active.size(), width);
        current.lanes.resize(active.size());
        current.retrieved.resize(active.size());
        banks.resize(active.size());
        const auto base_width = width + feedback_width;
        auto generator = at::detail::createCPUGenerator(policy_projection_seed);
        projection = at::empty(
            {static_cast<int64_t>(episodic_memory_width), static_cast<int64_t>(base_width)},
            at::kFloat);
        projection.normal_(0, 1, generator);
        projection /= std::sqrt(static_cast<double>(base_width));
        current.history =
            at::zeros({static_cast<int64_t>(active.size()), static_cast<int64_t>(window),
                       static_cast<int64_t>(frame_width)},
                      at::kFloat);
        for (std::size_t lane = 0; lane < active.size(); ++lane)
            initialize_lane(lane, current.lanes.at(lane));
        current.projected = project(current.raw, current.lanes);
        for (std::size_t lane = 0; lane < active.size(); ++lane) {
            current.history[static_cast<int64_t>(lane)][static_cast<int64_t>(window - 1)].copy_(
                frame(current, lane));
        }
        current.observations = current.history.flatten(1);
    }
    [[nodiscard]] bool uses_memory() const {
        return options.enabled && memory_variant(options.variant);
    }
    [[nodiscard]] bool uses_markov() const {
        return options.enabled && markov_variant(options.variant);
    }
    [[nodiscard]] MemoryScope memory_scope(std::size_t lane, uint64_t nonce) const {
        return {inputs.at(lane).tape->source_sha256 + ":" + std::to_string(nonce),
                inputs.at(lane).tape->partition, options.fold, options.representation,
                static_cast<uint64_t>(lane)};
    }
    void filter_step(std::size_t lane, Lane& state) const {
        if (!state.filter)
            return;
        filter_source(inputs.at(lane), state);
    }
    void filter_source(const simulation::BatchInput& source, Lane& state) const {
        if (!state.filter)
            return;
        std::vector<double> values;
        std::vector<uint8_t> masks;
        std::vector<int64_t> dates;
        for (const auto field : options.markov_fields) {
            const auto& value =
                checked_optional(source.context)
                    .values.at(state.cursor * checked_optional(source.context).fields.size() +
                               field);
            values.push_back(static_cast<double>(value.value));
            masks.push_back(value.present ? 1 : 0);
            dates.push_back(value.available_at);
        }
        checked_optional(state.filter)
            .step({values, masks, dates, source.tape->close_times.at(state.cursor)});
    }
    void initialize_lane(std::size_t lane, Lane& state) {
        if (uses_memory())
            banks.at(lane) =
                std::make_unique<EpisodicMemory>(memory_scope(lane, state.nonce), seed);
        if (uses_markov()) {
            state.filter.emplace(checked_optional(options.markov));
            filter_step(lane, state);
        }
    }
    [[nodiscard]] at::Tensor project(const at::Tensor& raw, const std::vector<Lane>& lanes) const {
        auto feedback = at::empty(
            {static_cast<int64_t>(lanes.size()), static_cast<int64_t>(feedback_width)}, at::kFloat);
        auto destination = feedback.accessor<float, 2>();
        for (std::size_t lane = 0; lane < lanes.size(); ++lane) {
            for (std::size_t field = 0; field < feedback_width; ++field)
                destination[static_cast<int64_t>(lane)][static_cast<int64_t>(field)] =
                    lanes.at(lane).feedback.at(field);
        }
        const auto base = at::cat({raw, feedback}, 1).tanh();
        const auto result = at::linear(base, projection).tanh();
        require(at::isfinite(result).all().item<bool>(),
                "La proyección fija contiene valores no finitos");
        return result;
    }
    [[nodiscard]] at::Tensor frame(const Stage& state, std::size_t lane) const {
        auto result = at::zeros({static_cast<int64_t>(frame_width)}, at::kFloat);
        result.narrow(0, 0, static_cast<int64_t>(width))
            .copy_(state.raw.select(0, static_cast<int64_t>(lane)));
        if (!options.enabled)
            return result;
        auto values = result.accessor<float, 1>();
        for (std::size_t field = 0; field < feedback_width; ++field)
            values[static_cast<int64_t>(width + field)] = state.lanes.at(lane).feedback.at(field);
        const auto& recalled = state.retrieved.at(lane);
        const auto memory_start = width + feedback_width;
        if (recalled.count != 0) {
            for (std::size_t component = 0; component < episodic_memory_width; ++component) {
                double mean = 0;
                for (std::size_t item = 0; item < recalled.count; ++item)
                    mean += static_cast<double>(
                                recalled.neighbors.at(item).record.value.at(component)) /
                            static_cast<double>(recalled.count);
                values[static_cast<int64_t>(memory_start + component)] = static_cast<float>(mean);
            }
            values[static_cast<int64_t>(memory_start + episodic_memory_width)] = 1;
            values[static_cast<int64_t>(memory_start + episodic_memory_width + 1)] =
                static_cast<float>(recalled.neighbors[0].similarity);
        }
        if (state.lanes.at(lane).filter) {
            const auto probabilities =
                checked_optional(state.lanes.at(lane).filter).probabilities();
            values[static_cast<int64_t>(frame_width - 2)] = static_cast<float>(probabilities[0]);
            values[static_cast<int64_t>(frame_width - 1)] = static_cast<float>(probabilities[1]);
        }
        return result;
    }
    [[nodiscard]] std::string identity() const {
        std::ostringstream result;
        result.imbue(std::locale::classic());
        result << std::setprecision(std::numeric_limits<double>::max_digits10) << options.enabled
               << '\n'
               << options.variant << '\n'
               << options.fold << '\n'
               << options.representation << '\n'
               << seed << '\n'
               << policy_projection_seed << '\n'
               << TORCH_VERSION << '\n'
               << width << '\n'
               << inputs.size() << '\n'
               << options.environments << '\n'
               << (options.trading_field ? std::to_string(checked_optional(options.trading_field))
                                         : "none")
               << '\n';
        for (const auto field : options.markov_fields)
            result << field << ',';
        result << '\n';
        if (options.markov) {
            result << checked_optional(options.markov).states << ','
                   << checked_optional(options.markov).dimensions << '\n';
            for (const auto* parameters : {&checked_optional(options.markov).prior,
                                           &checked_optional(options.markov).transitions,
                                           &checked_optional(options.markov).means,
                                           &checked_optional(options.markov).variances}) {
                for (const auto value : *parameters)
                    result << value << ',';
                result << '\n';
            }
        }
        return result.str();
    }
    PpoLearningOptions options;
    uint64_t seed;
    std::size_t width = 0;
    std::size_t window = 1;
    std::size_t frame_width = 0;
    std::vector<simulation::BatchInput> inputs;
    at::Tensor projection;
    std::vector<std::unique_ptr<EpisodicMemory>> banks;
    Stage current;
    std::unique_ptr<Stage> staged;
};

PolicyContext::PolicyContext(const std::vector<simulation::BatchInput>& active,
                             const PpoLearningOptions& options, uint64_t seed,
                             std::span<const float> observations) {
    const at::NoGradGuard no_grad;
    impl_ = std::make_unique<Impl>(active, options, seed, observations);
}
PolicyContext::~PolicyContext() = default;
PolicyContext::PolicyContext(PolicyContext&&) noexcept = default;
PolicyContext& PolicyContext::operator=(PolicyContext&&) noexcept = default;
const at::Tensor& PolicyContext::observations() const { return impl_->current.observations; }
at::Tensor PolicyContext::without_retrieved_memory() const {
    const at::NoGradGuard no_grad;
    auto result = impl_->current.observations.clone();
    if (impl_->options.enabled) {
        result
            .view({static_cast<int64_t>(impl_->inputs.size()), static_cast<int64_t>(impl_->window),
                   static_cast<int64_t>(impl_->frame_width)})
            .narrow(2, static_cast<int64_t>(impl_->width + feedback_width),
                    static_cast<int64_t>(episodic_memory_width + 2))
            .zero_();
    }
    return result;
}
const std::vector<MemoryQuery>& PolicyContext::retrieved() const {
    return impl_->current.retrieved;
}
std::vector<uint8_t> PolicyContext::training_mask() const {
    std::vector<uint8_t> result(impl_->inputs.size(), 1);
    for (std::size_t lane = 0; lane < result.size(); ++lane) {
        const auto& state = impl_->current.lanes.at(lane);
        if (state.done)
            result.at(lane) = 0;
        if (impl_->options.trading_field) {
            const auto& context = checked_optional(impl_->inputs.at(lane).context);
            const auto& value = context.values.at(state.cursor * context.fields.size() +
                                                  checked_optional(impl_->options.trading_field));
            if (!value.present || value.value == 0)
                result.at(lane) = 0;
        }
    }
    return result;
}
const at::Tensor& PolicyContext::prepare(std::span<const float> next_observations,
                                         std::span<const uint8_t> actions,
                                         const simulation::BatchTransition& transition,
                                         std::span<const uint8_t> active) {
    require(!impl_->staged, "Ya existe una transición de contexto preparada");
    const auto count = impl_->inputs.size();
    require(actions.size() == count && transition.rewards.size() == count &&
                transition.reward_valid.size() == count && transition.terminated.size() == count &&
                transition.truncated.size() == count &&
                (active.empty() || active.size() == count) &&
                std::all_of(active.begin(), active.end(), [](uint8_t value) { return value <= 1; }),
            "La transición de contexto no conserva las dimensiones del lote");
    const at::NoGradGuard no_grad;
    auto staged = std::make_unique<Impl::Stage>();
    staged->lanes = impl_->current.lanes;
    staged->retrieved = impl_->current.retrieved;
    staged->writes.resize(count);
    staged->raw = observations_tensor(next_observations, count, impl_->width);
    staged->history = impl_->current.history.clone();
    for (std::size_t lane = 0; lane < count; ++lane) {
        if (!active.empty() && active[lane] == 0) {
            staged->raw.select(0, static_cast<int64_t>(lane))
                .copy_(impl_->current.raw.select(0, static_cast<int64_t>(lane)));
            continue;
        }
        auto& state = staged->lanes.at(lane);
        require(!state.done && state.cursor + 1 < impl_->inputs.at(lane).tape->close_times.size() &&
                    actions[lane] < action_count && transition.reward_valid.at(lane) <= 1 &&
                    transition.terminated.at(lane) <= 1 && transition.truncated.at(lane) <= 1 &&
                    !(transition.terminated.at(lane) && transition.truncated.at(lane)) &&
                    std::isfinite(transition.rewards.at(lane)) &&
                    (transition.reward_valid.at(lane) || transition.rewards.at(lane) == 0),
                "La lane no puede avanzar o contiene un resultado inválido");
        const auto& tape = *impl_->inputs.at(lane).tape;
        if (impl_->uses_memory() && transition.reward_valid.at(lane)) {
            MemoryRecord record;
            record.id = state.cursor + 1;
            record.decision_at = tape.close_times.at(state.cursor);
            record.available_at = record.decision_at;
            record.maturity_at = tape.close_times.at(state.cursor + 1);
            record.key = projected_vector(impl_->current.projected, lane, true);
            record.value = projected_vector(impl_->current.projected, lane, false);
            constexpr std::size_t outcome_start = episodic_memory_width - action_count - 1;
            std::fill(std::next(record.value.begin(), static_cast<std::ptrdiff_t>(outcome_start)),
                      record.value.end(), 0);
            record.value.at(outcome_start + actions[lane]) = 1;
            record.value.back() = static_cast<float>(std::tanh(transition.rewards.at(lane)));
            record.reward_valid = true;
            record.reward = transition.rewards.at(lane);
            staged->writes.at(lane).emplace(
                impl_->banks.at(lane)->prepare_write(record, record.maturity_at));
        }
        ++state.cursor;
        state.done = transition.terminated.at(lane) || transition.truncated.at(lane);
        state.feedback.fill(0);
        state.feedback.at(actions[lane]) = 1;
        state.feedback.at(action_count) =
            static_cast<float>(std::tanh(transition.rewards.at(lane)));
        state.feedback.at(action_count + 1) = transition.reward_valid.at(lane) ? 1 : 0;
        impl_->filter_step(lane, state);
    }
    staged->projected = impl_->project(staged->raw, staged->lanes);
    for (std::size_t lane = 0; lane < count; ++lane) {
        if (!active.empty() && active[lane] == 0)
            continue;
        if (impl_->uses_memory()) {
            const auto key = projected_vector(staged->projected, lane, true);
            const auto cutoff =
                impl_->inputs.at(lane).tape->close_times.at(staged->lanes.at(lane).cursor);
            const auto excluded = static_cast<uint64_t>(staged->lanes.at(lane).cursor + 1);
            staged->retrieved.at(lane) =
                staged->writes.at(lane)
                    ? impl_->banks.at(lane)->query_prepared(
                          key, cutoff, checked_optional(staged->writes.at(lane)), excluded)
                    : impl_->banks.at(lane)->query(key, cutoff, excluded);
        }
        auto history = staged->history.select(0, static_cast<int64_t>(lane));
        if (impl_->window > 1)
            history.narrow(0, 0, static_cast<int64_t>(impl_->window - 1))
                .copy_(impl_->current.history.select(0, static_cast<int64_t>(lane))
                           .narrow(0, 1, static_cast<int64_t>(impl_->window - 1)));
        history.select(0, static_cast<int64_t>(impl_->window - 1))
            .copy_(impl_->frame(*staged, lane));
    }
    staged->observations = staged->history.flatten(1);
    impl_->staged = std::move(staged);
    return impl_->staged->observations;
}
void PolicyContext::commit() {
    require(impl_->staged != nullptr, "No existe una transición de contexto preparada");
    for (std::size_t lane = 0; lane < impl_->banks.size(); ++lane) {
        auto& write = impl_->staged->writes.at(lane);
        // Los bancos son privados y no admiten cambios mientras exista este candidato.
        if (write && !impl_->banks.at(lane)->commit(std::move(checked_optional(write))))
            std::terminate();
    }
    static_assert(std::is_nothrow_move_assignable_v<Impl::Stage>);
    impl_->current = std::move(*impl_->staged);
    impl_->current.writes.clear();
    impl_->staged.reset();
}
void PolicyContext::cancel() noexcept { impl_->staged.reset(); }
void PolicyContext::reset(std::size_t lane, const simulation::BatchInput& input,
                          std::span<const float> observation) {
    require(!impl_->staged && lane < impl_->inputs.size() &&
                impl_->current.lanes.at(lane).nonce <
                    static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
            "No se puede reiniciar esa lane durante una transición preparada");
    validate_input(input, impl_->inputs.front(), impl_->options);
    std::size_t context_bytes = 0;
    for (std::size_t index = 0; index < impl_->inputs.size(); ++index) {
        const auto& selected = index == lane ? input : impl_->inputs.at(index);
        if (selected.context) {
            require(selected.context->values.size() <=
                        (maximum_context_bytes - context_bytes) / sizeof(simulation::ContextValue),
                    "El reinicio supera el presupuesto conjunto de contextos");
            context_bytes += selected.context->values.size() * sizeof(simulation::ContextValue);
        }
    }
    const auto replacement = observations_tensor(observation, 1, impl_->width);
    const at::NoGradGuard no_grad;
    auto candidate_input = input;
    auto candidate = std::make_unique<Impl::Stage>();
    candidate->lanes = impl_->current.lanes;
    candidate->retrieved = impl_->current.retrieved;
    auto& state = candidate->lanes.at(lane);
    const auto nonce = state.nonce + 1;
    state = {};
    state.nonce = nonce;
    candidate->retrieved.at(lane) = {};
    std::unique_ptr<EpisodicMemory> bank;
    if (impl_->uses_memory()) {
        MemoryScope scope{input.tape->source_sha256 + ":" + std::to_string(nonce),
                          input.tape->partition, impl_->options.fold, impl_->options.representation,
                          static_cast<uint64_t>(lane)};
        bank = std::make_unique<EpisodicMemory>(std::move(scope), impl_->seed);
    }
    if (impl_->uses_markov()) {
        state.filter.emplace(checked_optional(impl_->options.markov));
        impl_->filter_source(input, state);
    }
    candidate->raw = impl_->current.raw.clone();
    candidate->raw.select(0, static_cast<int64_t>(lane)).copy_(replacement.select(0, 0));
    candidate->projected = impl_->project(candidate->raw, candidate->lanes);
    candidate->history = impl_->current.history.clone();
    auto history = candidate->history.select(0, static_cast<int64_t>(lane));
    history.zero_();
    history.select(0, static_cast<int64_t>(impl_->window - 1))
        .copy_(impl_->frame(*candidate, lane));
    candidate->observations = candidate->history.flatten(1);
    // La preparación anterior puede fallar. Estas sustituciones no reservan memoria.
    static_assert(std::is_nothrow_move_assignable_v<simulation::BatchInput>);
    impl_->inputs.at(lane) = std::move(candidate_input);
    impl_->banks.at(lane) = std::move(bank);
    impl_->current = std::move(*candidate);
}

PolicyContextSnapshot PolicyContext::snapshot() const {
    const at::NoGradGuard no_grad;
    torch::serialize::OutputArchive archive;
    archive.write("schema_version", c10::IValue(archive_version));
    archive.write("identity", c10::IValue(impl_->identity()));
    archive.write("raw", impl_->current.raw.clone(), true);
    archive.write("history", impl_->current.history.clone(), true);
    for (std::size_t lane = 0; lane < impl_->inputs.size(); ++lane) {
        torch::serialize::OutputArchive entry;
        const auto& state = impl_->current.lanes.at(lane);
        entry.write("source", c10::IValue(impl_->inputs.at(lane).tape->source_sha256));
        entry.write("context", c10::IValue(context_identity(impl_->inputs.at(lane))));
        entry.write("cursor", c10::IValue(static_cast<int64_t>(state.cursor)));
        entry.write("nonce", c10::IValue(static_cast<int64_t>(state.nonce)));
        entry.write("done", c10::IValue(state.done));
        entry.write("feedback",
                    at::tensor(at::ArrayRef<float>(state.feedback.data(), state.feedback.size()),
                               at::kFloat),
                    true);
        if (impl_->uses_memory()) {
            const auto bytes = serialize_memory(impl_->banks.at(lane)->snapshot());
            std::vector<uint8_t> buffer(bytes.begin(), bytes.end());
            entry.write("memory", at::tensor(buffer, at::kByte), true);
        }
        if (state.filter) {
            torch::serialize::OutputArchive filter;
            save_filter(filter, checked_optional(state.filter).snapshot());
            entry.write("filter", filter);
        }
        archive.write("lane_" + std::to_string(lane), entry);
    }
    std::ostringstream stream;
    archive.save_to(stream);
    auto bytes = std::move(stream).str();
    require(bytes.size() <= maximum_context_bytes, "El archivo de contexto supera 64 MiB");
    return {std::move(bytes)};
}
void PolicyContext::restore(const PolicyContextSnapshot& saved) {
    require(!impl_->staged && !saved.archive.empty() &&
                saved.archive.size() <= maximum_context_bytes,
            "La recuperación del contexto supera su límite o tiene un candidato pendiente");
    constexpr std::size_t footer_bytes = 22;
    const std::string_view bytes(saved.archive);
    require(bytes.size() >= footer_bytes && bytes.starts_with("PK\003\004") &&
                bytes.substr(bytes.size() - footer_bytes).starts_with("PK\005\006") &&
                bytes[bytes.size() - 1] == '\0' && bytes[bytes.size() - 2] == '\0',
            "El contenedor de contexto está truncado o es incompatible");
    const at::NoGradGuard no_grad;
    std::istringstream stream(saved.archive);
    torch::serialize::InputArchive archive;
    archive.load_from(stream, at::Device(at::kCPU));
    require(read_integer(archive, "schema_version") == archive_version &&
                read_text(archive, "identity") == impl_->identity(),
            "La recuperación pertenece a otro contrato de contexto");
    at::Tensor raw;
    archive.read("raw", raw, true);
    check_tensor(raw, at::kFloat,
                 {static_cast<int64_t>(impl_->inputs.size()), static_cast<int64_t>(impl_->width)});
    auto candidate = std::make_unique<Impl>(
        impl_->inputs, impl_->options, impl_->seed,
        std::span<const float>(raw.const_data_ptr<float>(), static_cast<std::size_t>(raw.numel())));
    archive.read("history", candidate->current.history, true);
    check_tensor(candidate->current.history, at::kFloat,
                 {static_cast<int64_t>(impl_->inputs.size()), static_cast<int64_t>(impl_->window),
                  static_cast<int64_t>(impl_->frame_width)});
    for (std::size_t lane = 0; lane < candidate->inputs.size(); ++lane) {
        torch::serialize::InputArchive entry;
        archive.read("lane_" + std::to_string(lane), entry);
        require(read_text(entry, "source") == candidate->inputs.at(lane).tape->source_sha256 &&
                    read_text(entry, "context") == context_identity(candidate->inputs.at(lane)),
                "La lane guardada pertenece a otra cinta o contexto");
        const auto cursor = read_integer(entry, "cursor");
        const auto nonce = read_integer(entry, "nonce");
        const auto done = read_value(entry, "done");
        require(cursor >= 0 &&
                    static_cast<uint64_t>(cursor) <
                        candidate->inputs.at(lane).tape->close_times.size() &&
                    nonce >= 0 && done.isBool() && (cursor != 0 || !done.toBool()),
                "El cursor o el nonce del contexto no están admitidos");
        auto& state = candidate->current.lanes.at(lane);
        state.cursor = static_cast<std::size_t>(cursor);
        state.nonce = static_cast<uint64_t>(nonce);
        state.done = done.toBool();
        at::Tensor feedback;
        entry.read("feedback", feedback, true);
        check_tensor(feedback, at::kFloat, {static_cast<int64_t>(feedback_width)});
        std::copy_n(feedback.const_data_ptr<float>(), feedback_width, state.feedback.begin());
        double chosen = 0;
        for (std::size_t action = 0; action < action_count; ++action) {
            require(state.feedback.at(action) == 0 || state.feedback.at(action) == 1,
                    "El feedback no conserva una acción discreta");
            chosen += static_cast<double>(state.feedback.at(action));
        }
        require(
            chosen == (cursor == 0 ? 0 : 1) && std::abs(state.feedback.at(action_count)) <= 1 &&
                (state.feedback.at(action_count + 1) == 0 ||
                 state.feedback.at(action_count + 1) == 1) &&
                (state.feedback.at(action_count + 1) != 0 || state.feedback.at(action_count) == 0),
            "El feedback guardado no conserva su máscara y recompensa");
        if (candidate->uses_memory()) {
            at::Tensor memory;
            entry.read("memory", memory, true);
            require(memory.defined() && memory.device().is_cpu() &&
                        memory.scalar_type() == at::kByte && memory.dim() == 1 &&
                        memory.is_contiguous() && memory.numel() > 0 &&
                        static_cast<uint64_t>(memory.numel()) <= maximum_episodic_archive_bytes,
                    "El banco guardado no conserva un archivo binario acotado");
            const auto buffer = std::span(memory.const_data_ptr<uint8_t>(),
                                          static_cast<std::size_t>(memory.numel()));
            const std::string serialized(buffer.begin(), buffer.end());
            const auto stored = deserialize_memory(serialized);
            candidate->banks.at(lane) = std::make_unique<EpisodicMemory>(
                candidate->memory_scope(lane, state.nonce), candidate->seed);
            candidate->banks.at(lane)->restore(stored);
            require(stored.last_id <= static_cast<uint64_t>(cursor) &&
                        stored.confirmed_at <=
                            candidate->inputs.at(lane).tape->close_times.at(state.cursor),
                    "La memoria episódica adelanta al cursor del contexto");
        }
        if (candidate->uses_markov()) {
            torch::serialize::InputArchive filter;
            entry.read("filter", filter);
            auto model = checked_optional(state.filter).snapshot();
            at::Tensor probabilities;
            filter.read("log_probabilities", probabilities, true);
            require(probabilities.defined() && probabilities.device().is_cpu() &&
                        probabilities.scalar_type() == at::kDouble &&
                        probabilities.sizes() == at::IntArrayRef({2}) &&
                        probabilities.is_contiguous(),
                    "El filtro no conserva sus dos probabilidades logarítmicas");
            const auto log_values =
                std::span(probabilities.const_data_ptr<double>(), std::size_t{2});
            model.log_probabilities.assign(log_values.begin(), log_values.end());
            const auto filter_cursor = read_integer(filter, "cursor");
            require(filter_cursor == cursor + 1, "El filtro no conserva el cursor de la lane");
            model.cursor = static_cast<std::size_t>(filter_cursor);
            model.last_decision_at = read_integer(filter, "last_decision_at");
            require(model.last_decision_at ==
                        candidate->inputs.at(lane).tape->close_times.at(state.cursor),
                    "El filtro no conserva la fecha del contexto");
            model.log_likelihood = read_number(filter, "log_likelihood");
            model.change_probability = read_number(filter, "change_probability");
            checked_optional(state.filter).restore(model);
        }
    }
    candidate->current.projected =
        candidate->project(candidate->current.raw, candidate->current.lanes);
    for (std::size_t lane = 0; lane < candidate->inputs.size(); ++lane) {
        const auto cursor = candidate->current.lanes.at(lane).cursor;
        if (candidate->uses_memory())
            candidate->current.retrieved.at(lane) = candidate->banks.at(lane)->query(
                projected_vector(candidate->current.projected, lane, true),
                candidate->inputs.at(lane).tape->close_times.at(cursor),
                static_cast<uint64_t>(cursor + 1));
        require(at::equal(candidate->frame(candidate->current, lane),
                          candidate->current.history[static_cast<int64_t>(lane)]
                                                    [static_cast<int64_t>(candidate->window - 1)]),
                "La última observación no concilia con memoria, filtro y feedback");
        if (cursor + 1 < candidate->window)
            require(
                at::count_nonzero(candidate->current.history[static_cast<int64_t>(lane)].narrow(
                                      0, 0, static_cast<int64_t>(candidate->window - cursor - 1)))
                        .item<int64_t>() == 0,
                "La ventana guardada inventa observaciones anteriores al episodio");
    }
    candidate->current.observations = candidate->current.history.flatten(1);
    impl_.swap(candidate);
}
std::size_t PolicyContext::observation_width() const noexcept {
    return impl_->window * impl_->frame_width;
}
std::size_t PolicyContext::cursor(std::size_t lane) const {
    return impl_->current.lanes.at(lane).cursor;
}
MemoryScope PolicyContext::memory_scope(std::size_t lane) const {
    return impl_->memory_scope(lane, impl_->current.lanes.at(lane).nonce);
}
uint64_t PolicyContext::episode(std::size_t lane) const {
    return impl_->current.lanes.at(lane).nonce;
}
} // namespace mars_titan::learning
