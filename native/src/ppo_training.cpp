#include "mars_titan/ppo_training.hpp"
#include "mars_titan/ppo_objectives.hpp"
#include "mars_titan/quantile_dqn.hpp"
#include "accurate_sum.hpp"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <torch/serialize/archive.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace mars_titan::learning {
namespace {
constexpr std::size_t rollout_field_count = 9;
constexpr std::size_t maximum_transitions = std::size_t{1} << 20;
constexpr std::size_t maximum_rollout = 16384;
constexpr std::size_t diagnostic_transitions = 32;
constexpr std::size_t maximum_archive_bytes = std::size_t{128} * 1024 * 1024;
constexpr std::size_t maximum_rollout_bytes = std::size_t{512} * 1024 * 1024;
constexpr std::size_t packed_fields = 3;
constexpr std::size_t digest_characters = 64;
constexpr std::size_t recurrent_width = 64;
constexpr uint64_t replay_seed_offset = 0xc6a4a7935bd1e995ULL;

struct RolloutShape {
    std::size_t ticks;
    std::size_t environments;
    std::size_t width;
    bool recurrent = false;
    bool full_distribution = false;
};

template<class Rollout> auto fields(Rollout& r) {
    using Pointer = decltype(&r.observations);
    return std::array<std::pair<const char*, Pointer>, rollout_field_count>{{
        {"observations", &r.observations}, {"actions", &r.actions},
        {"old_log_probabilities", &r.old_log_probabilities}, {"old_values", &r.old_values},
        {"rewards", &r.rewards}, {"next_values", &r.next_values},
        {"reward_valid", &r.reward_valid}, {"terminated", &r.terminated},
        {"truncated", &r.truncated}}};
}

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::invalid_argument(message);
    }
}

void validate_sources(const std::vector<simulation::BatchInput>& inputs, std::string_view partition) {
    require(!inputs.empty() && inputs.size() <= simulation::maximum_environments && inputs.front().tape,
            "El entrenamiento necesita fuentes acotadas");
    const auto& first = *inputs.front().tape;
    for (const auto& input : inputs) {
        require(input.tape && input.tape->partition == partition &&
                    simulation::same_policy_origin(*input.tape, first),
                "Las fuentes mezclan particiones, dominios, predictores padre o bases históricas");
    }
}

void validate_config(const PpoTrainingConfig& config, RolloutShape shape,
                     const std::string& device, bool diagnostic) {
    const auto environments = shape.environments;
    require(environments > 0 && environments <= simulation::maximum_environments &&
                shape.width > 0 && shape.width <= maximum_rollout_bytes / sizeof(float) &&
                config.total_transitions >= environments &&
                config.total_transitions <= maximum_transitions &&
                config.total_transitions % environments == 0 &&
                config.rollout_transitions >= environments && config.rollout_transitions <= maximum_rollout &&
                config.rollout_transitions <= config.total_transitions &&
                config.rollout_transitions % environments == 0 && config.rollout_bytes > 0 &&
                config.rollout_bytes <= maximum_rollout_bytes &&
                (device == "cuda:0" || device == "cpu") &&
                (device != "cpu" || (diagnostic && config.total_transitions <= diagnostic_transitions)) &&
                (!diagnostic || device == "cpu"),
            "El presupuesto, dispositivo o diagnóstico PPO no está admitido");
    constexpr std::size_t scalar_bytes = sizeof(int64_t) + 4 * sizeof(double) + 3 * sizeof(bool);
    // GRU puede conservar el rollback del reset a la vez que la copia de actualización.
    const auto rollout_copies = shape.recurrent ? std::size_t{3} : std::size_t{2};
    const auto distribution_bytes = shape.full_distribution
        ? static_cast<std::size_t>(ppo_action_count) * (rollout_copies * sizeof(float) + 16 * sizeof(double)) : 0;
    const auto row_bytes = rollout_copies * (shape.width * sizeof(float) + scalar_bytes) + distribution_bytes;
    require(config.rollout_transitions <= config.rollout_bytes / row_bytes,
            "El recorrido PPO supera su presupuesto de memoria");
    if (shape.recurrent) {
        auto remaining = config.rollout_bytes - config.rollout_transitions * row_bytes;
        // Historia y prefijo residentes, estado recuperado y copias candidatas al restaurar.
        constexpr std::size_t history_copies = 6;
        constexpr std::size_t hidden_copies = 4;
        const auto history_rows = environments * static_cast<std::size_t>(ppo_maximum_history);
        const auto history_row_bytes = shape.width * sizeof(float);
        require(history_rows <= remaining / history_copies / history_row_bytes,
                "La historia y el prefijo GRU superan el presupuesto del recorrido");
        remaining -= history_rows * history_row_bytes * history_copies;
        const auto lane_bytes = history_copies * sizeof(int64_t) +
                                hidden_copies * recurrent_width * sizeof(float);
        require(environments <= remaining / lane_bytes,
                "Los estados y longitudes GRU superan el presupuesto del recorrido");
        remaining -= environments * lane_bytes;
        require(config.rollout_transitions <= remaining / (rollout_copies * sizeof(bool)),
                "Las máscaras de reinicio GRU superan el presupuesto del recorrido");
    }
}

PpoRollout allocate_rollout(RolloutShape shape) {
    const auto time = static_cast<int64_t>(shape.ticks);
    const auto lanes = static_cast<int64_t>(shape.environments);
    PpoRollout result;
    result.observations = at::zeros({time, lanes, static_cast<int64_t>(shape.width)}, at::kFloat);
    result.actions = at::zeros({time, lanes}, at::kLong);
    result.old_log_probabilities = at::zeros({time, lanes}, at::kDouble);
    result.old_values = at::zeros({time, lanes}, at::kDouble);
    result.rewards = at::zeros({time, lanes}, at::kDouble);
    result.next_values = at::zeros({time, lanes}, at::kDouble);
    result.reward_valid = at::zeros({time, lanes}, at::kBool);
    result.terminated = at::zeros({time, lanes}, at::kBool);
    result.truncated = at::zeros({time, lanes}, at::kBool);
    if (shape.full_distribution) { result.old_action_weights = at::zeros({time, lanes, ppo_action_count}, at::kFloat); }
    return result;
}

PpoRollout rollout_prefix(const PpoRollout& source, std::size_t ticks, bool clone) {
    PpoRollout result;
    const auto origin = fields(source);
    auto destination = fields(result);
    for (std::size_t index = 0; index < origin.size(); ++index) {
        auto value = origin.at(index).second->narrow(0, 0, static_cast<int64_t>(ticks));
        *destination.at(index).second = clone ? value.detach().clone() : value;
    }
    if (source.episode_starts.defined()) {
        const auto starts = source.episode_starts.narrow(0, 0, static_cast<int64_t>(ticks));
        result.episode_starts = clone ? starts.clone() : starts;
        result.prefix_observations = clone ? source.prefix_observations.clone() : source.prefix_observations;
        result.prefix_lengths = clone ? source.prefix_lengths.clone() : source.prefix_lengths;
    }
    if (source.old_action_weights.defined()) {
        const auto weights = source.old_action_weights.narrow(0, 0, static_cast<int64_t>(ticks));
        result.old_action_weights = clone ? weights.detach().clone() : weights;
    }
    return result;
}

PpoRollout move_rollout(PpoRollout source, const at::Device& device) {
    for (auto [name, tensor] : fields(source)) {
        static_cast<void>(name);
        *tensor = tensor->to(device);
    }
    if (source.episode_starts.defined()) {
        source.episode_starts = source.episode_starts.to(device);
        source.prefix_observations = source.prefix_observations.to(device);
        source.prefix_lengths = source.prefix_lengths.to(device);
    }
    if (source.old_action_weights.defined()) { source.old_action_weights = source.old_action_weights.to(device); }
    return source;
}

at::Tensor observation_tensor(std::span<const float> observation, std::size_t lanes,
                             std::size_t width) {
    auto result = at::empty({static_cast<int64_t>(lanes), static_cast<int64_t>(width)}, at::kFloat);
    std::copy(observation.begin(), observation.end(), result.data_ptr<float>());
    return result;
}

template<class Scalar> std::span<Scalar> row(at::Tensor& tensor, std::size_t tick) {
    auto view = tensor.select(0, static_cast<int64_t>(tick));
    return {view.data_ptr<Scalar>(), static_cast<std::size_t>(view.numel())};
}

void validate_partial(const PpoRollout& rollout, RolloutShape dimensions) {
    const auto expected = allocate_rollout({0, dimensions.environments, dimensions.width});
    const auto schema = fields(expected);
    const auto values = fields(rollout);
    for (std::size_t index = 0; index < values.size(); ++index) {
        const auto& value = *values.at(index).second;
        auto shape = schema.at(index).second->sizes().vec();
        shape.front() = static_cast<int64_t>(dimensions.ticks);
        require(value.defined() && value.device().is_cpu() && value.layout() == at::kStrided &&
                    value.sizes() == shape && value.scalar_type() == schema.at(index).second->scalar_type(),
                "El recorrido guardado no conserva forma, dispositivo o precisión");
        if (value.is_floating_point()) {
            require(at::isfinite(value).all().item<bool>(), "El recorrido guardado contiene valores no finitos");
        }
    }
    require(((rollout.actions >= 0) & (rollout.actions < ppo_action_count)).all().item<bool>(),
            "El recorrido guardado contiene acciones no admitidas");
    require(rollout.old_action_weights.defined() == dimensions.full_distribution,
            "El recorrido parcial no corresponde al objetivo PPO");
    if (dimensions.full_distribution) {
        const auto& weights = rollout.old_action_weights;
        require(weights.device().is_cpu() && weights.layout() == at::kStrided &&
                    weights.scalar_type() == at::kFloat && weights.sizes() ==
                        at::IntArrayRef({static_cast<int64_t>(dimensions.ticks),
                                        static_cast<int64_t>(dimensions.environments), ppo_action_count}),
                "Los pesos históricos guardados no tienen forma o tipo válidos");
        if (dimensions.ticks > 0) {
            validate_ppo_behavior(weights.flatten(0, 1), rollout.actions.flatten(),
                                  rollout.old_log_probabilities.flatten(), rollout.reward_valid.flatten());
        }
    }
}

PpoArchitecture architecture_for(const PpoLearningOptions& learning, std::size_t width) {
    PpoArchitecture result;
    if (learning.enabled && learning.variant == "ppo_gru") {
        result.kind = PpoNetworkKind::gru;
    } else if (learning.enabled && learning.variant == "ppo_window") {
        const auto one_step = width / policy_window;
        const auto gru_parameters = 3 * recurrent_width * (one_step + recurrent_width + 2) +
                                    (recurrent_width + 1) * static_cast<std::size_t>(ppo_action_count + 1);
        std::size_t best = 0;
        auto difference = std::numeric_limits<std::size_t>::max();
        constexpr std::size_t maximum_hidden = 512;
        for (std::size_t hidden = 1; hidden <= maximum_hidden; ++hidden) {
            const auto count = (width + 1) * hidden + (hidden + 1) * hidden +
                               (hidden + 1) * static_cast<std::size_t>(ppo_action_count + 1);
            const auto delta = count > gru_parameters ? count - gru_parameters : gru_parameters - count;
            if (delta < difference) {
                difference = delta;
                best = hidden;
            }
        }
        constexpr double maximum_parameter_gap = 0.05;
        require(static_cast<double>(difference) <= maximum_parameter_gap * static_cast<double>(gru_parameters),
                "No hay una ventana con capacidad comparable dentro del límite declarado");
        result.hidden_width = static_cast<int64_t>(best);
    }
    result.auxiliary = learning.enabled &&
        (learning.variant == "ppo_recent_aux" || learning.variant == "ppo_replay_aux");
    result.double_dqn = learning.enabled && value_variant(learning.variant);
    if (result.double_dqn && learning.variant != "double_dqn") {
        result.quantiles = qr_dqn_quantiles;
        result.risk_alpha = learning.variant == "qr_dqn_cvar" ? qr_dqn_cvar_alpha : 1;
    }
    return result;
}

at::Tensor bytes_tensor(std::string_view bytes) {
    auto result = at::empty({static_cast<int64_t>(bytes.size())}, at::kByte);
    if (!bytes.empty()) {
        std::memcpy(result.data_ptr<uint8_t>(), bytes.data(), bytes.size());
    }
    return result;
}

std::string tensor_bytes(const at::Tensor& tensor) {
    require(tensor.device().is_cpu() && tensor.scalar_type() == at::kByte && tensor.dim() == 1 &&
                tensor.is_contiguous() && tensor.numel() <= static_cast<int64_t>(maximum_archive_bytes),
            "El estado binario adaptativo no conserva su tipo y presupuesto");
    std::string result(static_cast<std::size_t>(tensor.numel()), '\0');
    if (!result.empty()) {
        std::memcpy(result.data(), tensor.const_data_ptr<uint8_t>(), result.size());
    }
    return result;
}

struct AdaptiveState {
    at::Tensor hidden;
    at::Tensor history;
    at::Tensor lengths;
    std::string context;
    std::size_t ticks = 0;
    std::string replay = {};
    std::size_t auxiliary_samples = 0;
};

std::unique_ptr<LearningReplay> make_replay(std::size_t width, const PpoArchitecture& architecture,
                                           const PpoLearningOptions& learning, uint64_t seed) {
    if (!architecture.auxiliary && !architecture.double_dqn) {
        return {};
    }
    const bool dqn = architecture.double_dqn;
    const auto mode = learning.variant == "ppo_replay_aux" ? ReplayMode::reservoir : ReplayMode::recent;
    return std::make_unique<LearningReplay>(width, dqn ? dqn_replay_capacity : auxiliary_replay_capacity,
        mode, dqn, seed ^ replay_seed_offset, dqn ? dqn_replay_bytes : auxiliary_replay_bytes);
}

std::string adaptive_archive(const AdaptiveState& state) {
    torch::serialize::OutputArchive archive;
    archive.write("version", at::tensor(int64_t{1}), true);
    archive.write("ticks", at::tensor(static_cast<int64_t>(state.ticks)), true);
    archive.write("hidden", state.hidden.detach().to(at::kCPU), true);
    archive.write("history", state.history, true);
    archive.write("lengths", state.lengths, true);
    archive.write("context", bytes_tensor(state.context), true);
    archive.write("replay", bytes_tensor(state.replay), true);
    archive.write("auxiliary_samples", at::tensor(static_cast<int64_t>(state.auxiliary_samples)), true);
    std::ostringstream output;
    archive.save_to(output);
    auto bytes = std::move(output).str();
    require(bytes.size() <= maximum_archive_bytes, "El estado adaptativo supera su presupuesto");
    return bytes;
}

AdaptiveState read_adaptive(std::string_view bytes) {
    require(!bytes.empty() && bytes.size() <= maximum_archive_bytes, "El estado adaptativo está vacío o excede sus límites");
    std::istringstream input{std::string(bytes)};
    torch::serialize::InputArchive archive;
    archive.load_from(input, at::Device(at::kCPU));
    at::Tensor version, ticks, context, replay, auxiliary_samples;
    AdaptiveState result;
    archive.read("version", version, true);
    archive.read("ticks", ticks, true);
    archive.read("hidden", result.hidden, true);
    archive.read("history", result.history, true);
    archive.read("lengths", result.lengths, true);
    archive.read("context", context, true);
    archive.read("replay", replay, true);
    archive.read("auxiliary_samples", auxiliary_samples, true);
    require(version.scalar_type() == at::kLong && version.numel() == 1 && version.item<int64_t>() == 1 &&
                ticks.scalar_type() == at::kLong && ticks.numel() == 1 && ticks.item<int64_t>() >= 0,
            "El estado adaptativo no conserva versión y cursor");
    result.ticks = static_cast<std::size_t>(ticks.item<int64_t>());
    result.context = tensor_bytes(context);
    result.replay = tensor_bytes(replay);
    require(auxiliary_samples.scalar_type() == at::kLong && auxiliary_samples.numel() == 1 &&
                auxiliary_samples.item<int64_t>() >= 0,
            "La recuperación necesita un contador válido de ejemplos auxiliares");
    result.auxiliary_samples = static_cast<std::size_t>(auxiliary_samples.item<int64_t>());
    return result;
}
}

PpoTrainer::PpoTrainer(std::vector<simulation::BatchInput> inputs, PpoTrainingConfig config,
                       PpoHyperparameters hyperparameters, std::string device, bool diagnostic,
                       PpoLearningOptions learning, PpoObjectiveConfig objective)
    : inputs_(std::move(inputs)), config_(config), hyperparameters_(hyperparameters),
      device_(std::move(device)), diagnostic_(diagnostic), learning_(std::move(learning)), objective_(objective) {
    objective_.validate();
    validate_sources(inputs_, "train");
    hyperparameters_.validate();
    const auto lanes = learning_.enabled ? learning_.environments : inputs_.size();
    require(lanes > 0 && lanes <= simulation::maximum_environments, "El número de entornos no está admitido");
    for (std::size_t lane = 0; lane < lanes; ++lane) {
        source_indices_.push_back(lane % inputs_.size());
        active_inputs_.push_back(inputs_[source_indices_.back()]);
    }
    next_source_ = lanes;
    batch_ = std::make_unique<simulation::FinancialBatch>(active_inputs_, config_.workers);
    auto width = batch_->observation_width();
    validate_config(config_, {config_.rollout_transitions / lanes, lanes, width}, device_, diagnostic_);
    if (learning_.enabled) {
        for (const auto& input : inputs_) {
            require(input.tape->close_times.size() <= static_cast<std::size_t>(ppo_maximum_history),
                    "Un episodio adaptativo supera la historia recurrente acotada");
            input.tape->validate();
            if (learning_.trading_field) {
                if (!input.context) {
                    throw std::invalid_argument("Falta el contexto del calentamiento");
                }
                const auto& context = input.context.value();
                require(*learning_.trading_field < context.fields.size(),
                        "Falta el indicador de calentamiento");
                bool enabled = false;
                for (std::size_t session = 0; session + 1 < input.tape->close_times.size(); ++session) {
                    const auto& value = context.values.at(session * context.fields.size() + *learning_.trading_field);
                    require(value.present && (value.value == 0 || value.value == 1) && (!enabled || value.value == 1),
                            "El calentamiento debe terminar antes de aprender y no reanudarse dentro del episodio");
                    enabled = enabled || value.value == 1;
                }
                require(enabled, "El episodio no contiene decisiones de aprendizaje");
            }
        }
        context_ = std::make_unique<PolicyContext>(active_inputs_, learning_, config_.seed, batch_->observations());
        width = context_->observation_width();
    }
    const RolloutShape shape{config_.rollout_transitions / lanes, lanes, width,
                            learning_.enabled && learning_.variant == "ppo_gru", objective_.enabled()};
    validate_config(config_, shape, device_, diagnostic_);
    buffers_ = allocate_rollout(shape);
    actions_.resize(batch_->size());
    reset_lanes_.reserve(batch_->size());
    policy_ = std::make_unique<PpoPolicy>(width, hyperparameters_, config_.seed, device_,
                                         default_ppo_memory_bytes, architecture_for(learning_, width), objective_);
    hidden_ = policy_->initial_state(lanes);
    history_ = at::empty({0}, at::kFloat);
    history_lengths_ = at::empty({0}, at::kLong);
    if (policy_->architecture().kind == PpoNetworkKind::gru) {
        history_ = at::zeros({ppo_maximum_history, static_cast<int64_t>(lanes), static_cast<int64_t>(width)}, at::kFloat);
        history_lengths_ = at::zeros({static_cast<int64_t>(lanes)}, at::kLong);
        buffers_.episode_starts = at::zeros({static_cast<int64_t>(shape.ticks), static_cast<int64_t>(lanes)}, at::kBool);
        buffers_.prefix_observations = history_.clone();
        buffers_.prefix_lengths = history_lengths_.clone();
    }
    replay_ = make_replay(width, policy_->architecture(), learning_, config_.seed);
}

void PpoTrainer::reset_pending() {
    if (reset_lanes_.empty()) {
        return;
    }
    if (!learning_.enabled) {
        batch_->reset(reset_lanes_);
        return;
    }
    auto state = batch_->snapshot();
    auto active = active_inputs_;
    auto indices = source_indices_;
    auto next = next_source_;
    auto context = std::make_unique<PolicyContext>(active_inputs_, learning_, config_.seed, batch_->observations());
    context->restore(context_->snapshot());
    for (const auto lane : reset_lanes_) {
        indices[lane] = next++ % inputs_.size();
        active[lane] = inputs_[indices[lane]];
        simulation::FinancialSession fresh(active[lane].tape, active[lane].parameters);
        state.sessions[lane] = fresh.snapshot();
        const auto& source_context = active[lane].context;
        state.context_sources[lane] = source_context ? source_context->source_sha256 : "";
    }
    auto batch = std::make_unique<simulation::FinancialBatch>(active, config_.workers);
    batch->restore(state);
    for (const auto lane : reset_lanes_) {
        context->reset(lane, active[lane], batch->observations().subspan(lane * batch->observation_width(), batch->observation_width()));
    }
    batch_ = std::move(batch);
    context_ = std::move(context);
    active_inputs_ = std::move(active);
    source_indices_ = std::move(indices);
    next_source_ = next;
    if (policy_->architecture().kind == PpoNetworkKind::gru) {
        for (const auto lane : reset_lanes_) {
            hidden_[static_cast<int64_t>(lane)].zero_();
            history_.select(1, static_cast<int64_t>(lane)).zero_();
            history_lengths_[static_cast<int64_t>(lane)].zero_();
        }
    }
}

void PpoTrainer::rebuild_hidden() {
    if (policy_->architecture().kind == PpoNetworkKind::gru) {
        hidden_ = policy_->state_from_history(history_.to(at::Device(device_)),
                                               history_lengths_.to(at::Device(device_)));
    }
}

bool PpoTrainer::advance(const PpoDecisionObserver& observer) {
    require(!failed_, "La actualización falló y necesita recuperar un checkpoint confirmado");
    if (transitions_ == config_.total_transitions) {
        return false;
    }
    const auto lanes = batch_->size();
    const bool recurrent = policy_->architecture().kind == PpoNetworkKind::gru;
    const bool double_dqn = policy_->architecture().double_dqn;
    const bool reusable = !recurrent && !double_dqn && !context_;
    const auto rng = policy_->random_state();
    auto cached = std::exchange(bootstrap_, std::nullopt);
    std::optional<BootstrapCache> proposed;
    std::optional<PpoTrainingState> before_reset;
    if (!reset_lanes_.empty()) {
        before_reset = snapshot();
    }
    std::size_t learning_steps = 0;
    bool committed = false;
    at::Tensor next_for_replay;
    try {
        reset_pending();
        const auto observation = context_ ? context_->observations() :
            observation_tensor(batch_->observations(), lanes, batch_->observation_width());
        buffers_.observations.select(0, static_cast<int64_t>(ticks_)).copy_(observation);
        auto eligible = context_ ? context_->training_mask() : std::vector<uint8_t>(lanes, 1);
        std::vector<uint8_t> active(lanes, 1);
        const auto remaining = config_.total_transitions - transitions_;
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            if (eligible[lane] != 0) {
                if (learning_steps < remaining) {
                    ++learning_steps;
                } else {
                    eligible[lane] = 0;
                    active[lane] = 0;
                }
            }
        }
        auto starts = at::zeros({static_cast<int64_t>(lanes)}, at::kBool);
        const std::span starts_data(starts.data_ptr<bool>(), lanes);
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            starts_data[lane] = batch_->cursor(lane) == 0;
        }
        if (recurrent) {
            require(history_lengths_.max().item<int64_t>() < ppo_maximum_history,
                    "La historia del episodio supera el presupuesto recurrente");
            if (ticks_ == 0) {
                buffers_.prefix_observations = history_.clone();
                buffers_.prefix_lengths = history_lengths_.clone();
            }
            buffers_.episode_starts[static_cast<int64_t>(ticks_)].copy_(starts);
        }
        constexpr double minimum_epsilon = 0.05;
        const auto fraction = std::min(1., static_cast<double>(transitions_) /
            static_cast<double>(std::max(std::size_t{1}, config_.total_transitions / 2)));
        // La MLP no depende del estado ni de los reinicios: el forward del bootstrap anterior
        // sobre la misma observación y con los mismos pesos da los mismos logits y valores.
        const bool reuse = reusable && cached && cached->optimizer_steps == policy_->optimizer_steps() &&
                           at::equal(cached->observation, observation);
        const auto chosen = double_dqn ? policy_->act_double_dqn(observation.to(at::Device(device_)),
                            1. + fraction * (minimum_epsilon - 1.)) :
            reuse ? policy_->sample(cached->output) :
            policy_->act_recurrent(observation.to(at::Device(device_)), hidden_, starts.to(at::Device(device_)));
        const auto packed = chosen.packed.to(at::kCPU).contiguous();
        if (objective_.enabled()) {
            buffers_.old_action_weights[static_cast<int64_t>(ticks_)].copy_(chosen.probabilities.to(at::kCPU));
        }
        const auto probabilities = observer ? chosen.probabilities.to(at::kCPU, at::kFloat).contiguous() : at::Tensor{};
        const std::span values(packed.const_data_ptr<double>(), lanes * packed_fields);
        require(std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); }),
                "La política produjo una probabilidad o un valor no finito");
        auto recorded_actions = row<int64_t>(buffers_.actions, ticks_);
        auto old_logp = row<double>(buffers_.old_log_probabilities, ticks_);
        auto old_values = row<double>(buffers_.old_values, ticks_);
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            const auto action = values[lane * packed_fields];
            require(action >= 0 && action < static_cast<double>(ppo_action_count) &&
                        std::floor(action) == action, "La política produjo una acción inválida");
            actions_[lane] = eligible[lane] != 0 ? static_cast<uint8_t>(action) : uint8_t{1};
            recorded_actions[lane] = static_cast<int64_t>(actions_[lane]);
            old_logp[lane] = values[lane * packed_fields + 1];
            old_values[lane] = values[lane * packed_fields + 2];
        }
        const auto& transition = batch_->step_checked(actions_, active, [&](auto next, const auto& outcome) {
            const auto following = context_ ? context_->prepare(next, actions_, outcome, active) :
                observation_tensor(next, lanes, batch_->observation_width());
            if (double_dqn) {
                next_for_replay = following;
            }
            auto bootstrap = policy_->infer(following.to(at::Device(device_)), chosen.next_state);
            const auto values_after = bootstrap.values.to(at::kCPU, at::kDouble).contiguous();
            if (reusable) {
                proposed = BootstrapCache{following, std::move(bootstrap), policy_->optimizer_steps()};
            }
            const std::span next_view(values_after.template const_data_ptr<double>(), lanes);
            require(std::all_of(next_view.begin(), next_view.end(), [](double value) { return std::isfinite(value); }),
                    "El bootstrap produjo un valor no finito");
            std::copy(next_view.begin(), next_view.end(), row<double>(buffers_.next_values, ticks_).begin());
            std::copy(outcome.rewards.begin(), outcome.rewards.end(), row<double>(buffers_.rewards, ticks_).begin());
            auto valid = row<bool>(buffers_.reward_valid, ticks_);
            auto terminated = row<bool>(buffers_.terminated, ticks_);
            auto truncated = row<bool>(buffers_.truncated, ticks_);
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                valid[lane] = eligible[lane] != 0 && outcome.reward_valid[lane] != 0;
                terminated[lane] = outcome.terminated[lane] != 0;
                truncated[lane] = outcome.truncated[lane] != 0;
            }
            if (observer) {
                std::vector<DecisionRecord> records;
                records.reserve(lanes);
                for (std::size_t lane = 0; lane < lanes; ++lane) {
                    if (active[lane] == 0) {
                        continue;
                    }
                    DecisionRecord record;
                    record.decision_id = observed_transitions_ + records.size() + 1;
                    record.lane = static_cast<uint32_t>(lane);
                    record.world_sha256 = active_inputs_[lane].tape->source_sha256;
                    record.context_sha256 = active_inputs_[lane].context ? active_inputs_[lane].context->source_sha256 : std::string(digest_characters, '0');
                    record.episode = context_ ? context_->episode(lane) : 0;
                    record.cursor = batch_->cursor(lane);
                    record.optimizer_step = optimizer_steps_;
                    record.decision_at = active_inputs_[lane].tape->close_times[record.cursor];
                    record.outcome_at = active_inputs_[lane].tape->close_times[record.cursor + 1];
                    record.action = actions_[lane];
                    record.mode = eligible[lane] != 0 ? "sampled" : "warmup";
                    record.learning_allowed = eligible[lane] != 0;
                    const auto likelihood = std::span(probabilities.const_data_ptr<float>(), lanes * trace_action_count)
                        .subspan(lane * trace_action_count, trace_action_count);
                    std::copy(likelihood.begin(), likelihood.end(), record.probabilities.begin());
                    if (eligible[lane] == 0) {
                        record.probabilities.fill(0);
                        record.probabilities[1] = 1;
                    }
                    record.critic = old_values[lane];
                    if (context_) {
                        const auto& recalled = context_->retrieved()[lane];
                        record.retrieved_count = static_cast<uint8_t>(recalled.count);
                        for (std::size_t neighbor = 0; neighbor < recalled.count; ++neighbor) {
                            const auto& previous = recalled.neighbors.at(neighbor);
                            record.ids.at(neighbor) = previous.record.id;
                            record.similarities.at(neighbor) = previous.similarity;
                            record.matured_at.at(neighbor) = previous.record.maturity_at;
                        }
                    }
                    record.reward = outcome.rewards[lane];
                    record.reward_valid = outcome.reward_valid[lane] != 0;
                    record.terminated = outcome.terminated[lane] != 0;
                    record.truncated = outcome.truncated[lane] != 0;
                    record.costs_delta = outcome.costs[lane];
                    records.push_back(std::move(record));
                }
                observer(records);
            }
        });
        committed = true;
        bootstrap_ = std::move(proposed);
        if (context_) {
            context_->commit();
        }
        if (recurrent) {
            const std::span lengths(history_lengths_.data_ptr<int64_t>(), lanes);
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                if (active[lane] != 0) {
                    history_[lengths[lane]][static_cast<int64_t>(lane)]
                        .copy_(buffers_.observations[static_cast<int64_t>(ticks_)][static_cast<int64_t>(lane)]);
                    ++lengths[lane];
                    hidden_[static_cast<int64_t>(lane)].copy_(chosen.next_state[static_cast<int64_t>(lane)]);
                }
            }
        }
        reset_lanes_.clear();
        auto environment_step = transitions_;
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            if (active[lane] == 0) {
                continue;
            }
            ++observed_transitions_;
            invalid_transitions_ += eligible[lane] != 0 && transition.reward_valid[lane] == 0 ? 1 : 0;
            if (eligible[lane] != 0) {
                ++environment_step;
            }
            if (replay_ && eligible[lane] != 0 && transition.reward_valid[lane] != 0) {
                replay_->add(buffers_.observations[static_cast<int64_t>(ticks_)][static_cast<int64_t>(lane)],
                    static_cast<int64_t>(actions_[lane]), transition.rewards[lane], transition.terminated[lane] != 0,
                    double_dqn ? next_for_replay[static_cast<int64_t>(lane)] : at::Tensor{});
                if (double_dqn && environment_step >= dqn_learning_warmup &&
                    replay_->size() >= static_cast<std::size_t>(default_ppo_minibatch_size)) {
                    const auto sampled = replay_->sample(static_cast<std::size_t>(default_ppo_minibatch_size));
                    const DqnBatch batch{sampled.observations, sampled.next_observations, sampled.actions,
                        sampled.rewards, sampled.terminated, at::ones_like(sampled.terminated)};
                    const auto update = policy_->update_double_dqn(batch, environment_step);
                    optimizer_steps_ += static_cast<std::size_t>(update.updates);
                }
            }
            if (transition.terminated[lane] != 0 || transition.truncated[lane] != 0) {
                reset_lanes_.push_back(lane);
                ++episodes_;
            }
        }
    } catch (...) {
        bootstrap_.reset();
        if (context_) {
            context_->cancel();
        }
        if (committed) {
            failed_ = true;
        } else {
            policy_->restore_random_state(rng);
            if (before_reset) {
                restore(*before_reset);
            }
        }
        throw;
    }
    transitions_ += learning_steps;
    ++collector_ticks_;
    ++ticks_;
    if (ticks_ * lanes == config_.rollout_transitions || transitions_ == config_.total_transitions) {
        // La actualización cambia los pesos: el siguiente paso vuelve a ejecutar la red.
        bootstrap_.reset();
        try {
            if (!double_dqn) {
                const auto rollout = rollout_prefix(buffers_, ticks_, false);
                last_update_ = recurrent ? policy_->update(move_rollout(rollout, at::Device(device_)))
                                         : policy_->update_from_cpu(rollout);
                optimizer_steps_ += static_cast<std::size_t>(last_update_.minibatches);
                if (policy_->architecture().auxiliary && last_update_.valid_transitions != 0) {
                    const auto auxiliary = policy_->consolidate(replay_->observations(), replay_->rewards());
                    auxiliary_samples_ += static_cast<std::size_t>(auxiliary.samples);
                }
            }
            ticks_ = 0;
            if (last_update_.minibatches != 0) {
                rebuild_hidden();
            }
        } catch (...) {
            // Una actualización puede haber aplicado minibatches. Solo se recupera del estado persistido.
            failed_ = true;
            throw;
        }
    }
    return true;
}

PpoTrainingState PpoTrainer::snapshot() const {
    require(!failed_, "No se guarda una actualización PPO incompleta");
    std::ostringstream policy;
    policy_->save(policy);
    PpoTrainingState result;
    result.config = config_;
    result.hyperparameters = hyperparameters_;
    result.device = device_;
    result.diagnostic = diagnostic_;
    result.transitions = transitions_;
    result.optimizer_steps = optimizer_steps_;
    result.invalid_transitions = invalid_transitions_;
    result.episodes = episodes_;
    result.reset_lanes = reset_lanes_;
    result.environment = batch_->snapshot();
    result.rollout = rollout_prefix(buffers_, ticks_, true);
    result.policy_archive = std::move(policy).str();
    result.learning = learning_;
    result.objective = objective_;
    result.controller = policy_->controller_state();
    result.source_indices = source_indices_;
    result.next_source = next_source_;
    result.observed_transitions = observed_transitions_;
    if (learning_.enabled) {
        result.adaptive_archive = adaptive_archive({hidden_, history_, history_lengths_,
            context_->snapshot().archive, collector_ticks_, replay_ ? serialize_replay(replay_->snapshot()) : "",
            auxiliary_samples_});
    }
    return result;
}

void PpoTrainer::restore(const PpoTrainingState& state) {
    bootstrap_.reset();
    const auto lanes = batch_->size();
    const auto observed = learning_.enabled ? state.observed_transitions : state.transitions;
    constexpr auto maximum_observed = maximum_transitions * static_cast<std::size_t>(ppo_maximum_history);
    require(state.config == config_ && state.hyperparameters == hyperparameters_ &&
                state.device == device_ && state.diagnostic == diagnostic_ && state.learning == learning_ &&
                state.objective == objective_ &&
                state.transitions <= config_.total_transitions && (learning_.enabled || state.transitions % lanes == 0) &&
                observed >= state.transitions && observed <= maximum_observed &&
                state.invalid_transitions <= state.transitions && state.episodes <= observed &&
                state.optimizer_steps <= maximum_observed * static_cast<std::size_t>(hyperparameters_.epochs) &&
                state.reset_lanes.size() <= lanes && state.episodes >= state.reset_lanes.size(),
            "La recuperación PPO no corresponde a la configuración y presupuesto");
    auto indices = state.source_indices;
    if (!learning_.enabled && indices.empty()) {
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            indices.push_back(lane);
        }
    }
    require(indices.size() == lanes, "La recuperación no conserva sus fuentes activas");
    std::vector<simulation::BatchInput> active;
    for (const auto index : indices) {
        require(index < inputs_.size(), "Una fuente recuperada sale del catálogo");
        active.push_back(inputs_[index]);
    }
    auto candidate_batch = std::make_unique<simulation::FinancialBatch>(active, config_.workers);
    candidate_batch->restore(state.environment);
    AdaptiveState adaptive;
    std::unique_ptr<PolicyContext> context;
    if (learning_.enabled) {
        adaptive = read_adaptive(state.adaptive_archive);
        const bool partial_batch = observed % lanes != 0;
        const auto observed_ticks = observed / lanes + static_cast<std::size_t>(partial_batch);
        require(adaptive.ticks == observed_ticks &&
                    (!partial_batch || state.transitions == config_.total_transitions) &&
                    state.next_source == lanes + state.episodes - state.reset_lanes.size(),
                "El catálogo o las observaciones no corresponden al cursor recuperado");
        context = std::make_unique<PolicyContext>(active, learning_, config_.seed, candidate_batch->observations());
        context->restore({adaptive.context});
        const auto frames = learning_.variant == "ppo_window" ? static_cast<int64_t>(policy_window) : int64_t{1};
        const auto restored_raw = context->observations().view({static_cast<int64_t>(lanes), frames, -1})
            .select(1, frames - 1).narrow(1, 0, static_cast<int64_t>(candidate_batch->observation_width()));
        require(at::equal(restored_raw, observation_tensor(candidate_batch->observations(), lanes,
                                                           candidate_batch->observation_width())),
                "El contexto recuperado no corresponde a las observaciones de su cartera");
        std::size_t accounted_episodes = state.reset_lanes.size();
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            require(context->cursor(lane) == candidate_batch->cursor(lane),
                    "El contexto y la contabilidad no comparten cursor");
            const auto resets = context->episode(lane);
            require(resets <= state.episodes - accounted_episodes,
                    "El contexto contiene más reinicios que episodios confirmados");
            accounted_episodes += static_cast<std::size_t>(resets);
        }
        require(accounted_episodes == state.episodes,
                "Los episodios no concuerdan con los reinicios del contexto y los pendientes");
    } else {
        adaptive.ticks = state.transitions / lanes;
    }
    if (objective_.enabled()) {
        state.controller.validate(objective_, static_cast<int64_t>(state.optimizer_steps),
                                   hyperparameters_.epochs, static_cast<int64_t>(config_.rollout_transitions));
        const auto rollout_ticks = config_.rollout_transitions / lanes;
        const auto completed = adaptive.ticks / rollout_ticks + static_cast<std::size_t>(
            state.transitions == config_.total_transitions && adaptive.ticks % rollout_ticks != 0);
        require(state.controller.completed_rollouts >= 0 &&
                    static_cast<std::size_t>(state.controller.completed_rollouts) == completed,
                "El controlador no conserva los rollouts realmente observados");
    }
    require(state.rollout.observations.defined() && state.rollout.observations.dim() == 3,
            "Falta la forma temporal del recorrido guardado");
    const auto ticks = static_cast<std::size_t>(state.rollout.observations.size(0));
    const auto expected_ticks = state.transitions == config_.total_transitions
        ? 0 : adaptive.ticks % (config_.rollout_transitions / lanes);
    require(ticks == expected_ticks, "El recorrido parcial no corresponde al cursor confirmado");
    const auto width = context ? context->observation_width() : batch_->observation_width();
    validate_partial(state.rollout, {ticks, lanes, width, false, objective_.enabled()});
    std::vector<uint8_t> expected_reset(lanes, 0);
    for (const auto lane : state.reset_lanes) {
        require(lane < lanes && expected_reset[lane] == 0, "Los reinicios guardados no son válidos");
        expected_reset[lane] = 1;
    }
    for (std::size_t lane = 0; lane < lanes; ++lane) {
        require(state.environment.sessions[lane].done == (expected_reset[lane] != 0),
                "Falta un reinicio pendiente en el estado PPO");
        const auto cursor = state.environment.sessions[lane].cursor;
        require(observed == 0 || cursor > 0,
                "El entorno inicial no corresponde a decisiones ya confirmadas");
        require(cursor <= adaptive.ticks && (state.episodes != 0 || cursor == adaptive.ticks),
                "El cursor del entorno adelanta al presupuesto confirmado");
    }
    std::istringstream policy_archive(state.policy_archive);
    auto candidate_policy = std::make_unique<PpoPolicy>(PpoPolicy::load(policy_archive, device_));
    require(candidate_policy->observation_width() == width &&
                candidate_policy->hyperparameters() == hyperparameters_ &&
                candidate_policy->architecture() == policy_->architecture() &&
                candidate_policy->objective() == objective_ &&
                (!objective_.enabled() || candidate_policy->controller_state() == state.controller) &&
                candidate_policy->seed() == config_.seed &&
                candidate_policy->optimizer_steps() == state.optimizer_steps,
            "El checkpoint neural usa otra arquitectura o hiperparámetros");
    require(adaptive.auxiliary_samples >= candidate_policy->auxiliary_steps() &&
                adaptive.auxiliary_samples <= candidate_policy->auxiliary_steps() *
                    static_cast<std::size_t>(ppo_auxiliary_batch_size),
            "Las muestras auxiliares no corresponden a las actualizaciones recuperadas");
    auto candidate_replay = make_replay(width, candidate_policy->architecture(), learning_, config_.seed);
    if (candidate_replay) {
        const auto stored = deserialize_replay(adaptive.replay);
        require(stored.seen == state.transitions - state.invalid_transitions &&
                    candidate_policy->dqn_environment_step() <= state.transitions &&
                    candidate_policy->auxiliary_steps() <= state.optimizer_steps,
                "El replay o sus actualizaciones no corresponden al cursor de aprendizaje");
        candidate_replay->restore(stored);
    } else {
        require(adaptive.replay.empty(), "Una política sin replay contiene experiencias ajenas");
    }
    auto candidate_buffers = allocate_rollout({config_.rollout_transitions / lanes, lanes, width,
                                              false, objective_.enabled()});
    auto target = fields(candidate_buffers);
    const auto source = fields(state.rollout);
    for (std::size_t index = 0; index < target.size(); ++index) {
        target.at(index).second->narrow(0, 0, static_cast<int64_t>(ticks)).copy_(*source.at(index).second);
    }
    if (objective_.enabled()) {
        candidate_buffers.old_action_weights.narrow(0, 0, static_cast<int64_t>(ticks)).copy_(state.rollout.old_action_weights);
    }
    auto candidate_hidden = candidate_policy->initial_state(lanes);
    auto candidate_history = at::empty({0}, at::kFloat);
    auto candidate_lengths = at::empty({0}, at::kLong);
    if (candidate_policy->architecture().kind == PpoNetworkKind::gru) {
        const auto count = static_cast<int64_t>(lanes);
        const auto dimensions = static_cast<int64_t>(width);
        const auto check = [](const at::Tensor& tensor, at::ScalarType dtype, at::IntArrayRef shape) {
            require(tensor.defined() && tensor.device().is_cpu() && tensor.scalar_type() == dtype &&
                        tensor.layout() == at::kStrided && tensor.sizes() == shape &&
                        at::isfinite(tensor).all().item<bool>(),
                    "El historial recurrente no conserva forma, precisión o valores finitos");
        };
        check(adaptive.hidden, at::kFloat, {count, static_cast<int64_t>(recurrent_width)});
        check(adaptive.history, at::kFloat, {ppo_maximum_history, count, dimensions});
        check(adaptive.lengths, at::kLong, {count});
        check(state.rollout.episode_starts, at::kBool, {static_cast<int64_t>(ticks), count});
        check(state.rollout.prefix_observations, at::kFloat, {ppo_maximum_history, count, dimensions});
        check(state.rollout.prefix_lengths, at::kLong, {count});
        require(((state.rollout.prefix_lengths >= 0) & (state.rollout.prefix_lengths <= ppo_maximum_history)).all().item<bool>(),
                "El prefijo recurrente tiene longitudes no admitidas");
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            require(adaptive.lengths[static_cast<int64_t>(lane)].item<int64_t>() ==
                        static_cast<int64_t>(candidate_batch->cursor(lane)),
                    "La historia recurrente no corresponde a su cartera");
        }
        candidate_hidden = adaptive.hidden.to(at::Device(device_));
        candidate_history = adaptive.history.clone();
        candidate_lengths = adaptive.lengths.clone();
        candidate_buffers.episode_starts = at::zeros({static_cast<int64_t>(config_.rollout_transitions / lanes), count}, at::kBool);
        candidate_buffers.episode_starts.narrow(0, 0, static_cast<int64_t>(ticks)).copy_(state.rollout.episode_starts);
        candidate_buffers.prefix_observations = state.rollout.prefix_observations.clone();
        candidate_buffers.prefix_lengths = state.rollout.prefix_lengths.clone();
    }
    auto resets = state.reset_lanes;
    resets.reserve(lanes);
    batch_ = std::move(candidate_batch);
    policy_ = std::move(candidate_policy);
    replay_ = std::move(candidate_replay);
    auxiliary_samples_ = adaptive.auxiliary_samples;
    context_ = std::move(context);
    active_inputs_ = std::move(active);
    source_indices_ = std::move(indices);
    next_source_ = learning_.enabled ? state.next_source : lanes;
    observed_transitions_ = observed;
    collector_ticks_ = adaptive.ticks;
    hidden_ = std::move(candidate_hidden);
    history_ = std::move(candidate_history);
    history_lengths_ = std::move(candidate_lengths);
    buffers_ = std::move(candidate_buffers);
    reset_lanes_ = std::move(resets);
    transitions_ = state.transitions;
    optimizer_steps_ = state.optimizer_steps;
    invalid_transitions_ = state.invalid_transitions;
    episodes_ = state.episodes;
    ticks_ = ticks;
    failed_ = false;
    last_update_ = {};
}

const PpoPolicy& PpoTrainer::policy() const noexcept { return *policy_; }
std::size_t PpoTrainer::transitions() const noexcept { return transitions_; }
std::size_t PpoTrainer::optimizer_steps() const noexcept { return optimizer_steps_; }
std::size_t PpoTrainer::invalid_transitions() const noexcept { return invalid_transitions_; }
std::size_t PpoTrainer::partial_ticks() const noexcept { return ticks_; }
std::size_t PpoTrainer::observed_transitions() const noexcept { return observed_transitions_; }
std::size_t PpoTrainer::auxiliary_samples() const noexcept { return auxiliary_samples_; }
const PpoUpdateStats& PpoTrainer::last_update() const noexcept { return last_update_; }

PpoEvaluation evaluate_policy(const PpoPolicy& policy, std::vector<simulation::BatchInput> inputs,
                              std::size_t workers, const std::function<bool()>& stop,
                              const PpoLearningOptions& learning, const PpoDecisionObserver& observer,
                              bool memory_sensitivity) {
    validate_sources(inputs, "validation");
    require(!memory_sensitivity || (learning.enabled && observer),
            "La sensibilidad necesita contexto adaptativo y un receptor de trazas");
    if (learning.enabled && inputs.size() > learning.environments) {
        require(learning.environments > 0, "La evaluación necesita un lote positivo");
        PpoEvaluation combined;
        simulation::AccurateSum growth;
        simulation::AccurateSum liquidated;
        uint64_t decision_id = 0;
        const PpoDecisionObserver chunk_observer = !observer ? PpoDecisionObserver{} :
            [&](std::span<const DecisionRecord> records) {
                std::vector<DecisionRecord> numbered(records.begin(), records.end());
                for (auto& record : numbered) {
                    record.decision_id = ++decision_id;
                }
                observer(numbered);
            };
        for (std::size_t begin = 0; begin < inputs.size(); begin += learning.environments) {
            const auto end = std::min(begin + learning.environments, inputs.size());
            std::vector<simulation::BatchInput> chunk(inputs.begin() + static_cast<std::ptrdiff_t>(begin),
                                                     inputs.begin() + static_cast<std::ptrdiff_t>(end));
            auto result = evaluate_policy(policy, std::move(chunk), workers, stop, learning,
                                          chunk_observer, memory_sensitivity);
            if (result.paused) {
                combined.paused = true;
                return combined;
            }
            combined.episodes += result.episodes;
            combined.incomplete += result.incomplete;
            combined.ruined += result.ruined;
            combined.metrics.insert(combined.metrics.end(), result.metrics.begin(), result.metrics.end());
            if (result.incomplete == 0) {
                const auto weight = static_cast<double>(result.episodes);
                require(growth.add(result.mean_log_growth * weight) &&
                            liquidated.add(result.mean_liquidated_log_growth * weight),
                        "El agregado de evaluación no es finito");
            }
        }
        if (combined.incomplete == 0) {
            const auto episodes = static_cast<double>(combined.episodes);
            combined.mean_log_growth = growth.value() / episodes;
            combined.mean_liquidated_log_growth = liquidated.value() / episodes;
        }
        return combined;
    }
    simulation::FinancialBatch batch(inputs, workers);
    std::unique_ptr<PolicyContext> context;
    if (learning.enabled) {
        context = std::make_unique<PolicyContext>(inputs, learning, policy.seed(), batch.observations());
    }
    require(policy.observation_width() == (context ? context->observation_width() : batch.observation_width()),
            "La evaluación utiliza otro esquema");
    auto hidden = policy.initial_state(batch.size());
    PpoEvaluation result;
    result.episodes = inputs.size();
    result.metrics.resize(inputs.size());
    std::vector<uint8_t> active(inputs.size(), 1);
    std::vector<uint8_t> actions(inputs.size());
    std::size_t remaining = inputs.size();
    uint64_t decision_id = 0;
    while (remaining != 0) {
        if (stop && stop()) {
            result.paused = true;
            return result;
        }
        const auto observation = context ? context->observations() :
            observation_tensor(batch.observations(), batch.size(), batch.observation_width());
        const auto output = policy.infer(observation.to(at::Device(policy.device())), hidden);
        const auto packed = at::cat({output.logits, output.values.unsqueeze(1)}, 1).to(at::kCPU).contiguous();
        constexpr auto width = static_cast<std::size_t>(ppo_action_count + 1);
        const std::span values(packed.const_data_ptr<float>(), batch.size() * width);
        require(std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); }),
                "La evaluación produjo una política o un valor no finito");
        const auto eligible = context ? context->training_mask() : std::vector<uint8_t>(inputs.size(), 1);
        for (std::size_t lane = 0; lane < actions.size(); ++lane) {
            const auto logits = values.subspan(lane * width, static_cast<std::size_t>(ppo_action_count));
            actions[lane] = eligible[lane] != 0 ?
                static_cast<uint8_t>(std::max_element(logits.begin(), logits.end()) - logits.begin()) : uint8_t{1};
        }
        const auto probabilities = observer ? at::softmax(packed.narrow(1, 0, ppo_action_count), 1) : at::Tensor{};
        at::Tensor masked_logits;
        at::Tensor masked_probabilities;
        if (memory_sensitivity) {
            const auto masked = policy.infer(context->without_retrieved_memory().to(at::Device(policy.device())), hidden);
            masked_logits = masked.logits.to(at::kCPU).contiguous();
            require(at::isfinite(masked_logits).all().item<bool>(),
                    "La sensibilidad produjo logits no finitos");
            masked_probabilities = at::softmax(masked_logits, 1).contiguous();
            require(at::isfinite(masked_probabilities).all().item<bool>(),
                    "La sensibilidad produjo una distribución no finita");
        }
        const auto& outcome = batch.step_checked(actions, active, [&](auto next, const auto& transition) {
            if (context) {
                static_cast<void>(context->prepare(next, actions, transition, active));
            }
            if (observer) {
                std::vector<DecisionRecord> records;
                records.reserve(batch.size());
                for (std::size_t lane = 0; lane < inputs.size(); ++lane) {
                    if (active[lane] == 0) {
                        continue;
                    }
                    DecisionRecord record;
                    record.decision_id = decision_id + records.size() + 1;
                    record.lane = static_cast<uint32_t>(lane);
                    record.world_sha256 = inputs[lane].tape->source_sha256;
                    record.context_sha256 = inputs[lane].context ? inputs[lane].context->source_sha256 : std::string(digest_characters, '0');
                    record.cursor = batch.cursor(lane);
                    record.decision_at = inputs[lane].tape->close_times[record.cursor];
                    record.outcome_at = inputs[lane].tape->close_times[record.cursor + 1];
                    record.optimizer_step = policy.optimizer_steps();
                    record.action = actions[lane];
                    record.mode = eligible[lane] != 0 ? "greedy" : "warmup";
                    record.learning_allowed = false;
                    record.critic = static_cast<double>(values[lane * width + trace_action_count]);
                    const auto likelihood = std::span(probabilities.const_data_ptr<float>(), inputs.size() * trace_action_count)
                        .subspan(lane * trace_action_count, trace_action_count);
                    std::copy(likelihood.begin(), likelihood.end(), record.probabilities.begin());
                    if (eligible[lane] == 0 || policy.architecture().double_dqn) {
                        record.probabilities.fill(0);
                        record.probabilities.at(record.action) = 1;
                    }
                    if (context) {
                        record.episode = context->episode(lane);
                        const auto& retrieved = context->retrieved()[lane];
                        record.retrieved_count = static_cast<uint8_t>(retrieved.count);
                        for (std::size_t neighbor = 0; neighbor < retrieved.count; ++neighbor) {
                            const auto& item = retrieved.neighbors.at(neighbor);
                            record.ids.at(neighbor) = item.record.id;
                            record.similarities.at(neighbor) = item.similarity;
                            record.matured_at.at(neighbor) = item.record.maturity_at;
                        }
                    }
                    if (memory_sensitivity) {
                        MemorySensitivity sensitivity;
                        const auto masked_likelihood = std::span(masked_probabilities.const_data_ptr<float>(),
                            inputs.size() * trace_action_count).subspan(lane * trace_action_count, trace_action_count);
                        std::copy(masked_likelihood.begin(), masked_likelihood.end(), sensitivity.probabilities.begin());
                        const auto masked_scores = std::span(masked_logits.const_data_ptr<float>(),
                            inputs.size() * trace_action_count).subspan(lane * trace_action_count, trace_action_count);
                        sensitivity.action = eligible[lane] != 0 ? static_cast<uint8_t>(
                            std::max_element(masked_scores.begin(), masked_scores.end()) -
                            masked_scores.begin()) : uint8_t{1};
                        if (eligible[lane] == 0 || policy.architecture().double_dqn) {
                            sensitivity.probabilities.fill(0);
                            sensitivity.probabilities.at(sensitivity.action) = 1;
                        }
                        for (std::size_t action = 0; action < trace_action_count; ++action) {
                            sensitivity.probability_l1 += std::abs(static_cast<double>(
                                sensitivity.probabilities.at(action)) - static_cast<double>(record.probabilities.at(action)));
                        }
                        sensitivity.action_changed = sensitivity.action != record.action;
                        record.memory_sensitivity = sensitivity;
                    }
                    record.reward = transition.rewards[lane];
                    record.reward_valid = transition.reward_valid[lane] != 0;
                    record.terminated = transition.terminated[lane] != 0;
                    record.truncated = transition.truncated[lane] != 0;
                    record.costs_delta = transition.costs[lane];
                    records.push_back(std::move(record));
                }
                observer(records);
                decision_id += records.size();
            }
        });
        if (context) {
            context->commit();
        }
        hidden = output.next_state;
        for (std::size_t lane = 0; lane < inputs.size(); ++lane) {
            if (active[lane] == 0 || (outcome.terminated[lane] == 0 && outcome.truncated[lane] == 0)) {
                continue;
            }
            result.metrics[lane] = batch.metrics(lane);
            active[lane] = 0;
            --remaining;
            if (outcome.reward_valid[lane] == 0) {
                ++result.incomplete;
            } else if (outcome.terminated[lane] != 0) {
                ++result.ruined;
            }
        }
    }
    // Una evaluación incompleta no tiene puntuación utilizable y conserva NaN en sus medias.
    if (result.incomplete == 0) {
        const auto final_state = batch.snapshot();
        simulation::AccurateSum log_growth;
        simulation::AccurateSum liquidated_growth;
        for (std::size_t lane = 0; lane < inputs.size(); ++lane) {
            const auto& session = final_state.sessions[lane];
            const auto& parameters = inputs[lane].parameters;
            const auto nav = session.account.nav;
            const auto sold = simulation::liquidated_nav(session, *inputs[lane].tape);
            const auto growth = nav == 0 ? parameters.ruin_penalty
                                        : std::log(nav) - std::log(parameters.capital);
            const auto net = nav == 0 || sold == 0 ? parameters.ruin_penalty
                                                   : std::log(sold) - std::log(parameters.capital);
            require(log_growth.add(growth) && liquidated_growth.add(net),
                    "La evaluación produjo un crecimiento no finito");
        }
        const auto episodes = static_cast<double>(result.episodes);
        result.mean_log_growth = log_growth.value() / episodes;
        result.mean_liquidated_log_growth = liquidated_growth.value() / episodes;
    }
    return result;
}

std::string serialize_rollout(const PpoRollout& rollout) {
    torch::serialize::OutputArchive archive;
    const bool distribution = rollout.old_action_weights.defined();
    if (distribution) {
        require(rollout.observations.defined() && rollout.observations.dim() == 3,
                "El recorrido necesita observaciones antes de escribir sus probabilidades");
        validate_partial(rollout, {static_cast<std::size_t>(rollout.observations.size(0)),
            static_cast<std::size_t>(rollout.observations.size(1)), static_cast<std::size_t>(rollout.observations.size(2)),
            false, true});
    }
    archive.write("schema_version", at::tensor(int64_t{distribution ? 2 : 1}), true);
    for (const auto [name, tensor] : fields(rollout)) {
        archive.write(name, tensor->detach().to(at::kCPU).clone(), true);
    }
    archive.write("recurrent", at::tensor(int64_t{rollout.episode_starts.defined() ? 1 : 0}), true);
    if (rollout.episode_starts.defined()) {
        archive.write("episode_starts", rollout.episode_starts.to(at::kCPU), true);
        archive.write("prefix_observations", rollout.prefix_observations.to(at::kCPU), true);
        archive.write("prefix_lengths", rollout.prefix_lengths.to(at::kCPU), true);
    }
    if (distribution) {
        archive.write("sampler", c10::IValue(std::string(ppo_sampler_contract)));
        archive.write("old_action_weights", rollout.old_action_weights.detach().to(at::kCPU).clone(), true);
    }
    std::ostringstream output;
    archive.save_to(output);
    auto bytes = std::move(output).str();
    require(bytes.size() <= maximum_archive_bytes, "El recorrido serializado supera el presupuesto");
    return bytes;
}

PpoRollout deserialize_rollout(std::string_view bytes) {
    require(!bytes.empty() && bytes.size() <= maximum_archive_bytes, "El archivo de recorrido excede sus límites");
    std::istringstream input{std::string(bytes)};
    torch::serialize::InputArchive archive;
    archive.load_from(input, at::Device(at::kCPU));
    at::Tensor version;
    archive.read("schema_version", version, true);
    require(version.scalar_type() == at::kLong && version.numel() == 1 &&
                (version.item<int64_t>() == 1 || version.item<int64_t>() == 2), "La versión del recorrido no está admitida");
    PpoRollout result;
    for (auto [name, tensor] : fields(result)) {
        archive.read(name, *tensor, true);
    }
    at::Tensor recurrent;
    if (archive.try_read("recurrent", recurrent, true) && recurrent.item<int64_t>() == 1) {
        archive.read("episode_starts", result.episode_starts, true);
        archive.read("prefix_observations", result.prefix_observations, true);
        archive.read("prefix_lengths", result.prefix_lengths, true);
    }
    if (version.item<int64_t>() == 2) {
        c10::IValue sampler;
        archive.read("sampler", sampler);
        require(sampler.isString() && sampler.toStringRef() == ppo_sampler_contract,
                "El recorrido declara otra normalización del muestreador");
        archive.read("old_action_weights", result.old_action_weights, true);
        require(result.observations.defined() && result.observations.dim() == 3,
                "Faltan observaciones para comprobar el muestreador guardado");
        validate_partial(result, {static_cast<std::size_t>(result.observations.size(0)),
            static_cast<std::size_t>(result.observations.size(1)), static_cast<std::size_t>(result.observations.size(2)),
            false, true});
    } else {
        at::Tensor unexpected;
        require(!archive.try_read("old_action_weights", unexpected, true),
                "El recorrido antiguo contiene datos de otro esquema");
    }
    return result;
}

std::string serialize_training_buffer(const PpoTrainingState& state) {
    torch::serialize::OutputArchive archive;
    archive.write("version", at::tensor(int64_t{2}), true);
    archive.write("rollout", bytes_tensor(serialize_rollout(state.rollout)), true);
    archive.write("adaptive", bytes_tensor(state.adaptive_archive), true);
    std::ostringstream output;
    archive.save_to(output);
    auto bytes = std::move(output).str();
    require(bytes.size() <= maximum_archive_bytes, "El checkpoint del recorrido supera 128 MiB");
    return bytes;
}

void restore_training_buffer(std::string_view bytes, PpoTrainingState& state) {
    require(!bytes.empty() && bytes.size() <= maximum_archive_bytes,
            "El checkpoint del recorrido está vacío o supera su presupuesto");
    std::istringstream input{std::string(bytes)};
    torch::serialize::InputArchive archive;
    archive.load_from(input, at::Device(at::kCPU));
    at::Tensor version, rollout, adaptive;
    archive.read("version", version, true);
    require(version.scalar_type() == at::kLong && version.numel() == 1 && version.item<int64_t>() == 2,
            "El checkpoint del recorrido usa otra versión");
    archive.read("rollout", rollout, true);
    archive.read("adaptive", adaptive, true);
    auto restored = deserialize_rollout(tensor_bytes(rollout));
    auto context = tensor_bytes(adaptive);
    state.rollout = std::move(restored);
    state.adaptive_archive = std::move(context);
}
}
