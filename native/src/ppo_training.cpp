#include "mars_titan/ppo_training.hpp"
#include "accurate_sum.hpp"

#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <torch/serialize/archive.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace mars_titan::learning {
namespace {
constexpr std::size_t rollout_field_count = 9;
constexpr std::size_t maximum_transitions = std::size_t{1} << 20;
constexpr std::size_t maximum_rollout = 16384;
constexpr std::size_t diagnostic_transitions = 32;
constexpr std::size_t maximum_archive_bytes = std::size_t{128} * 1024 * 1024;
constexpr std::size_t maximum_rollout_bytes = std::size_t{512} * 1024 * 1024;
constexpr std::size_t packed_fields = 3;

struct RolloutShape {
    std::size_t ticks;
    std::size_t environments;
    std::size_t width;
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
                    input.tape->domain == first.domain && input.tape->parent_id == first.parent_id,
                "Las fuentes mezclan particiones, dominios o predictores padre");
    }
}

void validate_config(const PpoTrainingConfig& config, RolloutShape shape,
                     const std::string& device, bool diagnostic) {
    const auto environments = shape.environments;
    require(environments > 0 && config.total_transitions >= environments &&
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
    // Reserva el buffer y una copia compacta para recuperación o actualización.
    const auto row_bytes = 2 * (shape.width * sizeof(float) + scalar_bytes);
    require(config.rollout_transitions <= config.rollout_bytes / row_bytes,
            "El recorrido PPO supera su presupuesto de memoria");
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
    return result;
}

PpoRollout move_rollout(PpoRollout source, const at::Device& device) {
    for (auto [name, tensor] : fields(source)) {
        static_cast<void>(name);
        *tensor = tensor->to(device);
    }
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
}
}

PpoTrainer::PpoTrainer(std::vector<simulation::BatchInput> inputs, PpoTrainingConfig config,
                       PpoHyperparameters hyperparameters, std::string device, bool diagnostic)
    : inputs_(std::move(inputs)), config_(config), hyperparameters_(hyperparameters),
      device_(std::move(device)), diagnostic_(diagnostic) {
    validate_sources(inputs_, "train");
    hyperparameters_.validate();
    batch_ = std::make_unique<simulation::FinancialBatch>(inputs_, config_.workers);
    const RolloutShape shape{config_.rollout_transitions / batch_->size(), batch_->size(),
                             batch_->observation_width()};
    validate_config(config_, shape, device_, diagnostic_);
    buffers_ = allocate_rollout(shape);
    actions_.resize(batch_->size());
    reset_lanes_.reserve(batch_->size());
    policy_ = std::make_unique<PpoPolicy>(batch_->observation_width(), hyperparameters_, config_.seed,
                                         device_);
}

bool PpoTrainer::advance() {
    require(!failed_, "La actualización falló y necesita recuperar un checkpoint confirmado");
    if (transitions_ == config_.total_transitions) {
        return false;
    }
    const auto lanes = batch_->size();
    const auto rng = policy_->random_state();
    std::optional<simulation::BatchSnapshot> before_reset;
    if (!reset_lanes_.empty()) {
        before_reset = batch_->snapshot();
        batch_->reset(reset_lanes_);
    }
    try {
        const auto observation = observation_tensor(batch_->observations(), lanes, batch_->observation_width());
        buffers_.observations.select(0, static_cast<int64_t>(ticks_)).copy_(observation);
        const auto packed = policy_->act(observation.to(at::Device(device_))).to(at::kCPU).contiguous();
        const std::span values(packed.const_data_ptr<double>(), lanes * packed_fields);
        require(std::all_of(values.begin(), values.end(), [](double value) { return std::isfinite(value); }),
                "La política produjo una probabilidad o un valor no finito");
        auto recorded_actions = row<int64_t>(buffers_.actions, ticks_);
        auto old_logp = row<double>(buffers_.old_log_probabilities, ticks_);
        auto old_values = row<double>(buffers_.old_values, ticks_);
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            const auto action = values[lane * packed_fields];
            require(std::isfinite(action) && action >= 0 && action < static_cast<double>(ppo_action_count) &&
                        std::floor(action) == action, "La política produjo una acción inválida");
            actions_[lane] = static_cast<uint8_t>(action);
            recorded_actions[lane] = static_cast<int64_t>(action);
            old_logp[lane] = values[lane * packed_fields + 1];
            old_values[lane] = values[lane * packed_fields + 2];
        }
        const auto& transition = batch_->step_checked(actions_, {}, [&](auto next, const auto& outcome) {
            const auto following = observation_tensor(next, lanes, batch_->observation_width());
            const auto values_after = policy_->values(following.to(at::Device(device_))).to(at::kCPU, at::kDouble).contiguous();
            const std::span next_view(values_after.template const_data_ptr<double>(), lanes);
            require(std::all_of(next_view.begin(), next_view.end(), [](double value) { return std::isfinite(value); }),
                    "El bootstrap produjo un valor no finito");
            std::copy(next_view.begin(), next_view.end(), row<double>(buffers_.next_values, ticks_).begin());
            std::copy(outcome.rewards.begin(), outcome.rewards.end(), row<double>(buffers_.rewards, ticks_).begin());
            auto valid = row<bool>(buffers_.reward_valid, ticks_);
            auto terminated = row<bool>(buffers_.terminated, ticks_);
            auto truncated = row<bool>(buffers_.truncated, ticks_);
            for (std::size_t lane = 0; lane < lanes; ++lane) {
                valid[lane] = outcome.reward_valid[lane] != 0;
                terminated[lane] = outcome.terminated[lane] != 0;
                truncated[lane] = outcome.truncated[lane] != 0;
            }
        });
        reset_lanes_.clear();
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            invalid_transitions_ += transition.reward_valid[lane] == 0 ? 1 : 0;
            if (transition.terminated[lane] != 0 || transition.truncated[lane] != 0) {
                reset_lanes_.push_back(lane);
                ++episodes_;
            }
        }
    } catch (...) {
        policy_->restore_random_state(rng);
        if (before_reset) {
            batch_->restore(*before_reset);
        }
        throw;
    }
    transitions_ += lanes;
    ++ticks_;
    if (ticks_ * lanes == config_.rollout_transitions || transitions_ == config_.total_transitions) {
        try {
            last_update_ = policy_->update(move_rollout(rollout_prefix(buffers_, ticks_, false), at::Device(device_)));
            optimizer_steps_ += static_cast<std::size_t>(last_update_.minibatches);
            ticks_ = 0;
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
    return {config_, hyperparameters_, device_, diagnostic_, transitions_, optimizer_steps_,
            invalid_transitions_, episodes_, reset_lanes_, batch_->snapshot(),
            rollout_prefix(buffers_, ticks_, true), std::move(policy).str()};
}

void PpoTrainer::restore(const PpoTrainingState& state) {
    const auto lanes = batch_->size();
    require(state.config == config_ && state.hyperparameters == hyperparameters_ &&
                state.device == device_ && state.diagnostic == diagnostic_ &&
                state.transitions <= config_.total_transitions && state.transitions % lanes == 0 &&
                state.invalid_transitions <= state.transitions && state.episodes <= state.transitions &&
                state.optimizer_steps <= maximum_transitions * static_cast<std::size_t>(hyperparameters_.epochs) &&
                state.reset_lanes.size() <= lanes && state.episodes >= state.reset_lanes.size(),
            "La recuperación PPO no corresponde a la configuración y presupuesto");
    require(state.rollout.observations.defined() && state.rollout.observations.dim() == 3,
            "Falta la forma temporal del recorrido guardado");
    const auto ticks = static_cast<std::size_t>(state.rollout.observations.size(0));
    const auto expected_ticks = state.transitions == config_.total_transitions
        ? 0 : (state.transitions % config_.rollout_transitions) / lanes;
    require(ticks == expected_ticks, "El recorrido parcial no corresponde al cursor confirmado");
    validate_partial(state.rollout, {ticks, lanes, batch_->observation_width()});
    auto candidate_batch = std::make_unique<simulation::FinancialBatch>(inputs_, config_.workers);
    candidate_batch->restore(state.environment);
    std::vector<uint8_t> expected_reset(lanes, 0);
    for (const auto lane : state.reset_lanes) {
        require(lane < lanes && expected_reset[lane] == 0, "Los reinicios guardados no son válidos");
        expected_reset[lane] = 1;
    }
    for (std::size_t lane = 0; lane < lanes; ++lane) {
        require(state.environment.sessions[lane].done == (expected_reset[lane] != 0),
                "Falta un reinicio pendiente en el estado PPO");
        const auto cursor = state.environment.sessions[lane].cursor;
        const auto elapsed_ticks = state.transitions / lanes;
        // Reiniciar y avanzar se confirman juntos. Ninguna pausa posterior conserva cursor cero.
        require(state.transitions == 0 || cursor > 0,
                "El entorno inicial no corresponde a decisiones ya confirmadas");
        require(cursor <= elapsed_ticks && (state.episodes != 0 || cursor == elapsed_ticks),
                "El cursor del entorno adelanta al presupuesto confirmado");
    }
    std::istringstream policy_archive(state.policy_archive);
    auto candidate_policy = std::make_unique<PpoPolicy>(PpoPolicy::load(policy_archive, device_));
    require(candidate_policy->observation_width() == batch_->observation_width() &&
                candidate_policy->hyperparameters() == hyperparameters_ &&
                candidate_policy->seed() == config_.seed &&
                candidate_policy->optimizer_steps() == state.optimizer_steps,
            "El checkpoint neural usa otra arquitectura o hiperparámetros");
    auto candidate_buffers = allocate_rollout({config_.rollout_transitions / lanes, lanes, batch_->observation_width()});
    auto target = fields(candidate_buffers);
    const auto source = fields(state.rollout);
    for (std::size_t index = 0; index < target.size(); ++index) {
        target.at(index).second->narrow(0, 0, static_cast<int64_t>(ticks)).copy_(*source.at(index).second);
    }
    auto resets = state.reset_lanes;
    resets.reserve(lanes);
    batch_ = std::move(candidate_batch);
    policy_ = std::move(candidate_policy);
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
const PpoUpdateStats& PpoTrainer::last_update() const noexcept { return last_update_; }

PpoEvaluation evaluate_policy(const PpoPolicy& policy, std::vector<simulation::BatchInput> inputs,
                              std::size_t workers, const std::function<bool()>& stop) {
    validate_sources(inputs, "validation");
    simulation::FinancialBatch batch(inputs, workers);
    require(policy.observation_width() == batch.observation_width(), "La evaluación utiliza otro esquema");
    PpoEvaluation result;
    result.episodes = inputs.size();
    result.metrics.resize(inputs.size());
    std::vector<uint8_t> active(inputs.size(), 1);
    std::vector<uint8_t> actions(inputs.size());
    std::size_t remaining = inputs.size();
    while (remaining != 0) {
        if (stop && stop()) {
            result.paused = true;
            return result;
        }
        const auto observation = observation_tensor(batch.observations(), batch.size(), batch.observation_width());
        const auto output = policy.forward(observation.to(at::Device(policy.device())));
        const auto packed = at::cat({output.logits, output.values.unsqueeze(1)}, 1).to(at::kCPU).contiguous();
        constexpr auto width = static_cast<std::size_t>(ppo_action_count + 1);
        const std::span values(packed.const_data_ptr<float>(), batch.size() * width);
        require(std::all_of(values.begin(), values.end(), [](float value) { return std::isfinite(value); }),
                "La evaluación produjo una política o un valor no finito");
        for (std::size_t lane = 0; lane < actions.size(); ++lane) {
            const auto logits = values.subspan(lane * width, static_cast<std::size_t>(ppo_action_count));
            actions[lane] = static_cast<uint8_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        }
        const auto& outcome = batch.step_active(actions, active);
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
    // Una evaluación incompleta no tiene puntuación utilizable para seleccionar modelos.
    if (result.incomplete != 0) {
        result.mean_log_growth = 0;
    } else {
        const auto final_state = batch.snapshot();
        simulation::AccurateSum log_growth;
        for (std::size_t lane = 0; lane < inputs.size(); ++lane) {
            const auto nav = final_state.sessions[lane].account.nav;
            const auto growth = nav == 0 ? inputs[lane].parameters.ruin_penalty
                                        : std::log(nav) - std::log(inputs[lane].parameters.capital);
            require(log_growth.add(growth), "La evaluación produjo un crecimiento no finito");
        }
        result.mean_log_growth = log_growth.value() / static_cast<double>(result.episodes);
    }
    return result;
}

std::string serialize_rollout(const PpoRollout& rollout) {
    torch::serialize::OutputArchive archive;
    archive.write("schema_version", at::tensor(int64_t{1}), true);
    for (const auto [name, tensor] : fields(rollout)) {
        archive.write(name, tensor->detach().to(at::kCPU).clone(), true);
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
                version.item<int64_t>() == 1, "La versión del recorrido no está admitida");
    PpoRollout result;
    for (auto [name, tensor] : fields(result)) {
        archive.read(name, *tensor, true);
    }
    return result;
}
}
