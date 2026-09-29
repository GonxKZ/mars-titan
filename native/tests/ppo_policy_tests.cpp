#include "mars_titan/ppo_policy.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <torch/serialize/archive.h>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace {
using namespace mars_titan::learning;
constexpr int64_t observation_width = 4;
constexpr int64_t sample_count = 16;
constexpr uint64_t seed = 719;
constexpr double tolerance = 1e-10;
constexpr double altered_observation = 100;
constexpr double invalid_reward = 1e20;

void require(bool condition, std::string_view reason) {
    if (!condition) {
        throw std::runtime_error(std::string(reason));
    }
}

template<class Function> void rejected(Function&& function) {
    try {
        std::forward<Function>(function)();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("Se aceptó una entrada inválida");
}

void same(const at::Tensor& actual, const at::Tensor& expected, std::string_view reason) {
    require(at::allclose(actual, expected, tolerance, tolerance), reason);
}

PpoRollout controlled_rollout(PpoPolicy& policy) {
    const auto observations = at::ones({sample_count, observation_width}, at::kFloat);
    const auto output = policy.forward(observations);
    const auto actions = at::arange(sample_count, at::kLong) >= sample_count / 2;
    const auto old_log_probabilities = output.logits.log_softmax(-1)
                                          .gather(1, actions.to(at::kLong).unsqueeze(1)).squeeze(1);
    PpoRollout result;
    result.observations = observations.reshape({1, sample_count, observation_width});
    result.actions = actions.to(at::kLong).reshape({1, sample_count});
    result.old_log_probabilities = old_log_probabilities.reshape({1, sample_count});
    result.old_values = output.values.reshape({1, sample_count});
    result.rewards = result.old_values.to(at::kDouble) +
                     at::where(actions.reshape({1, sample_count}), -1., 1.);
    result.next_values = at::zeros({1, sample_count}, at::kDouble);
    result.reward_valid = at::ones({1, sample_count}, at::kBool);
    result.terminated = at::ones({1, sample_count}, at::kBool);
    result.truncated = at::zeros({1, sample_count}, at::kBool);
    return result;
}

void clipped_objective_matches_signed_analytic_values() {
    const auto ratios = at::tensor({1.5, 0.5, 1.5, 0.5}, at::kDouble);
    const auto advantages = at::tensor({2.0, 2.0, -2.0, -2.0}, at::kDouble);
    const auto expected = at::tensor({2.4, 1.0, -3.0, -1.6}, at::kDouble);
    const auto logarithms = ratios.log().set_requires_grad(true);
    const auto objective = ppo_clipped_objective(logarithms, at::zeros_like(ratios), advantages, 0.2);
    same(objective, expected, "El objetivo recortado altera las ventajas positivas o negativas");
    objective.sum().backward();
    const auto expected_gradient = at::tensor({0., 1., -3., 0.}, at::kDouble);
    same(logarithms.grad(), expected_gradient,
         "El objetivo recortado propaga un gradiente incorrecto");
}

void clipped_objective_keeps_large_finite_log_ratios() {
    constexpr float rare_log_probability = -100;
    constexpr double clipped_gain = 2.4;
    constexpr float positive_advantage = 2;
    const auto logarithms = at::full({1}, -1., at::kFloat).set_requires_grad(true);
    const auto old = at::full({1}, rare_log_probability, at::kFloat);
    const auto advantages = at::full({1}, positive_advantage, at::kFloat);
    const auto objective = ppo_clipped_objective(logarithms, old, advantages, 0.2);
    same(objective, at::full({1}, clipped_gain, at::kDouble),
         "Un cociente finito en FP64 se rechaza o altera antes del recorte");
    objective.sum().backward();
    require(at::equal(logarithms.grad(), at::zeros_like(logarithms)),
            "El objetivo recortado propaga el gradiente de un cociente extremo");
}

void gae_cuts_each_lane_at_the_correct_boundary() {
    constexpr double previous_value = 0.5;
    constexpr double following_value = 2;
    constexpr double discount = 0.5;
    PpoRollout rollout;
    rollout.rewards = at::tensor({1., 1., 1., 1., 1., 1., 100., 100., 100.}, at::kDouble)
                          .reshape({3, 3});
    rollout.old_values = at::full({3, 3}, previous_value, at::kDouble);
    rollout.next_values = at::full({3, 3}, following_value, at::kDouble);
    rollout.reward_valid = at::ones({3, 3}, at::kBool);
    rollout.terminated = at::zeros({3, 3}, at::kBool);
    rollout.truncated = at::zeros({3, 3}, at::kBool);
    rollout.reward_valid.select(0, 1).select(0, 2).fill_(false);
    rollout.terminated.select(0, 1).select(0, 0).fill_(true);
    rollout.truncated.select(0, 1).select(0, 1).fill_(true);
    PpoHyperparameters parameters;
    parameters.gamma = discount;
    parameters.gae_lambda = discount;
    const auto result = ppo_gae(rollout, parameters);
    const auto expected = at::tensor({1.625, 1.875, 1.5, 0.5, 1.5, 0., 100.5, 100.5, 100.5},
                                    at::kDouble).reshape({3, 3});
    same(result.advantages, expected, "GAE mezcla terminales, truncaciones o carriles inválidos");
    same(result.returns, at::where(rollout.reward_valid, expected + previous_value, 0.),
         "El retorno no coincide con la ventaja más el valor anterior");
}

void inference_uses_only_its_own_observation() {
    PpoPolicy policy(observation_width, {}, seed);
    const auto observations = at::ones({2, observation_width}, at::kFloat);
    const auto before = policy.forward(observations);
    observations.select(0, 1).fill_(altered_observation);
    const auto after = policy.forward(observations);
    same(before.logits.select(0, 0), after.logits.select(0, 0),
         "Una observación ajena altera la acción actual");
    same(before.values.select(0, 0), after.values.select(0, 0),
         "Una observación ajena altera el valor actual");
    require(!before.logits.requires_grad() && !before.values.requires_grad(),
            "La inferencia conserva el grafo de gradientes");
    const auto packed = policy.act(observations);
    require(packed.sizes() == at::IntArrayRef({2, 3}) && packed.scalar_type() == at::kDouble,
            "Las acciones no se pueden transferir en un único lote");
    require((packed.select(1, 0) >= 0).all().item<bool>() &&
                (packed.select(1, 0) < ppo_action_count).all().item<bool>(),
            "La política devuelve una acción fuera del contrato");
    same(policy.values(observations), after.values, "La consulta de valores cambia la predicción");
}

void update_increases_probability_for_a_positive_advantage() {
    PpoHyperparameters parameters;
    constexpr double learning_rate = 0.005;
    parameters.learning_rate = learning_rate;
    parameters.entropy = 0;
    parameters.value_weight = 0;
    parameters.epochs = 2;
    parameters.minibatch_size = sample_count;
    PpoPolicy policy(observation_width, parameters, seed);
    const auto rollout = controlled_rollout(policy);
    const double before = rollout.old_log_probabilities.flatten()
                              .narrow(0, 0, sample_count / 2).mean().item<double>();
    const auto result = policy.update(rollout);
    const auto after = policy.forward(rollout.observations.flatten(0, 1));
    require(after.logits.log_softmax(-1).select(1, 0).mean().item<double>() > before,
            "La actualización no favorece una acción con ventaja positiva");
    require(result.valid_transitions == sample_count && result.minibatches == parameters.epochs,
            "La actualización no registra las transiciones y pasos efectivos");
    require(std::isfinite(result.policy_loss) && std::isfinite(result.value_loss) &&
                std::isfinite(result.entropy), "La actualización devuelve métricas no finitas");
}

void constant_advantages_leave_actor_unchanged() {
    PpoHyperparameters parameters;
    parameters.entropy = 0;
    parameters.value_weight = 0;
    PpoPolicy policy(observation_width, parameters, seed);
    auto rollout = controlled_rollout(policy);
    rollout.rewards = rollout.old_values.to(at::kDouble) + 1;
    const auto observations = rollout.observations.flatten(0, 1);
    const auto before = policy.forward(observations);
    static_cast<void>(policy.update(rollout));
    const auto after = policy.forward(observations);
    require(at::equal(after.logits, before.logits),
            "Las ventajas constantes no se normalizan a cero");
}

void invalid_rewards_do_not_update_actor_critic_or_rng() {
    PpoPolicy policy(observation_width, {}, seed);
    PpoPolicy control(observation_width, {}, seed);
    auto rollout = controlled_rollout(policy);
    rollout.reward_valid.zero_();
    rollout.rewards.fill_(invalid_reward);
    auto unlabelled = controlled_rollout(control);
    unlabelled.reward_valid.zero_();
    unlabelled.rewards.zero_();
    const auto random_before = policy.random_state();
    const auto before = policy.forward(rollout.observations.flatten(0, 1));
    const auto result = policy.update(rollout);
    const auto expected = control.update(unlabelled);
    const auto after = policy.forward(rollout.observations.flatten(0, 1));
    const std::array actual_metrics{result.policy_loss, result.value_loss, result.entropy,
        result.approximate_kl, result.clip_fraction, result.gradient_norm};
    const std::array expected_metrics{expected.policy_loss, expected.value_loss, expected.entropy,
        expected.approximate_kl, expected.clip_fraction, expected.gradient_norm};
    require(result.valid_transitions == 0 && result.minibatches == 0 &&
                expected.valid_transitions == 0 && expected.minibatches == 0 &&
                actual_metrics == expected_metrics && actual_metrics == decltype(actual_metrics){} &&
                policy.optimizer_steps() == 0,
            "Se entrenó con recompensas inválidas");
    require(at::equal(after.logits, before.logits) && at::equal(after.values, before.values),
            "Las recompensas inválidas cambian el actor o el crítico");
    const auto random_after = policy.random_state();
    require(at::equal(random_after.sampling, random_before.sampling) &&
                at::equal(random_after.shuffle, random_before.shuffle),
            "Una actualización vacía consume el estado de muestreo o de minibatches");
    rollout.rewards.fill_(std::numeric_limits<double>::quiet_NaN());
    rejected([&] { static_cast<void>(policy.update(rollout)); });
    rollout.rewards.zero_();
    rollout.actions = at::zeros({sample_count}, at::kLong);
    rejected([&] { static_cast<void>(policy.update(rollout)); });
}

void masked_transitions_match_a_rollout_containing_only_valid_steps() {
    PpoHyperparameters parameters;
    parameters.epochs = 1;
    PpoPolicy policy(observation_width, parameters, seed);
    PpoPolicy control(observation_width, parameters, seed);
    auto rollout = controlled_rollout(policy);
    constexpr int64_t valid_count = sample_count / 2;
    rollout.reward_valid.narrow(1, valid_count, valid_count).fill_(false);
    rollout.observations.narrow(1, valid_count, valid_count).fill_(altered_observation);
    rollout.rewards.narrow(1, valid_count, valid_count).fill_(invalid_reward);
    PpoRollout filtered{
        rollout.observations.narrow(1, 0, valid_count),
        rollout.actions.narrow(1, 0, valid_count),
        rollout.old_log_probabilities.narrow(1, 0, valid_count),
        rollout.old_values.narrow(1, 0, valid_count),
        rollout.rewards.narrow(1, 0, valid_count),
        rollout.next_values.narrow(1, 0, valid_count),
        rollout.reward_valid.narrow(1, 0, valid_count),
        rollout.terminated.narrow(1, 0, valid_count),
        rollout.truncated.narrow(1, 0, valid_count)
    };
    static_cast<void>(policy.update(rollout));
    static_cast<void>(control.update(filtered));
    const auto observations = at::ones({sample_count, observation_width}, at::kFloat);
    const auto actual = policy.forward(observations);
    const auto expected = control.forward(observations);
    require(at::equal(actual.logits, expected.logits) && at::equal(actual.values, expected.values),
            "Las transiciones inválidas contribuyen al actor, crítico o entropía");
}

void policy_sampling_does_not_consume_global_random_state() {
    const auto& global = at::detail::getDefaultCPUGenerator();
    const auto before = global.get_state();
    PpoPolicy policy(observation_width, {}, seed);
    static_cast<void>(policy.act(at::ones({sample_count, observation_width}, at::kFloat)));
    require(at::equal(before, global.get_state()), "La política consume el generador aleatorio global");
}

void random_state_restoration_is_atomic() {
    PpoPolicy policy(observation_width, {}, seed);
    const auto observations = at::ones({sample_count, observation_width}, at::kFloat);
    const auto beginning = policy.random_state();
    const auto expected = policy.act(observations);
    policy.restore_random_state(beginning);
    require(at::equal(expected, policy.act(observations)), "No se recuperó el muestreo del paso fallido");
    const auto before_failure = policy.random_state();
    auto invalid = beginning;
    invalid.shuffle = at::zeros({1}, at::kByte);
    rejected([&] { policy.restore_random_state(invalid); });
    const auto after_failure = policy.random_state();
    require(at::equal(before_failure.sampling, after_failure.sampling) &&
                at::equal(before_failure.shuffle, after_failure.shuffle),
            "Un estado aleatorio inválido deja una recuperación parcial");
}

void checkpoint_preserves_sampling_optimizer_and_next_update() {
    PpoHyperparameters parameters;
    parameters.minibatch_size = 4;
    parameters.epochs = 2;
    PpoPolicy policy(observation_width, parameters, seed);
    auto rollout = controlled_rollout(policy);
    const auto trained = policy.update(rollout);
    require(trained.minibatches > 0, "No se creó estado del optimizador para la prueba");
    static_cast<void>(policy.act(rollout.observations.flatten(0, 1)));
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    policy.save(stream);
    auto restored = PpoPolicy::load(stream, "cpu");
    require(restored.hyperparameters() == parameters && restored.observation_width() == observation_width,
            "La recuperación altera los hiperparámetros o la dimensión de observación");
    same(policy.act(rollout.observations.flatten(0, 1)),
         restored.act(rollout.observations.flatten(0, 1)),
         "El checkpoint altera el siguiente muestreo");
    rollout = controlled_rollout(policy);
    static_cast<void>(policy.update(rollout));
    static_cast<void>(restored.update(rollout));
    const auto expected = policy.forward(rollout.observations.flatten(0, 1));
    const auto actual = restored.forward(rollout.observations.flatten(0, 1));
    require(at::equal(expected.logits, actual.logits) && at::equal(expected.values, actual.values),
            "La recuperación no reproduce exactamente la siguiente actualización");
}

void malformed_or_different_device_checkpoints_are_rejected() {
    std::stringstream malformed("archivo incompleto");
    rejected([&] { static_cast<void>(PpoPolicy::load(malformed)); });
    PpoPolicy policy(observation_width, {}, seed);
    std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
    policy.save(stream);
    rejected([&] { static_cast<void>(PpoPolicy::load(stream, "cuda:0")); });
}

void invalid_rollout_fails_before_mutation() {
    PpoPolicy policy(observation_width, {}, seed);
    auto rollout = controlled_rollout(policy);
    const auto observations = rollout.observations.flatten(0, 1);
    const auto before = policy.forward(observations);
    rollout.actions.select(0, 0).select(0, 0).fill_(ppo_action_count);
    rejected([&] { static_cast<void>(policy.update(rollout)); });
    rollout.actions.zero_();
    const auto saved_mask = rollout.reward_valid;
    rollout.reward_valid = rollout.reward_valid.to(at::kByte);
    rejected([&] { static_cast<void>(policy.update(rollout)); });
    rollout.reward_valid = saved_mask;
    rollout.rewards.select(0, 0).select(0, 0).fill_(std::numeric_limits<double>::quiet_NaN());
    rejected([&] { static_cast<void>(policy.update(rollout)); });
    const auto after = policy.forward(observations);
    same(after.logits, before.logits, "Una entrada inválida cambia el actor antes de fallar");
    same(after.values, before.values, "Una entrada inválida cambia el crítico antes de fallar");
    PpoHyperparameters invalid;
    invalid.learning_rate = std::numeric_limits<double>::infinity();
    rejected([&] { PpoPolicy unusable(observation_width, invalid, seed); });
    rejected([&] { PpoPolicy unusable(0, {}, seed); });
    rejected([&] { PpoPolicy unusable(observation_width, {}, seed, "cpu", 1); });
    rejected([&] { PpoPolicy unusable(observation_width, {}, seed, "other"); });
}

void non_finite_float_gradients_fail_before_the_optimizer_step() {
    PpoHyperparameters parameters;
    parameters.epochs = 1;
    PpoPolicy policy(observation_width, parameters, seed);
    auto rollout = controlled_rollout(policy);
    constexpr double rare_log_probability = -100;
    rollout.old_log_probabilities.fill_(rare_log_probability);
    const auto observations = rollout.observations.flatten(0, 1);
    const auto before = policy.forward(observations);
    rejected([&] { static_cast<void>(policy.update(rollout)); });
    const auto after = policy.forward(observations);
    require(at::equal(before.logits, after.logits) && at::equal(before.values, after.values),
            "Adam modifica parámetros después de encontrar un gradiente no finito");
}

void hyperparameters_reject_non_finite_values_and_invalid_ranges() {
    constexpr std::array fields = {
        &PpoHyperparameters::learning_rate, &PpoHyperparameters::gamma,
        &PpoHyperparameters::gae_lambda, &PpoHyperparameters::clip,
        &PpoHyperparameters::entropy, &PpoHyperparameters::value_weight,
        &PpoHyperparameters::gradient_norm
    };
    constexpr std::array invalid_values = {
        std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(),
        -1., std::numeric_limits<double>::max()
    };
    for (const auto field : fields) {
        for (const auto value : invalid_values) {
            PpoHyperparameters invalid;
            invalid.*field = value;
            rejected([&] { invalid.validate(); });
        }
    }
    for (const auto field : {&PpoHyperparameters::learning_rate, &PpoHyperparameters::clip,
                             &PpoHyperparameters::gradient_norm}) {
        PpoHyperparameters invalid;
        invalid.*field = 0;
        rejected([&] { invalid.validate(); });
    }
    for (const auto field : {&PpoHyperparameters::epochs, &PpoHyperparameters::minibatch_size}) {
        for (const auto value : {int64_t{0}, std::numeric_limits<int64_t>::max()}) {
            PpoHyperparameters invalid;
            invalid.*field = value;
            rejected([&] { invalid.validate(); });
        }
    }
    PpoHyperparameters valid;
    valid.gamma = 0;
    valid.gae_lambda = 0;
    valid.entropy = 0;
    valid.value_weight = 0;
    valid.validate();
}

std::string checkpoint_with_step(std::string_view bytes, std::optional<int64_t> step) {
    std::istringstream source{std::string(bytes)};
    torch::serialize::InputArchive input;
    input.load_from(source, at::Device(at::kCPU));
    c10::IValue optimizer;
    input.read("optimizer", optimizer);
    const auto states = optimizer.toObject()->getAttr("state").toObject();
    require(states->type()->numAttributes() != 0, "La prueba necesita estado de Adam");
    if (step) {
        states->getSlot(0).toObject()->setAttr("step", c10::IValue(*step));
    }
    torch::serialize::OutputArchive output;
    for (const auto& key : input.keys()) {
        c10::IValue value;
        input.read(key, value);
        output.write(key, value);
    }
    std::ostringstream destination;
    output.save_to(destination);
    return std::move(destination).str();
}

void checkpoint_rejects_inconsistent_adam_steps() {
    PpoHyperparameters parameters;
    parameters.epochs = 1;
    parameters.minibatch_size = sample_count;
    PpoPolicy policy(observation_width, parameters, seed);
    require(policy.seed() == seed && policy.optimizer_steps() == 0,
            "La política inicial no conserva semilla o estado vacío de Adam");
    static_cast<void>(policy.update(controlled_rollout(policy)));
    require(policy.optimizer_steps() == 1, "La política no cuenta el paso Adam confirmado");
    std::ostringstream archive;
    policy.save(archive);
    const auto bytes = archive.str();
    std::istringstream control(checkpoint_with_step(bytes, std::nullopt));
    auto restored = PpoPolicy::load(control);
    require(restored.seed() == seed && restored.optimizer_steps() == 1,
            "El checkpoint no conserva semilla y paso común de Adam");
    for (const auto invalid : {int64_t{0}, int64_t{2}, std::numeric_limits<int64_t>::max()}) {
        std::istringstream corrupted(checkpoint_with_step(bytes, invalid));
        rejected([&] { static_cast<void>(PpoPolicy::load(corrupted)); });
    }
}

PpoArchitecture recurrent_architecture() { return {PpoNetworkKind::gru, default_ppo_hidden_width}; }

void architectures_count_parameters_and_keep_legacy_defaults() {
    constexpr int64_t custom_hidden = 80;
    PpoPolicy legacy(observation_width, {}, seed);
    PpoPolicy wider(observation_width, {}, seed, "cpu", default_ppo_memory_bytes,
                    {PpoNetworkKind::mlp, custom_hidden});
    PpoHyperparameters parameters;
    parameters.minibatch_size = ppo_sequence_length;
    PpoPolicy recurrent(observation_width, parameters, seed, "cpu", default_ppo_memory_bytes,
                        recurrent_architecture());
    constexpr std::size_t expected_legacy = 4935;
    constexpr std::size_t expected_wide = 7447;
    constexpr std::size_t expected_gru = 13895;
    require(legacy.architecture() == PpoArchitecture{} && legacy.parameter_count() == expected_legacy,
            "La arquitectura predeterminada altera la MLP existente");
    require(wider.parameter_count() == expected_wide && recurrent.parameter_count() == expected_gru,
            "El número de parámetros no refleja la arquitectura declarada");
    require(wider.initial_state(2).sizes() == at::IntArrayRef({2, 0}),
            "La MLP crea un estado recurrente inexistente");
}

void recurrent_inference_retains_history_and_resets_only_selected_lanes() {
    PpoPolicy policy(observation_width, {}, seed, "cpu", default_ppo_memory_bytes, recurrent_architecture());
    const auto observations = at::ones({2, observation_width}, at::kFloat);
    const auto random_before = policy.random_state();
    const auto initial = policy.infer(observations);
    const auto continued = policy.infer(observations, initial.next_state);
    require(!at::equal(initial.logits, continued.logits), "La GRU no conserva historia entre observaciones");
    const auto starts = at::tensor({1, 0}, at::kLong).to(at::kBool);
    const auto reset = policy.infer(observations, initial.next_state, starts);
    same(reset.logits.select(0, 0), initial.logits.select(0, 0), "La GRU conserva historia tras un reinicio");
    same(reset.logits.select(0, 1), continued.logits.select(0, 1), "El reinicio de un carril altera otro");
    require(!reset.next_state.requires_grad() && !reset.logits.requires_grad(),
            "La inferencia GRU conserva un grafo de gradientes");
    require(at::equal(random_before.sampling, policy.random_state().sampling),
            "La inferencia recurrente consume el generador de acciones");
    const auto action = policy.act_recurrent(observations, initial.next_state, starts);
    same(action.next_state, reset.next_state, "El muestreo recurrente altera el estado inferido");
    require(action.packed.sizes() == at::IntArrayRef({2, 3}), "La acción GRU no conserva el lote empaquetado");
    same(action.probabilities, reset.logits.log_softmax(-1).exp(),
         "La traza de probabilidades no corresponde al muestreo");
}

PpoRollout recurrent_rollout(PpoPolicy& policy, int64_t ticks = ppo_sequence_length) {
    PpoRollout rollout;
    rollout.observations = at::ones({ticks, 1, observation_width}, at::kFloat);
    rollout.actions = (at::arange(ticks, at::kLong) % 2).reshape({ticks, 1});
    rollout.old_log_probabilities = at::zeros({ticks, 1}, at::kDouble);
    rollout.old_values = at::zeros({ticks, 1}, at::kDouble);
    rollout.rewards = at::where(rollout.actions == 0, 1., -1.);
    rollout.next_values = at::zeros({ticks, 1}, at::kDouble);
    rollout.reward_valid = at::ones({ticks, 1}, at::kBool);
    rollout.terminated = at::zeros({ticks, 1}, at::kBool);
    rollout.truncated = at::zeros({ticks, 1}, at::kBool);
    rollout.episode_starts = at::zeros({ticks, 1}, at::kBool);
    constexpr int64_t prefix_length = 4;
    constexpr float prefix_value = 2;
    rollout.prefix_observations = at::full({prefix_length, 1, observation_width}, prefix_value, at::kFloat);
    rollout.prefix_lengths = at::full({1}, prefix_length, at::kLong);
    auto state = policy.initial_state(1);
    for (int64_t time = 0; time < prefix_length; ++time) {
        state = policy.infer(rollout.prefix_observations.select(0, time), state).next_state;
    }
    for (int64_t time = 0; time < ticks; ++time) {
        const auto prediction = policy.infer(rollout.observations.select(0, time), state);
        state = prediction.next_state;
        const auto log_probability = prediction.logits.log_softmax(-1).gather(
            1, rollout.actions.select(0, time).unsqueeze(1)).squeeze(1);
        rollout.old_log_probabilities.select(0, time).copy_(log_probability);
    }
    return rollout;
}

void recurrent_update_reconstructs_prefix_after_each_epoch() {
    PpoHyperparameters parameters;
    parameters.gamma = 0;
    parameters.minibatch_size = ppo_sequence_length;
    parameters.epochs = 2;
    PpoPolicy joint(observation_width, parameters, seed, "cpu", default_ppo_memory_bytes, recurrent_architecture());
    parameters.epochs = 1;
    PpoPolicy separate(observation_width, parameters, seed, "cpu", default_ppo_memory_bytes, recurrent_architecture());
    const auto rollout = recurrent_rollout(joint);
    const auto statistics = joint.update(rollout);
    static_cast<void>(separate.update(rollout));
    static_cast<void>(separate.update(rollout));
    require(statistics.valid_transitions == ppo_sequence_length && statistics.minibatches == 2,
            "PPO recurrente no agrupa las transiciones en secuencias");
    const auto observations = rollout.observations.select(0, 0);
    require(at::equal(joint.infer(observations).logits, separate.infer(observations).logits),
            "La actualización recurrente reutiliza un prefijo con parámetros obsoletos");
}

void recurrent_update_replays_recorded_probabilities_and_excludes_padding() {
    PpoHyperparameters parameters;
    parameters.gamma = 0;
    parameters.minibatch_size = ppo_sequence_length;
    parameters.epochs = 1;
    PpoPolicy policy(observation_width, parameters, seed, "cpu", default_ppo_memory_bytes, recurrent_architecture());
    constexpr int64_t short_ticks = 7;
    constexpr int64_t invalid_tick = 3;
    auto rollout = recurrent_rollout(policy, short_ticks);
    rollout.reward_valid.select(0, invalid_tick).fill_(false);
    const auto statistics = policy.update(rollout);
    require(statistics.valid_transitions == short_ticks - 1 && statistics.minibatches == 1,
            "El relleno o las recompensas inválidas contribuyen al objetivo GRU");
    require(statistics.approximate_kl < tolerance,
            "La actualización GRU no reconstruye la historia con la que se recogió el rollout");
}

void recurrent_checkpoint_restores_architecture_state_and_next_update() {
    PpoHyperparameters parameters;
    parameters.gamma = 0;
    parameters.minibatch_size = ppo_sequence_length;
    parameters.epochs = 1;
    PpoPolicy policy(observation_width, parameters, seed, "cpu", default_ppo_memory_bytes, recurrent_architecture());
    constexpr int64_t short_ticks = 7;
    auto rollout = recurrent_rollout(policy, short_ticks);
    const auto first_update = policy.update(rollout);
    require(first_update.valid_transitions == short_ticks && first_update.minibatches == 1,
            "El relleno de una secuencia corta contribuye al contador PPO");
    std::stringstream archive;
    policy.save(archive);
    auto restored = PpoPolicy::load(archive);
    require(restored.architecture() == policy.architecture(), "La recuperación pierde la arquitectura GRU");
    const auto observation = rollout.observations.select(0, 0);
    const auto state = policy.infer(observation).next_state;
    require(at::equal(policy.act_recurrent(observation, state).packed,
                       restored.act_recurrent(observation, state).packed),
            "El checkpoint GRU altera el siguiente muestreo con estado");
    static_cast<void>(policy.update(rollout));
    static_cast<void>(restored.update(rollout));
    require(at::equal(policy.infer(observation, state).next_state, restored.infer(observation, state).next_state),
            "El checkpoint GRU altera la siguiente actualización");
}

void recurrent_rollout_rejects_missing_or_inconsistent_history() {
    PpoHyperparameters parameters;
    parameters.minibatch_size = ppo_sequence_length;
    parameters.epochs = 1;
    PpoPolicy policy(observation_width, parameters, seed, "cpu", default_ppo_memory_bytes, recurrent_architecture());
    auto rollout = recurrent_rollout(policy);
    const auto before = policy.infer(rollout.observations.select(0, 0));
    const auto lengths = rollout.prefix_lengths;
    rollout.prefix_lengths = at::full({1}, ppo_maximum_history + 1, at::kLong);
    rejected([&] { static_cast<void>(policy.update(rollout)); });
    rollout.prefix_lengths = lengths;
    rollout.episode_starts = at::Tensor{};
    rejected([&] { static_cast<void>(policy.update(rollout)); });
    require(at::equal(before.logits, policy.infer(rollout.observations.select(0, 0)).logits),
            "Una historia recurrente inválida cambia los parámetros");
}

std::string rewrite_checkpoint(torch::serialize::InputArchive& input, bool legacy = false) {
    torch::serialize::OutputArchive output;
    for (const auto& key : input.keys()) {
        if (legacy && (key == "architecture_kind" || key == "hidden_width")) {
            continue;
        }
        c10::IValue value;
        input.read(key, value);
        output.write(key, legacy && key == "schema_version" ? c10::IValue(int64_t{1}) : value);
    }
    std::ostringstream destination;
    output.save_to(destination);
    return std::move(destination).str();
}

void legacy_checkpoint_keeps_the_original_mlp_and_next_update() {
    PpoHyperparameters parameters;
    parameters.epochs = 1;
    PpoPolicy policy(observation_width, parameters, seed);
    const auto rollout = controlled_rollout(policy);
    static_cast<void>(policy.update(rollout));
    std::stringstream checkpoint;
    policy.save(checkpoint);
    torch::serialize::InputArchive input;
    input.load_from(checkpoint, at::Device(at::kCPU));
    std::istringstream legacy(rewrite_checkpoint(input, true));
    auto restored = PpoPolicy::load(legacy);
    require(restored.architecture() == PpoArchitecture{}, "Un checkpoint v1 deja de usar la MLP original");
    static_cast<void>(policy.update(rollout));
    static_cast<void>(restored.update(rollout));
    require(at::equal(policy.forward(rollout.observations.flatten(0, 1)).logits,
                       restored.forward(rollout.observations.flatten(0, 1)).logits),
            "La compatibilidad v1 altera el siguiente paso de Adam");
}

void recurrent_gate_order_matches_an_analytic_recurrence() {
    PpoPolicy policy(observation_width, {}, seed, "cpu", default_ppo_memory_bytes, recurrent_architecture());
    std::stringstream checkpoint;
    policy.save(checkpoint);
    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint, at::Device(at::kCPU));
    c10::IValue network;
    archive.read("network", network);
    const auto object = network.toObject();
    {
        const at::NoGradGuard no_grad;
        for (std::size_t index = 0; index < object->type()->numAttributes(); ++index) {
            object->getSlot(index).toTensor().zero_();
        }
        constexpr double candidate = 0.5;
        object->getAttr("bias_ih_l0").toTensor().select(0, 2 * default_ppo_hidden_width).fill_(std::atanh(candidate));
        auto output = object->getAttr("output_weight").toTensor();
        output.select(0, 0).select(0, 0).fill_(1);
        output.select(0, ppo_action_count).select(0, 0).fill_(1);
    }
    std::istringstream analytic(rewrite_checkpoint(archive));
    auto controlled = PpoPolicy::load(analytic);
    const auto observations = at::zeros({1, observation_width}, at::kFloat);
    constexpr std::array<float, 3> expected{0.25F, 0.375F, 0.4375F};
    auto state = controlled.initial_state(1);
    for (const auto value : expected) {
        const auto result = controlled.infer(observations, state);
        require(result.values.item<float>() == value && result.logits.select(1, 0).item<float>() == value,
                "La GRU no aplica el orden de puertas y la recurrencia declarados");
        state = result.next_state;
    }
}

void recurrent_update_respects_episode_boundaries_inside_the_rollout() {
    PpoHyperparameters parameters;
    parameters.gamma = 0;
    parameters.epochs = 1;
    parameters.minibatch_size = 2 * ppo_sequence_length;
    PpoPolicy policy(observation_width, parameters, seed, "cpu", default_ppo_memory_bytes, recurrent_architecture());
    auto rollout = recurrent_rollout(policy);
    constexpr int64_t restart = 5;
    rollout.episode_starts.select(0, restart).fill_(true);
    rollout.truncated.select(0, restart - 1).fill_(true);
    auto state = policy.initial_state(1);
    for (int64_t time = restart; time < ppo_sequence_length; ++time) {
        const auto output = policy.infer(rollout.observations.select(0, time), state);
        state = output.next_state;
        rollout.old_log_probabilities.select(0, time).copy_(output.logits.log_softmax(-1)
            .gather(1, rollout.actions.select(0, time).unsqueeze(1)).squeeze(1));
    }
    const auto result = policy.update(rollout);
    require(result.valid_transitions == ppo_sequence_length && result.minibatches == 1 &&
                result.approximate_kl < tolerance,
            "La GRU mezcla episodios o incluye el relleno de secuencias de distinta longitud");
}

void gru_prefix_padding_does_not_affect_training() {
    PpoHyperparameters parameters;
    parameters.epochs = 1;
    parameters.minibatch_size = ppo_sequence_length;
    PpoPolicy policy(observation_width, parameters, seed, "cpu", default_ppo_memory_bytes, recurrent_architecture());
    PpoPolicy control(observation_width, parameters, seed, "cpu", default_ppo_memory_bytes, recurrent_architecture());
    auto rollout = recurrent_rollout(policy);
    const auto unchanged = rollout;
    constexpr int64_t padded_prefix = 7;
    rollout.prefix_observations = at::full({padded_prefix, 1, observation_width}, altered_observation, at::kFloat);
    rollout.prefix_observations.narrow(0, 0, unchanged.prefix_observations.size(0)).copy_(unchanged.prefix_observations);
    static_cast<void>(policy.update(rollout));
    static_cast<void>(control.update(unchanged));
    require(at::equal(policy.infer(rollout.observations.select(0, 0)).logits,
                       control.infer(rollout.observations.select(0, 0)).logits),
            "Las observaciones de relleno del prefijo participan en la actualización");
}

void gae_does_not_cross_an_explicit_episode_start() {
    PpoPolicy policy(observation_width, {}, seed);
    auto rollout = controlled_rollout(policy);
    rollout.rewards = at::ones({2, 1}, at::kDouble);
    rollout.old_values = at::zeros({2, 1}, at::kDouble);
    rollout.next_values = at::zeros({2, 1}, at::kDouble);
    rollout.reward_valid = at::ones({2, 1}, at::kBool);
    rollout.terminated = at::zeros({2, 1}, at::kBool);
    rollout.truncated = at::zeros({2, 1}, at::kBool);
    rollout.episode_starts = at::ones({2, 1}, at::kBool);
    PpoHyperparameters parameters;
    parameters.gamma = 1;
    parameters.gae_lambda = 1;
    same(ppo_gae(rollout, parameters).advantages, at::ones({2, 1}, at::kDouble),
         "GAE arrastra una recompensa de un episodio posterior");
}

void architectures_and_recurrent_states_fail_fast() {
    rejected([&] { PpoPolicy invalid(observation_width, {}, seed, "cpu", default_ppo_memory_bytes,
                                     {PpoNetworkKind::mlp, 0}); });
    rejected([&] { PpoPolicy invalid(observation_width, {}, seed, "cpu", default_ppo_memory_bytes,
                                     {PpoNetworkKind::gru, 1}); });
    PpoHyperparameters parameters;
    parameters.minibatch_size = ppo_sequence_length - 1;
    rejected([&] { PpoPolicy invalid(observation_width, parameters, seed, "cpu", default_ppo_memory_bytes,
                                     recurrent_architecture()); });
    PpoPolicy policy(observation_width, {}, seed, "cpu", default_ppo_memory_bytes, recurrent_architecture());
    const auto observations = at::ones({1, observation_width}, at::kFloat);
    rejected([&] { static_cast<void>(policy.infer(observations, at::zeros({1, 1}, at::kFloat))); });
    rejected([&] { static_cast<void>(policy.infer(observations, policy.initial_state(1).to(at::kDouble))); });
    rejected([&] { static_cast<void>(policy.infer(observations, {}, at::zeros({1}, at::kByte))); });
    auto rollout = recurrent_rollout(policy);
    rollout.prefix_lengths.zero_();
    rejected([&] { static_cast<void>(policy.update(rollout)); });
    rollout.episode_starts.select(0, 0).fill_(true);
    rollout.terminated.select(0, 0).fill_(true);
    rejected([&] { static_cast<void>(policy.update(rollout)); });
}

PpoArchitecture auxiliary_architecture() { return {PpoNetworkKind::mlp, default_ppo_hidden_width, true}; }

void consolidation_reduces_scalar_mae_without_advancing_ppo() {
    PpoHyperparameters parameters;
    constexpr double auxiliary_learning_rate = 0.01;
    parameters.learning_rate = auxiliary_learning_rate;
    PpoPolicy policy(observation_width, parameters, seed, "cpu", default_ppo_memory_bytes, auxiliary_architecture());
    const auto observations = at::ones({sample_count, observation_width}, at::kFloat);
    const auto rewards = at::ones({sample_count}, at::kDouble);
    const auto random = policy.random_state();
    const auto result = policy.consolidate(observations, rewards, 1);
    require(result.samples == sample_count && result.steps == 1 && result.mae_after < result.mae_before,
            "La consolidación no reduce el MAE del objetivo escalar maduro");
    require(policy.optimizer_steps() == 0 && policy.auxiliary_steps() == 1,
            "La consolidación avanza el optimizador PPO");
    require(at::equal(random.sampling, policy.random_state().sampling) &&
                at::equal(random.shuffle, policy.random_state().shuffle),
            "La consolidación consume los generadores PPO");
    PpoPolicy disabled(observation_width, parameters, seed);
    rejected([&] { static_cast<void>(disabled.consolidate(observations, rewards)); });
}

void auxiliary_checkpoint_restores_both_optimizers_and_sampling() {
    PpoHyperparameters parameters;
    parameters.epochs = 1;
    PpoPolicy policy(observation_width, parameters, seed, "cpu", default_ppo_memory_bytes, auxiliary_architecture());
    const auto observations = at::arange(sample_count * observation_width, at::kFloat)
                                  .reshape({sample_count, observation_width}) / sample_count;
    const auto rewards = at::linspace(-1., 1., sample_count, at::kFloat);
    static_cast<void>(policy.consolidate(observations, rewards));
    const auto rollout = controlled_rollout(policy);
    static_cast<void>(policy.update(rollout));
    std::stringstream checkpoint;
    policy.save(checkpoint);
    auto restored = PpoPolicy::load(checkpoint);
    require(restored.architecture().auxiliary && restored.auxiliary_steps() == policy.auxiliary_steps(),
            "El checkpoint pierde el estado auxiliar");
    const auto first = policy.consolidate(observations, rewards);
    const auto second = restored.consolidate(observations, rewards);
    require(first.mae_after == second.mae_after, "La recuperación altera la siguiente consolidación");
    static_cast<void>(policy.update(rollout));
    static_cast<void>(restored.update(rollout));
    require(at::equal(policy.forward(observations).logits, restored.forward(observations).logits),
            "La recuperación altera la combinación de consolidación y PPO");
}

void history_reconstruction_matches_independent_lanes_without_sampling() {
    PpoPolicy policy(observation_width, {}, seed, "cpu", default_ppo_memory_bytes, recurrent_architecture());
    constexpr int64_t history_steps = 4;
    const auto observations = at::ones({history_steps, 2, observation_width}, at::kFloat);
    const auto lengths = at::tensor({history_steps, int64_t{1}}, at::kLong);
    const auto random = policy.random_state();
    const auto reconstructed = policy.state_from_history(observations, lengths);
    auto state = policy.initial_state(2);
    const auto first = policy.infer(observations.select(0, 0), state).next_state;
    for (int64_t time = 0; time < history_steps; ++time) {
        state = policy.infer(observations.select(0, time), state).next_state;
    }
    same(reconstructed.select(0, 0), state.select(0, 0), "El prefijo batched no conserva el estado de la primera lane");
    same(reconstructed.select(0, 1), first.select(0, 1), "La reconstrucción usa el relleno posterior a la longitud");
    require(at::equal(random.sampling, policy.random_state().sampling) && !reconstructed.requires_grad(),
            "La reconstrucción de historia consume RNG o conserva gradientes");
    require(at::equal(policy.state_from_history(observations, at::zeros({2}, at::kLong)), policy.initial_state(2)),
            "Una historia vacía no reconstruye el estado inicial");
    rejected([&] { static_cast<void>(policy.state_from_history(observations, lengths + history_steps)); });
}

at::Tensor stored_tensor(PpoPolicy& policy, std::string_view name, bool network = true) {
    std::stringstream checkpoint;
    policy.save(checkpoint);
    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint, at::Device(at::kCPU));
    c10::IValue value;
    archive.read(network ? "network" : std::string(name), value);
    return network ? value.toObject()->getAttr(std::string(name)).toTensor().detach().clone() :
                     value.toTensor().clone();
}

void auxiliary_and_ppo_optimizers_modify_only_their_own_heads() {
    PpoHyperparameters parameters;
    parameters.epochs = 1;
    PpoPolicy policy(observation_width, parameters, seed, "cpu", default_ppo_memory_bytes, auxiliary_architecture());
    const auto output_before = stored_tensor(policy, "output_weight");
    const auto body_before = stored_tensor(policy, "first_weight");
    const auto random_before = stored_tensor(policy, "auxiliary_rng", false);
    const auto rollout = controlled_rollout(policy);
    const auto log_probabilities = rollout.old_log_probabilities.clone();
    static_cast<void>(policy.consolidate(rollout.observations.flatten(0, 1), at::ones({sample_count}, at::kFloat)));
    require(at::equal(output_before, stored_tensor(policy, "output_weight")) &&
                !at::equal(body_before, stored_tensor(policy, "first_weight")),
            "La consolidación cambia la cabeza del actor o no actualiza el tronco compartido");
    require(!at::equal(random_before, stored_tensor(policy, "auxiliary_rng", false)),
            "La consolidación no usa un generador auxiliar independiente");
    const auto auxiliary_before = stored_tensor(policy, "auxiliary_weight");
    static_cast<void>(policy.update(rollout));
    require(at::equal(auxiliary_before, stored_tensor(policy, "auxiliary_weight")) && policy.auxiliary_steps() == 1,
            "PPO actualiza la cabeza o el contador del objetivo auxiliar");
    require(at::equal(log_probabilities, rollout.old_log_probabilities),
            "La consolidación o PPO reescriben las probabilidades antiguas del rollout");
}

void consolidation_rejects_invalid_targets_before_consuming_rng() {
    PpoPolicy policy(observation_width, {}, seed, "cpu", default_ppo_memory_bytes, auxiliary_architecture());
    const auto observations = at::ones({sample_count, observation_width}, at::kFloat);
    const auto random_before = stored_tensor(policy, "auxiliary_rng", false);
    const auto logits = policy.forward(observations).logits;
    const auto invalid = at::full({sample_count}, std::numeric_limits<double>::quiet_NaN(), at::kDouble);
    rejected([&] { static_cast<void>(policy.consolidate(observations, invalid)); });
    rejected([&] { static_cast<void>(policy.consolidate(observations, at::ones({sample_count}, at::kFloat), 0)); });
    require(at::equal(logits, policy.forward(observations).logits) &&
                at::equal(random_before, stored_tensor(policy, "auxiliary_rng", false)),
            "Un objetivo auxiliar inválido consume RNG o altera la política");
}

void model_size_is_bounded_by_checkpoint_capacity() {
    constexpr std::size_t large_observation = 32768;
    constexpr int64_t large_hidden = 1024;
    constexpr std::size_t larger_memory = std::size_t{2} * 1024 * 1024 * 1024;
    rejected([&] { PpoPolicy invalid(large_observation, {}, seed, "cpu", larger_memory,
                                     {PpoNetworkKind::mlp, large_hidden}); });
}

PpoArchitecture dqn_architecture() {
    return {PpoNetworkKind::mlp, default_ppo_hidden_width, false, true};
}

DqnBatch dqn_batch() {
    constexpr float following_value = 0.5;
    return {at::ones({sample_count, observation_width}, at::kFloat),
        at::full({sample_count, observation_width}, following_value, at::kFloat),
        at::zeros({sample_count}, at::kLong), at::ones({sample_count}, at::kFloat),
        at::zeros({sample_count}, at::kBool), at::ones({sample_count}, at::kBool)};
}

void double_dqn_uses_online_choice_and_target_value() {
    constexpr double discount = 0.5;
    const auto online = at::tensor({1., 4., 2., 3., 0., -1., 4., 2., 1., 0., -1., -2.}, at::kDouble).reshape({2, 6});
    const auto target = at::tensor({10., 3., 8., 6., 4., 5., 10., 3., 8., 6., 4., 5.}, at::kDouble).reshape({2, 6});
    const auto rewards = at::ones({2}, at::kDouble);
    const auto terminal = at::tensor({0, 1}, at::kLong).to(at::kBool);
    const auto expected = at::tensor({2.5, 1.}, at::kDouble);
    same(double_dqn_targets(rewards, terminal, online, target, discount), expected,
         "Double DQN usa el máximo de la red objetivo o conserva bootstrap en un terminal");
}

void double_dqn_targets_keep_exact_values_with_mixed_precision_and_strides() {
    constexpr double discount = 0.5;
    const auto terminal = at::tensor({0, 1}, at::kLong).to(at::kBool);
    for (const auto reward_type : {at::kFloat, at::kDouble}) {
        for (const auto online_type : {at::kFloat, at::kDouble}) {
            for (const auto target_type : {at::kFloat, at::kDouble}) {
                const auto rewards = at::ones({4}, reward_type).slice(0, 0, 4, 2);
                const auto online = at::tensor({1., 4., 4., 2., 2., 1., 3., 0., 0., -1., -1., -2.},
                                                online_type).reshape({6, 2}).transpose(0, 1);
                const auto target = at::tensor({10., 10., 3., 3., 8., 8., 6., 6., 4., 4., 5., 5.},
                                                target_type).reshape({6, 2}).transpose(0, 1);
                const auto expected_type = reward_type == at::kDouble || target_type == at::kDouble
                    ? at::kDouble : at::kFloat;
                const auto expected = at::tensor({2.5, 1.}, expected_type);
                const auto actual = double_dqn_targets(rewards, terminal, online, target, discount);
                require(actual.scalar_type() == expected_type && at::equal(actual, expected),
                        "La validación altera el objetivo Double DQN con tipos mixtos o vistas no contiguas");
            }
        }
    }
}

void double_dqn_targets_reject_nonfinite_and_malformed_inputs() {
    constexpr double discount = 0.5;
    const auto terminal = at::ones({2}, at::kBool);
    for (const auto dtype : {at::kFloat, at::kDouble}) {
        const auto rewards = at::ones({2}, dtype);
        const auto online = at::ones({2, ppo_action_count}, dtype);
        const auto target = at::ones_like(online);
        for (const auto invalid : {std::numeric_limits<double>::quiet_NaN(),
                                   std::numeric_limits<double>::infinity(),
                                   -std::numeric_limits<double>::infinity()}) {
            for (std::size_t field = 0; field < 3; ++field) {
                auto values = std::array{rewards.clone(), online.clone(), target.clone()};
                values.at(field).flatten().select(0, values.at(field).numel() - 1).fill_(invalid);
                rejected([&] {
                    static_cast<void>(double_dqn_targets(values.at(0), terminal, values.at(1),
                                                         values.at(2), discount));
                });
            }
        }
        for (const auto& invalid : {at::Tensor{}, at::empty({0}, dtype), rewards.unsqueeze(1),
                                    rewards.to(at::kLong)}) {
            rejected([&] { static_cast<void>(double_dqn_targets(invalid, terminal, online, target, discount)); });
        }
        for (const auto& invalid : {at::Tensor{}, online.narrow(1, 0, ppo_action_count - 1),
                                    online.to(at::kLong),
                                    at::empty(online.sizes(), online.options().device(at::kMeta))}) {
            rejected([&] { static_cast<void>(double_dqn_targets(rewards, terminal, invalid, target, discount)); });
            rejected([&] { static_cast<void>(double_dqn_targets(rewards, terminal, online, invalid, discount)); });
        }
        for (const auto& invalid : {at::Tensor{}, terminal.to(at::kByte), terminal.unsqueeze(1)}) {
            rejected([&] { static_cast<void>(double_dqn_targets(rewards, invalid, online, target, discount)); });
        }
    }
}

void empty_mlp_state_preserves_inference_and_still_validates_its_schema() {
    PpoPolicy policy(observation_width, {}, seed, "cpu");
    const auto observations = at::ones({2, observation_width}, at::kFloat);
    const auto empty = policy.initial_state(2);
    const auto expected = policy.infer(observations);
    const auto actual = policy.infer(observations, empty);
    require(empty.numel() == 0 && at::equal(actual.logits, expected.logits) &&
                at::equal(actual.values, expected.values) && at::equal(actual.next_state, empty),
            "El estado vacío de la MLP altera su inferencia");
    for (const auto& invalid : {at::empty({0, 2}, at::kFloat), empty.to(at::kDouble),
                                at::empty({2, 0}, empty.options().device(at::kMeta))}) {
        rejected([&] { static_cast<void>(policy.infer(observations, invalid)); });
    }
}

void epsilon_greedy_probabilities_and_packed_values_match_q_values() {
    PpoPolicy policy(observation_width, {}, seed, "cpu", default_ppo_memory_bytes, dqn_architecture());
    const auto observations = at::ones({sample_count, observation_width}, at::kFloat);
    const auto scores = policy.forward(observations).logits;
    const auto greedy = scores.argmax(1);
    constexpr double epsilon = 0.3;
    const auto action = policy.act_double_dqn(observations, epsilon);
    auto expected = at::full({sample_count, ppo_action_count}, epsilon / ppo_action_count, at::kDouble);
    expected.scatter_add_(1, greedy.unsqueeze(1), at::full({sample_count, 1}, 1 - epsilon, at::kDouble));
    same(action.probabilities, expected, "La distribución epsilon-greedy no es la mezcla uniforme acordada");
    same(action.packed.select(1, 2), std::get<0>(scores.max(1)).to(at::kDouble),
         "La acción Double DQN no conserva maxQ en el lote");
    const auto sampled = action.packed.select(1, 0).to(at::kLong);
    same(action.packed.select(1, 1), expected.gather(1, sampled.unsqueeze(1)).squeeze(1).log(),
         "La logprobabilidad Double DQN no corresponde a la acción muestreada");
    require(at::equal(policy.act_double_dqn(observations, 0).packed.select(1, 0).to(at::kLong), greedy),
            "La selección epsilon cero no es voraz");
    constexpr std::size_t reference_parameters = 4870;
    require(policy.parameter_count() == reference_parameters, "Double DQN conserva una cabeza de valor sin uso");
}

void double_dqn_update_syncs_at_environment_steps_and_restores_exactly() {
    PpoPolicy policy(observation_width, {}, seed, "cpu", default_ppo_memory_bytes, dqn_architecture());
    const auto batch = dqn_batch();
    const auto first = policy.update_double_dqn(batch, dqn_warmup_steps);
    require(first.valid_transitions == sample_count && first.updates == 1 && first.target_synchronized &&
                policy.target_sync_step() == dqn_warmup_steps,
            "La primera actualización Double DQN no sigue el cursor de sincronización de la referencia");
    const auto second = policy.update_double_dqn(batch, dqn_warmup_steps + 1);
    require(!second.target_synchronized && policy.optimizer_steps() == 2 &&
                policy.target_sync_step() == dqn_warmup_steps,
            "La red objetivo Double DQN se sincroniza fuera de su intervalo");
    std::stringstream checkpoint;
    policy.save(checkpoint);
    auto restored = PpoPolicy::load(checkpoint);
    require(restored.architecture().double_dqn && restored.dqn_environment_step() == dqn_warmup_steps + 1 &&
                restored.target_sync_step() == dqn_warmup_steps,
            "El checkpoint Double DQN pierde el algoritmo o los cursores de la red objetivo");
    const auto expected_action = policy.act_double_dqn(batch.observations, default_ppo_clip);
    const auto actual_action = restored.act_double_dqn(batch.observations, default_ppo_clip);
    require(at::equal(expected_action.packed, actual_action.packed), "El checkpoint altera la exploración Double DQN");
    const auto expected = policy.update_double_dqn(batch, dqn_warmup_steps + 2);
    const auto actual = restored.update_double_dqn(batch, dqn_warmup_steps + 2);
    require(expected.loss == actual.loss &&
                at::equal(policy.forward(batch.observations).logits, restored.forward(batch.observations).logits),
            "La recuperación Double DQN altera el target o el siguiente paso Adam");
}

void double_dqn_excludes_invalid_rewards_and_rejects_cursor_changes() {
    PpoPolicy policy(observation_width, {}, seed, "cpu", default_ppo_memory_bytes, dqn_architecture());
    auto batch = dqn_batch();
    const auto before = policy.forward(batch.observations).logits;
    batch.reward_valid.zero_();
    const auto empty = policy.update_double_dqn(batch, dqn_warmup_steps);
    require(empty.updates == 0 && policy.optimizer_steps() == 0 && policy.dqn_environment_step() == 0 &&
                at::equal(before, policy.forward(batch.observations).logits),
            "Una recompensa inválida actualiza Double DQN o su cursor");
    batch.reward_valid.fill_(true);
    static_cast<void>(policy.update_double_dqn(batch, dqn_warmup_steps));
    rejected([&] { static_cast<void>(policy.update_double_dqn(batch, dqn_warmup_steps)); });
    rejected([&] { static_cast<void>(policy.update_double_dqn(batch, dqn_warmup_steps + 1, 1)); });
    rejected([&] { static_cast<void>(policy.update(controlled_rollout(policy))); });
    rejected([&] { static_cast<void>(policy.act_recurrent(batch.observations)); });
    rejected([&] { static_cast<void>(policy.act_double_dqn(batch.observations, 2)); });
}

void double_dqn_smooth_l1_matches_an_analytic_target() {
    PpoHyperparameters parameters;
    constexpr double discount = 0.5;
    parameters.gamma = discount;
    PpoPolicy policy(observation_width, parameters, seed, "cpu", default_ppo_memory_bytes, dqn_architecture());
    std::stringstream checkpoint;
    policy.save(checkpoint);
    torch::serialize::InputArchive archive;
    archive.load_from(checkpoint, at::Device(at::kCPU));
    const auto online_scores = at::tensor({1.F, 4.F, 2.F, 3.F, 0.F, -1.F}, at::kFloat);
    const auto target_scores = at::tensor({10.F, 3.F, 8.F, 6.F, 4.F, 5.F}, at::kFloat);
    {
        const at::NoGradGuard no_grad;
        for (const auto name : {"network", "target_network"}) {
            c10::IValue network;
            archive.read(name, network);
            const auto object = network.toObject();
            for (std::size_t index = 0; index < object->type()->numAttributes(); ++index) {
                object->getSlot(index).toTensor().zero_();
            }
            object->getAttr("output_bias").toTensor().copy_(std::string_view(name) == "network" ?
                                                            online_scores : target_scores);
        }
    }
    std::istringstream fixture(rewrite_checkpoint(archive));
    auto controlled = PpoPolicy::load(fixture);
    DqnBatch batch{at::zeros({2, observation_width}, at::kFloat), at::zeros({2, observation_width}, at::kFloat),
        at::tensor({0, 2}, at::kLong), at::ones({2}, at::kFloat), at::zeros({2}, at::kBool),
        at::ones({2}, at::kBool)};
    constexpr double expected_loss = 0.5625;
    constexpr double expected_squared_gradient = 0.3125;
    constexpr double gradient_tolerance = 1e-6;
    const auto result = controlled.update_double_dqn(batch, dqn_warmup_steps + 1);
    require(result.loss == expected_loss &&
                std::abs(result.gradient_norm - std::sqrt(expected_squared_gradient)) < gradient_tolerance,
            "La pérdida o el gradiente Double DQN no corresponden a SmoothL1 con el objetivo desacoplado");
    require(!result.target_synchronized && controlled.target_sync_step() == 0,
            "La red objetivo se sincroniza al alcanzar un número de updates distinto del cursor de entorno");
}

void double_dqn_mask_matches_a_batch_containing_only_valid_rows() {
    PpoPolicy policy(observation_width, {}, seed, "cpu", default_ppo_memory_bytes, dqn_architecture());
    PpoPolicy control(observation_width, {}, seed, "cpu", default_ppo_memory_bytes, dqn_architecture());
    auto batch = dqn_batch();
    constexpr int64_t valid_rows = sample_count / 2;
    batch.reward_valid.narrow(0, valid_rows, valid_rows).zero_();
    batch.rewards.narrow(0, valid_rows, valid_rows).fill_(invalid_reward);
    const DqnBatch filtered{batch.observations.narrow(0, 0, valid_rows),
        batch.next_observations.narrow(0, 0, valid_rows), batch.actions.narrow(0, 0, valid_rows),
        batch.rewards.narrow(0, 0, valid_rows), batch.terminated.narrow(0, 0, valid_rows),
        batch.reward_valid.narrow(0, 0, valid_rows)};
    const auto actual = policy.update_double_dqn(batch, dqn_warmup_steps);
    const auto expected = control.update_double_dqn(filtered, dqn_warmup_steps);
    require(actual.loss == expected.loss &&
                at::equal(policy.forward(batch.observations).logits, control.forward(batch.observations).logits),
            "Las filas inválidas participan en la actualización Double DQN");
}

void double_dqn_checkpoint_rejects_an_incoherent_sync_cursor() {
    PpoPolicy policy(observation_width, {}, seed, "cpu", default_ppo_memory_bytes, dqn_architecture());
    static_cast<void>(policy.update_double_dqn(dqn_batch(), dqn_warmup_steps));
    std::stringstream checkpoint;
    policy.save(checkpoint);
    torch::serialize::InputArchive input;
    input.load_from(checkpoint, at::Device(at::kCPU));
    torch::serialize::OutputArchive output;
    for (const auto& key : input.keys()) {
        c10::IValue value;
        input.read(key, value);
        output.write(key, key == "target_sync_step" ? c10::IValue(static_cast<int64_t>(dqn_warmup_steps + 1)) : value);
    }
    std::stringstream corrupted;
    output.save_to(corrupted);
    rejected([&] { static_cast<void>(PpoPolicy::load(corrupted)); });
}

void cpu_rollout_update_matches_existing_precision_rng_and_steps() {
    constexpr int64_t horizon = 3;
    PpoHyperparameters parameters;
    parameters.epochs = 1;
    parameters.minibatch_size = sample_count / 2;
    const std::array architectures{PpoArchitecture{}, auxiliary_architecture(),
        PpoArchitecture{PpoNetworkKind::mlp, default_ppo_hidden_width / 2}};
    for (const auto& architecture : architectures) {
        for (const auto dtype : {at::kFloat, at::kDouble}) {
            PpoPolicy original(observation_width, parameters, seed, "cpu", default_ppo_memory_bytes, architecture);
            PpoPolicy candidate(observation_width, parameters, seed, "cpu", default_ppo_memory_bytes, architecture);
            for (int update = 0; update < 2; ++update) {
                auto rollout = controlled_rollout(original);
                rollout.observations = rollout.observations.repeat({horizon, 1, 1});
                const std::array fields{&rollout.actions, &rollout.old_log_probabilities, &rollout.old_values,
                    &rollout.rewards, &rollout.next_values, &rollout.reward_valid, &rollout.terminated, &rollout.truncated};
                for (auto* field : fields) {
                    *field = field->repeat({horizon, 1});
                }
                for (auto* field : {&rollout.old_log_probabilities, &rollout.old_values, &rollout.rewards,
                                   &rollout.next_values}) {
                    *field = field->to(dtype);
                }
                rollout.terminated.zero_();
                rollout.terminated.select(0, horizon - 1).fill_(true);
                rollout.truncated[1][0].fill_(true);
                rollout.reward_valid.select(0, 1).narrow(0, 0, sample_count / 2).zero_();
                rollout.rewards.select(0, 1).narrow(0, 0, sample_count / 2).fill_(invalid_reward);
                const auto expected = original.update(rollout);
                const auto actual = candidate.update_from_cpu(rollout);
                const std::array expected_metrics{expected.policy_loss, expected.value_loss, expected.entropy,
                    expected.approximate_kl, expected.clip_fraction, expected.gradient_norm};
                const std::array actual_metrics{actual.policy_loss, actual.value_loss, actual.entropy,
                    actual.approximate_kl, actual.clip_fraction, actual.gradient_norm};
                require(expected.valid_transitions == actual.valid_transitions &&
                            expected.minibatches == actual.minibatches && actual_metrics == expected_metrics &&
                            original.optimizer_steps() == candidate.optimizer_steps(),
                        "La preparación CPU altera estadísticas o pasos del optimizador PPO");
                const auto before = original.random_state();
                const auto after = candidate.random_state();
                require(at::equal(before.sampling, after.sampling) && at::equal(before.shuffle, after.shuffle),
                        "La preparación CPU altera los generadores de la política");
                for (const auto name : {"first_weight", "first_bias", "second_weight", "second_bias",
                                        "output_weight", "output_bias"}) {
                    require(at::equal(stored_tensor(original, name), stored_tensor(candidate, name)),
                            "La preparación CPU altera los parámetros aprendidos");
                }
            }
        }
    }
}

void cpu_rollout_update_rejects_invalid_rows_and_other_architectures() {
    PpoPolicy policy(observation_width, {}, seed, "cpu");
    auto rollout = controlled_rollout(policy);
    rollout.reward_valid.zero_();
    const auto before = policy.random_state();
    const auto empty = policy.update_from_cpu(rollout);
    require(empty.valid_transitions == 0 && empty.minibatches == 0 && policy.optimizer_steps() == 0,
            "La preparación CPU actualiza una política sin etiquetas válidas");
    rollout.rewards.fill_(std::numeric_limits<double>::quiet_NaN());
    rejected([&] { static_cast<void>(policy.update_from_cpu(rollout)); });
    rollout.rewards.zero_();
    rollout.observations[0][0][0].fill_(std::numeric_limits<float>::infinity());
    rejected([&] { static_cast<void>(policy.update_from_cpu(rollout)); });
    rollout.observations.zero_();
    rollout.rewards = at::empty(rollout.rewards.sizes(), rollout.rewards.options().device(at::kMeta));
    rejected([&] { static_cast<void>(policy.update_from_cpu(rollout)); });
    rollout.rewards = at::zeros({1, sample_count}, at::kDouble);
    rollout.actions = at::zeros({sample_count}, at::kLong);
    rejected([&] { static_cast<void>(policy.update_from_cpu(rollout)); });
    const auto after = policy.random_state();
    require(policy.optimizer_steps() == 0 && at::equal(before.sampling, after.sampling) &&
                at::equal(before.shuffle, after.shuffle),
            "Una preparación CPU inválida altera el optimizador o los generadores");
    PpoPolicy recurrent(observation_width, {}, seed, "cpu", default_ppo_memory_bytes, recurrent_architecture());
    const auto recurrent_data = recurrent_rollout(recurrent);
    rejected([&] { static_cast<void>(recurrent.update_from_cpu(recurrent_data)); });
    PpoPolicy dqn(observation_width, {}, seed, "cpu", default_ppo_memory_bytes, dqn_architecture());
    rejected([&] { static_cast<void>(dqn.update_from_cpu(controlled_rollout(dqn))); });
}
}

int main() {
    try {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        clipped_objective_matches_signed_analytic_values();
        clipped_objective_keeps_large_finite_log_ratios();
        gae_cuts_each_lane_at_the_correct_boundary();
        inference_uses_only_its_own_observation();
        update_increases_probability_for_a_positive_advantage();
        constant_advantages_leave_actor_unchanged();
        invalid_rewards_do_not_update_actor_critic_or_rng();
        masked_transitions_match_a_rollout_containing_only_valid_steps();
        policy_sampling_does_not_consume_global_random_state();
        random_state_restoration_is_atomic();
        checkpoint_preserves_sampling_optimizer_and_next_update();
        malformed_or_different_device_checkpoints_are_rejected();
        invalid_rollout_fails_before_mutation();
        non_finite_float_gradients_fail_before_the_optimizer_step();
        hyperparameters_reject_non_finite_values_and_invalid_ranges();
        checkpoint_rejects_inconsistent_adam_steps();
        architectures_count_parameters_and_keep_legacy_defaults();
        recurrent_inference_retains_history_and_resets_only_selected_lanes();
        recurrent_update_reconstructs_prefix_after_each_epoch();
        recurrent_update_replays_recorded_probabilities_and_excludes_padding();
        recurrent_checkpoint_restores_architecture_state_and_next_update();
        recurrent_rollout_rejects_missing_or_inconsistent_history();
        legacy_checkpoint_keeps_the_original_mlp_and_next_update();
        recurrent_gate_order_matches_an_analytic_recurrence();
        recurrent_update_respects_episode_boundaries_inside_the_rollout();
        gru_prefix_padding_does_not_affect_training();
        gae_does_not_cross_an_explicit_episode_start();
        architectures_and_recurrent_states_fail_fast();
        consolidation_reduces_scalar_mae_without_advancing_ppo();
        auxiliary_checkpoint_restores_both_optimizers_and_sampling();
        history_reconstruction_matches_independent_lanes_without_sampling();
        auxiliary_and_ppo_optimizers_modify_only_their_own_heads();
        consolidation_rejects_invalid_targets_before_consuming_rng();
        model_size_is_bounded_by_checkpoint_capacity();
        double_dqn_uses_online_choice_and_target_value();
        double_dqn_targets_keep_exact_values_with_mixed_precision_and_strides();
        double_dqn_targets_reject_nonfinite_and_malformed_inputs();
        empty_mlp_state_preserves_inference_and_still_validates_its_schema();
        epsilon_greedy_probabilities_and_packed_values_match_q_values();
        double_dqn_update_syncs_at_environment_steps_and_restores_exactly();
        double_dqn_excludes_invalid_rewards_and_rejects_cursor_changes();
        double_dqn_smooth_l1_matches_an_analytic_target();
        double_dqn_mask_matches_a_batch_containing_only_valid_rows();
        double_dqn_checkpoint_rejects_an_incoherent_sync_cursor();
        cpu_rollout_update_matches_existing_precision_rng_and_steps();
        cpu_rollout_update_rejects_invalid_rows_and_other_architectures();
        std::cout << "PPO nativo: pruebas analíticas y de recuperación correctas\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
