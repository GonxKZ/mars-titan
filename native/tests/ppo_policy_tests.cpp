#include "mars_titan/ppo_policy.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Parallel.h>
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
    const auto before = policy.forward(rollout.observations.flatten(0, 1));
    const auto result = policy.update(rollout);
    const auto after = policy.forward(rollout.observations.flatten(0, 1));
    require(result.valid_transitions == 0 && result.minibatches == 0,
            "Se entrenó con recompensas inválidas");
    same(after.logits, before.logits, "Las recompensas inválidas cambian el actor");
    same(after.values, before.values, "Las recompensas inválidas cambian el crítico");
    same(policy.act(rollout.observations.flatten(0, 1)),
         control.act(rollout.observations.flatten(0, 1)),
         "Una actualización vacía consume el estado de muestreo");
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
        std::cout << "PPO nativo: pruebas analíticas y de recuperación correctas\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
