#include "mars_titan/ppo_training.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/grad_mode.h>
#include <torch/serialize/archive.h>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace {
using namespace mars_titan::learning;
using namespace mars_titan::simulation;
constexpr std::size_t short_sessions = 4;
constexpr std::size_t long_sessions = 7;
constexpr std::size_t total_transitions = 24;
constexpr std::size_t rollout_transitions = 8;
constexpr double price_base = 10;
constexpr double volume = 10000;
constexpr double prediction = 0.01;
constexpr std::size_t digest_size = 64;
constexpr std::size_t observation_width = 8;
constexpr std::size_t full_exposure = 5;
constexpr std::size_t paused_transitions = 6;

void require(bool condition, const char* reason) {
    if (!condition) {
        throw std::runtime_error(reason);
    }
}
template<class Function> void rejected(Function&& function) {
    try { std::forward<Function>(function)(); }
    catch (const std::exception&) { return; }
    throw std::runtime_error("La entrada inválida se ha aceptado");
}

std::shared_ptr<MarketTape> tape(std::size_t count, std::string partition = "train", char hash = 'a') {
    auto data = std::make_shared<MarketTape>();
    data->assets = {"FIC0"};
    data->currency = "USD";
    data->domain = "synthetic";
    data->partition = std::move(partition);
    data->source_sha256.assign(digest_size, hash);
    data->parent_id = "analytic-frozen";
    for (std::size_t at = 0; at < count; ++at) {
        data->open_times.push_back(static_cast<int64_t>(2 * at));
        data->close_times.push_back(static_cast<int64_t>(2 * at + 1));
        data->prediction_times.push_back(static_cast<int64_t>(2 * at + 1));
        const auto price = price_base + static_cast<double>(at);
        data->prices.insert(data->prices.end(), {price, price, price, price, volume});
        data->scores.push_back(prediction);
    }
    return data;
}

std::vector<BatchInput> inputs() {
    return {{tape(short_sessions), {}, {}}, {tape(long_sessions), {}, {}}};
}

PpoTrainingConfig config() {
    PpoTrainingConfig result;
    result.total_transitions = total_transitions;
    result.rollout_transitions = rollout_transitions;
    return result;
}

PpoHyperparameters hyperparameters() {
    PpoHyperparameters result;
    result.epochs = 2;
    result.minibatch_size = 4;
    return result;
}

void paused_rollout_resumes_exactly_and_is_owned() {
    PpoTrainer continuous(inputs(), config(), hyperparameters(), "cpu", true);
    PpoTrainer interrupted(inputs(), config(), hyperparameters(), "cpu", true);
    require(interrupted.advance() && interrupted.advance() && interrupted.advance(),
            "Faltan decisiones antes de la pausa");
    const auto saved = interrupted.snapshot();
    require(saved.transitions == paused_transitions && interrupted.partial_ticks() == 3 &&
                !saved.reset_lanes.empty(), "La pausa debe conservar un rollout parcial y su reset pendiente");
    const auto first_observation = saved.rollout.observations[0].clone();
    static_cast<void>(interrupted.advance());
    require(at::equal(first_observation, saved.rollout.observations[0]),
            "El snapshot no puede conservar vistas de buffers reutilizados");
    PpoTrainer restored(inputs(), config(), hyperparameters(), "cpu", true);
    restored.restore(saved);
    while (continuous.advance()) {}
    while (restored.advance()) {}
    const auto first = continuous.snapshot();
    const auto second = restored.snapshot();
    require(first.transitions == total_transitions && first.optimizer_steps == second.optimizer_steps &&
                first.episodes == second.episodes && first.reset_lanes == second.reset_lanes,
            "La recuperación cambia contadores de aprendizaje");
    const auto observation = at::ones({2, static_cast<int64_t>(observation_width)}, at::kFloat);
    require(at::equal(continuous.policy().forward(observation).logits,
                      restored.policy().forward(observation).logits),
            "La recuperación cambia la actualización siguiente");
    for (std::size_t lane = 0; lane < first.environment.sessions.size(); ++lane) {
        require(first.environment.sessions[lane].cursor == second.environment.sessions[lane].cursor &&
                    first.environment.sessions[lane].account.nav == second.environment.sessions[lane].account.nav,
                "La cartera recuperada diverge");
    }
    const auto encoded = serialize_rollout(saved.rollout);
    require(at::equal(deserialize_rollout(encoded).observations, saved.rollout.observations),
            "La serialización pierde observaciones del rollout");
}

void inputs_and_corrupt_recovery_fail_before_mutation() {
    auto mixed = inputs();
    auto different = tape(short_sessions);
    different->parent_id = "another-parent";
    mixed[1].tape = different;
    rejected([&] { PpoTrainer bad(mixed, config(), hyperparameters(), "cpu", true); });
    rejected([] { PpoTrainer bad(inputs(), config(), hyperparameters(), "cpu", false); });
    auto large = config();
    large.total_transitions *= 2;
    rejected([&] { PpoTrainer bad(inputs(), large, hyperparameters(), "cpu", true); });
    PpoTrainer trainer(inputs(), config(), hyperparameters(), "cpu", true);
    const auto initial_environment = trainer.snapshot().environment;
    static_cast<void>(trainer.advance());
    const auto before = trainer.snapshot();
    auto wrong = before;
    wrong.environment.sessions.back().account.nav += 1;
    rejected([&] { trainer.restore(wrong); });
    require(trainer.snapshot().transitions == before.transitions &&
                at::equal(trainer.snapshot().rollout.actions, before.rollout.actions),
            "Un snapshot corrupto modifica el entrenamiento confirmado");
    wrong = before;
    wrong.rollout.observations = at::empty({1, 2, 1}, at::kFloat);
    rejected([&] { trainer.restore(wrong); });
    wrong = before;
    wrong.environment = initial_environment;
    wrong.episodes = 1;
    rejected([&] { trainer.restore(wrong); });
}

void evaluation_keeps_training_rng_and_independent_endings() {
    PpoTrainer trainer(inputs(), config(), hyperparameters(), "cpu", true);
    const auto before = trainer.snapshot().policy_archive;
    const auto result = evaluate_policy(trainer.policy(),
        {{tape(short_sessions, "validation", 'b'), {}, {}},
         {tape(long_sessions, "validation", 'c'), {}, {}}}, 2);
    require(result.episodes == 2 && result.incomplete == 0 && result.metrics[0].steps == 3 &&
                result.metrics[1].steps == long_sessions - 1, "La evaluación mezcla los horizontes de sus entornos");
    std::istringstream original(before);
    auto policy = PpoPolicy::load(original, "cpu");
    const auto expected = policy.act(at::ones({2, static_cast<int64_t>(observation_width)}, at::kFloat));
    std::istringstream after(trainer.snapshot().policy_archive);
    auto actual = PpoPolicy::load(after, "cpu");
    require(at::equal(expected, actual.act(at::ones({2, static_cast<int64_t>(observation_width)}, at::kFloat))),
            "La evaluación ha consumido el RNG de entrenamiento");
}

template<class Edit> std::string edit_network(const std::string& bytes, Edit&& edit) {
    std::istringstream source(bytes);
    torch::serialize::InputArchive input;
    input.load_from(source, at::Device(at::kCPU));
    c10::IValue network;
    input.read("network", network);
    {
        const at::NoGradGuard guard;
        std::forward<Edit>(edit)(network.toObject());
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

void nonfinite_critic_never_enters_a_confirmed_rollout() {
    PpoTrainer trainer(inputs(), config(), hyperparameters(), "cpu", true);
    auto state = trainer.snapshot();
    state.policy_archive = edit_network(state.policy_archive, [](const auto& network) {
        network->getAttr("second_weight").toTensor().zero_();
        network->getAttr("second_bias").toTensor().fill_(1);
        network->getAttr("output_weight").toTensor().select(0, ppo_action_count)
            .fill_(std::numeric_limits<float>::max());
    });
    trainer.restore(state);
    const auto rng = trainer.policy().random_state();
    rejected([&] { static_cast<void>(trainer.advance()); });
    const auto after = trainer.snapshot();
    require(after.transitions == 0 && after.environment.sessions[0].cursor == 0 &&
                at::isfinite(after.rollout.next_values).all().template item<bool>() &&
                at::equal(rng.sampling, trainer.policy().random_state().sampling),
            "Un valor no finito se confirmó o consumió el muestreo");
    trainer.restore(after);
    rejected([&] {
        static_cast<void>(evaluate_policy(trainer.policy(),
            {{tape(short_sessions, "validation", 'b'), {}, {}}}, 1));
    });
}

void budget_cannot_rewind_without_the_environment() {
    PpoTrainer trainer(inputs(), config(), hyperparameters(), "cpu", true);
    static_cast<void>(trainer.advance());
    auto state = trainer.snapshot();
    state.transitions = 0;
    for (auto* tensor : {&state.rollout.observations, &state.rollout.actions,
                        &state.rollout.old_log_probabilities, &state.rollout.old_values,
                        &state.rollout.rewards, &state.rollout.next_values, &state.rollout.reward_valid,
                        &state.rollout.terminated, &state.rollout.truncated}) {
        *tensor = tensor->narrow(0, 0, 0);
    }
    rejected([&] { trainer.restore(state); });
    require(trainer.transitions() == 2, "El contador retrocedió pese al rechazo");
}

void bootstrap_failure_rolls_back_the_environment_and_sampler() {
    auto market = tape(short_sessions);
    ContextTape context;
    context.source_sha256.assign(digest_size, 'b');
    context.fields.push_back({"evento_controlado", "indicador"});
    for (std::size_t session = 0; session < short_sessions; ++session) {
        context.values.push_back({session == 0 ? 0.0F : 1.0F, true, market->close_times[session]});
    }
    PpoTrainer trainer({{market, {}, context}}, config(), hyperparameters(), "cpu", true);
    auto state = trainer.snapshot();
    state.policy_archive = edit_network(state.policy_archive, [](const auto& network) {
        for (const auto* name : {"first_weight", "first_bias", "second_weight", "second_bias",
                                 "output_weight", "output_bias"}) {
            network->getAttr(name).toTensor().zero_();
        }
        network->getAttr("first_weight").toTensor()[0][observation_width].fill_(1);
        network->getAttr("second_weight").toTensor().select(1, 0).fill_(1);
        network->getAttr("output_weight").toTensor().select(0, ppo_action_count)
            .fill_(std::numeric_limits<float>::max());
    });
    trainer.restore(state);
    const auto rng = trainer.policy().random_state();
    rejected([&] { static_cast<void>(trainer.advance()); });
    const auto after = trainer.snapshot();
    require(after.transitions == 0 && after.environment.sessions[0].cursor == 0 &&
                after.environment.sessions[0].account.nav == state.environment.sessions[0].account.nav &&
                at::equal(rng.sampling, trainer.policy().random_state().sampling),
            "El bootstrap fallido confirmó la cartera o consumió el RNG");
    trainer.restore(after);
}

void partial_optimizer_failure_requires_a_confirmed_checkpoint() {
    auto budget = config();
    budget.total_transitions = 4;
    budget.rollout_transitions = 4;
    auto parameters = hyperparameters();
    parameters.epochs = 1;
    parameters.minibatch_size = 1;
    parameters.gamma = 0;
    parameters.gae_lambda = 0;
    PpoTrainer trainer({{tape(long_sessions), {}, {}}}, budget, parameters, "cpu", true);
    for (std::size_t step = 0; step < budget.total_transitions - 1; ++step) {
        static_cast<void>(trainer.advance());
    }
    const auto confirmed = trainer.snapshot();
    auto altered = trainer.snapshot();
    auto generator = at::detail::createCPUGenerator(0);
    generator.set_state(trainer.policy().random_state().shuffle);
    const auto order = at::randperm(4, generator, at::TensorOptions().dtype(at::kLong));
    const auto position = order[3].item<int64_t>() == 3 ? order[2].item<int64_t>()
                                                     : order[3].item<int64_t>();
    constexpr double extreme_log_probability = -100;
    constexpr double negative_reward = -10;
    altered.rollout.old_log_probabilities[position].fill_(extreme_log_probability);
    altered.rollout.rewards[position].fill_(negative_reward);
    trainer.restore(altered);
    rejected([&] { static_cast<void>(trainer.advance()); });
    require(trainer.policy().optimizer_steps() > 0, "La prueba necesita un Adam parcialmente aplicado");
    rejected([&] { static_cast<void>(trainer.snapshot()); });
    rejected([&] { static_cast<void>(trainer.advance()); });
    trainer.restore(confirmed);
    while (trainer.advance()) {}
    require(trainer.transitions() == budget.total_transitions && trainer.optimizer_steps() == 4,
            "La recuperación no completó la actualización confirmada");
}

void tiny_positive_nav_keeps_a_finite_logarithm() {
    PpoPolicy policy(observation_width, {}, ppo_default_training_seed);
    std::ostringstream serialized;
    policy.save(serialized);
    const auto archive = edit_network(serialized.str(), [](const auto& network) {
        network->getAttr("output_weight").toTensor().zero_();
        auto bias = network->getAttr("output_bias").toTensor();
        bias.zero_();
        bias[full_exposure].fill_(1);
    });
    std::istringstream source(archive);
    auto buyer = PpoPolicy::load(source);
    constexpr std::size_t count = 3;
    constexpr double tiny_price = 1e-200;
    auto market = tape(count, "validation");
    constexpr std::size_t price_columns = 5;
    for (std::size_t at = 0; at < count; ++at) {
        for (std::size_t column = 0; column < price_columns - 1; ++column) {
            market->prices[at * price_columns + column] = at == count - 1 ? tiny_price : price_base;
        }
    }
    Parameters parameters;
    constexpr double capital = 100;
    parameters.capital = capital;
    parameters.cost_bps = 0;
    parameters.participation = 1;
    const auto result = evaluate_policy(buyer, {{market, parameters, {}}}, 1);
    constexpr double tolerance = 1e-10;
    require(result.incomplete == 0 && result.ruined == 0 && std::isfinite(result.mean_log_growth) &&
                std::abs(result.mean_log_growth - (std::log(tiny_price) - std::log(price_base))) < tolerance,
            "Un patrimonio positivo perdió el logaritmo al redondear su retorno");
}
}

int main() {
    try {
        paused_rollout_resumes_exactly_and_is_owned();
        inputs_and_corrupt_recovery_fail_before_mutation();
        evaluation_keeps_training_rng_and_independent_endings();
        nonfinite_critic_never_enters_a_confirmed_rollout();
        budget_cannot_rewind_without_the_environment();
        bootstrap_failure_rolls_back_the_environment_and_sampler();
        partial_optimizer_failure_requires_a_confirmed_checkpoint();
        tiny_positive_nav_keeps_a_finite_logarithm();
        std::cout << "Recorridos PPO, recuperación y evaluación separados comprobados\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
