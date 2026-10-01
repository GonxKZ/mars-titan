#include "mars_titan/ppo_training.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/grad_mode.h>
#include <torch/serialize/archive.h>

#include <array>
#include <cmath>
#include <cstring>
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
constexpr double equal_prior = 0.5;
constexpr double persistence = 0.9;
constexpr double switching = 0.1;

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

namespace {
std::vector<BatchInput> learning_sources() {
    std::vector<BatchInput> result;
    constexpr std::size_t sessions = 6;
    for (const char digest : {'a', 'b', 'c'}) {
        auto market = tape(sessions, "train", digest);
        ContextTape context;
        context.source_sha256.assign(digest_size, digest);
        context.fields = {{"trading_enabled", "indicador"}, {"signal", "indicador"}};
        for (std::size_t session = 0; session < sessions; ++session) {
            context.values.push_back({session == 0 ? 0.F : 1.F, true, market->close_times[session]});
            context.values.push_back({static_cast<float>(session), true, market->close_times[session]});
        }
        result.push_back({std::move(market), {}, std::move(context)});
    }
    return result;
}

void adaptive_catalog_and_warmup_resume_exactly() {
    for (const auto* variant : {"ppo", "double_dqn", "ppo_window", "ppo_gru", "ppo_episodic",
                                "ppo_hmm", "ppo_episodic_hmm", "ppo_recent_aux", "ppo_replay_aux"}) {
        PpoLearningOptions options;
        options.enabled = true;
        options.environments = 2;
        options.trading_field = 0;
        options.variant = variant;
        options.markov = MarkovParameters{2, 1, {equal_prior, equal_prior},
            {persistence, switching, switching, persistence}, {0, 2}, {1, 1}};
        options.markov_fields = {1};
        auto parameters = hyperparameters();
        parameters.minibatch_size = ppo_sequence_length;
        PpoTrainer continuous(learning_sources(), config(), parameters, "cpu", true, options);
        PpoTrainer interrupted(learning_sources(), config(), parameters, "cpu", true, options);
        require(interrupted.advance() && interrupted.transitions() == 0 &&
                    interrupted.observed_transitions() == 2,
                "El calentamiento ha contado como aprendizaje");
        require(interrupted.advance() && interrupted.advance(), "Falta progreso adaptativo");
        auto state = interrupted.snapshot();
        require(state.source_indices == std::vector<std::size_t>{0, 1},
                "El catálogo no conserva sus fuentes activas");
        const auto encoded = serialize_training_buffer(state);
        state.rollout = {};
        state.adaptive_archive.clear();
        restore_training_buffer(encoded, state);
        PpoTrainer recovered(learning_sources(), config(), parameters, "cpu", true, options);
        recovered.restore(state);
        while (continuous.advance()) {}
        while (recovered.advance()) {}
        const auto expected = continuous.snapshot();
        const auto actual = recovered.snapshot();
        require(actual.transitions == total_transitions &&
                    actual.observed_transitions > actual.transitions &&
                    actual.observed_transitions == expected.observed_transitions &&
                    actual.source_indices == expected.source_indices &&
                    actual.next_source == expected.next_source,
                "La recuperación cambia la secuencia de mundos o su presupuesto");
        const auto observation = at::ones({2, static_cast<int64_t>(continuous.policy().observation_width())}, at::kFloat);
        require(at::equal(continuous.policy().forward(observation).logits,
                          recovered.policy().forward(observation).logits),
                "La recuperación adaptativa altera los pesos finales");
        if (continuous.policy().architecture().auxiliary) {
            require(continuous.policy().auxiliary_steps() > 0 &&
                        continuous.policy().auxiliary_steps() == recovered.policy().auxiliary_steps() &&
                        continuous.auxiliary_samples() == recovered.auxiliary_samples() &&
                        continuous.auxiliary_samples() > continuous.policy().auxiliary_steps(),
                    "La consolidación no se ejecutó o su recuperación cambió el presupuesto");
        } else if (continuous.policy().architecture().double_dqn) {
            require(actual.optimizer_steps == 0 && continuous.policy().dqn_environment_step() == 0,
                    "Double DQN actualizó antes de reunir su calentamiento de 256 decisiones");
        }
        auto corrupted = actual;
        corrupted.source_indices[0] = learning_sources().size();
        rejected([&] { recovered.restore(corrupted); });
    }
}

void recovered_episode_count_must_match_context_resets() {
    PpoLearningOptions options;
    options.enabled = true;
    options.environments = 2;
    options.trading_field = 0;
    PpoTrainer trainer(learning_sources(), config(), hyperparameters(), "cpu", true, options);
    // Antes del final, con reinicios pendientes y después de aplicarlos.
    constexpr std::array<std::size_t, 3> checkpoint_steps{1, 5, 6};
    std::size_t advanced = 0;
    for (const auto checkpoint : checkpoint_steps) {
        while (advanced < checkpoint) {
            require(trainer.advance(), "Falta una transición para comprobar el cursor de episodios");
            ++advanced;
        }
        const auto confirmed = trainer.snapshot();
        auto corrupted = confirmed;
        ++corrupted.episodes;
        ++corrupted.next_source;
        rejected([&] { trainer.restore(corrupted); });
        const auto after = trainer.snapshot();
        require(after.episodes == confirmed.episodes && after.next_source == confirmed.next_source &&
                    after.source_indices == confirmed.source_indices &&
                    after.adaptive_archive == confirmed.adaptive_archive,
                "El rechazo de episodios inventados alteró la fuente o el contexto confirmado");
        trainer.restore(confirmed);
    }
}

void recurrent_history_must_fit_the_rollout_budget_before_allocation() {
    PpoLearningOptions options;
    options.enabled = true;
    options.environments = 2;
    options.trading_field = 0;
    options.variant = "ppo_gru";
    auto parameters = hyperparameters();
    parameters.minibatch_size = ppo_sequence_length;
    auto limited = config();
    constexpr std::size_t small_budget = std::size_t{32} * 1024;
    limited.rollout_bytes = small_budget;
    rejected([&] {
        PpoTrainer oversized(learning_sources(), limited, parameters, "cpu", true, options);
    });
    options.variant = "ppo";
    PpoTrainer feedforward(learning_sources(), limited, parameters, "cpu", true, options);
    require(!feedforward.snapshot().rollout.prefix_observations.defined(),
            "La reserva recurrente cambió el contrato del control sin recurrencia");
    options.variant = "ppo_gru";
    PpoTrainer admitted(learning_sources(), config(), parameters, "cpu", true, options);
    const auto prefix = admitted.snapshot().rollout.prefix_observations;
    const auto prefix_bytes = static_cast<std::size_t>(prefix.numel()) * sizeof(float);
    require(prefix_bytes > small_budget && prefix_bytes < config().rollout_bytes,
            "La prueba no distingue el prefijo recurrente del presupuesto insuficiente");
}

void observed_decisions_must_match_complete_ticks_and_the_final_partial_batch() {
    auto sources = learning_sources();
    auto& context = sources.at(1).context;
    if (!context) {
        throw std::runtime_error("La prueba necesita el contexto de calentamiento");
    }
    context->values.at(context->fields.size()).value = 0;
    PpoLearningOptions options;
    options.enabled = true;
    options.environments = 2;
    options.trading_field = 0;
    PpoTrainer trainer(sources, config(), hyperparameters(), "cpu", true, options);
    require(trainer.advance() && trainer.advance(), "Faltan ticks para comprobar sus decisiones");
    const auto confirmed = trainer.snapshot();
    auto missing_decision = confirmed;
    --missing_decision.observed_transitions;
    rejected([&] { trainer.restore(missing_decision); });
    require(trainer.observed_transitions() == confirmed.observed_transitions,
            "Un contador de decisiones falso modificó el estado confirmado");
    trainer.restore(confirmed);
    while (trainer.advance()) {}
    const auto completed = trainer.snapshot();
    require(completed.observed_transitions % options.environments != 0,
            "El calentamiento desigual no produjo el lote final parcial de la prueba");
    PpoTrainer recovered(sources, config(), hyperparameters(), "cpu", true, options);
    recovered.restore(completed);
    require(recovered.observed_transitions() == completed.observed_transitions &&
                !recovered.advance(),
            "La recuperación rechazó o repitió el último lote parcial válido");
    auto missing_tick = completed;
    missing_tick.observed_transitions -= options.environments;
    rejected([&] { recovered.restore(missing_tick); });
}

void audit_sensitivity_keeps_policy_rng_and_actual_trajectory() {
    PpoLearningOptions options;
    options.enabled = true;
    options.environments = 2;
    options.trading_field = 0;
    options.variant = "ppo_episodic";
    auto parameters = hyperparameters();
    parameters.minibatch_size = ppo_sequence_length;
    PpoTrainer trainer(learning_sources(), config(), parameters, "cpu", true, options);
    while (trainer.advance()) {}
    auto validation = learning_sources();
    for (auto& input : validation) {
        auto market = std::make_shared<MarketTape>(*input.tape);
        market->partition = "validation";
        input.tape = std::move(market);
    }
    const auto rng = trainer.policy().random_state();
    const auto expected = evaluate_policy(trainer.policy(), validation, 1, {}, options);
    std::vector<DecisionRecord> records;
    const auto traced = evaluate_policy(trainer.policy(), validation, 1, {}, options,
        [&](std::span<const DecisionRecord> batch) { records.insert(records.end(), batch.begin(), batch.end()); }, true);
    require(traced.mean_log_growth == expected.mean_log_growth && traced.incomplete == 0 &&
                traced.episodes == validation.size() &&
                records.size() == validation.size() * (validation.front().tape->close_times.size() - 1),
            "La sensibilidad cambia la trayectoria real o pierde decisiones al paginar mundos");
    require(at::equal(rng.sampling, trainer.policy().random_state().sampling) &&
                at::equal(rng.shuffle, trainer.policy().random_state().shuffle),
            "La auditoría consume el RNG del entrenamiento");
    bool recalled = false;
    for (std::size_t index = 0; index < records.size(); ++index) {
        const auto& record = records[index];
        require(record.decision_id == index + 1 && record.outcome_at > record.decision_at &&
                    !record.learning_allowed && record.memory_sensitivity.has_value(),
                "La traza de auditoría pierde su identidad, tiempo o sensibilidad");
        if (!record.memory_sensitivity) {
            throw std::runtime_error("Falta el resultado de sensibilidad");
        }
        const auto& sensitivity = record.memory_sensitivity.value();
        require(sensitivity.probability_l1 >= 0 && sensitivity.probability_l1 <= 2 &&
                    sensitivity.action_changed == (sensitivity.action != record.action),
                "La sensibilidad no conserva su diferencia de distribución y acción");
        if (record.mode == "warmup") {
            require(record.action == 1 && record.probabilities[1] == 1 &&
                        sensitivity.probability_l1 == 0,
                    "El calentamiento registra una acción no ejecutada");
        }
        for (std::size_t neighbor = 0; neighbor < record.retrieved_count; ++neighbor) {
            require(record.matured_at.at(neighbor) <= record.decision_at,
                    "La explicación consulta un resultado futuro");
            recalled = true;
        }
    }
    require(recalled, "La auditoría no ha ejercitado la recuperación de recuerdos");
}

void sensitivity_preserves_argmax_when_softmax_rounds_nearly_tied_logits() {
    for (const auto* variant : {"ppo", "double_dqn"}) {
        PpoLearningOptions options;
        options.enabled = true;
        options.environments = 2;
        options.trading_field = 0;
        options.variant = variant;
        PpoTrainer trainer(learning_sources(), config(), hyperparameters(), "cpu", true, options);
        const auto bytes = edit_network(trainer.snapshot().policy_archive, [](const auto& network) {
            for (std::size_t index = 0; index < network->type()->numAttributes(); ++index) {
                network->getSlot(index).toTensor().zero_();
            }
            constexpr float margin = 1e-8F;
            network->getAttr("output_bias").toTensor()[2].fill_(margin);
        });
        std::istringstream archive(bytes);
        const auto policy = PpoPolicy::load(archive);
        auto validation = learning_sources();
        for (auto& input : validation) {
            auto market = std::make_shared<MarketTape>(*input.tape);
            market->partition = "validation";
            input.tape = std::move(market);
        }
        std::size_t checked = 0;
        static_cast<void>(evaluate_policy(policy, validation, 1, {}, options,
            [&](std::span<const DecisionRecord> records) {
                for (const auto& record : records) {
                    if (record.mode == "warmup") {
                        continue;
                    }
                    if (!record.memory_sensitivity) {
                        throw std::runtime_error("Falta la sensibilidad de la política controlada");
                    }
                    const auto& sensitivity = *record.memory_sensitivity;
                    require(record.action == 2 && sensitivity.action == record.action &&
                                !sensitivity.action_changed && sensitivity.probability_l1 == 0,
                            "El redondeo del softmax inventó un cambio de acción sin cambiar los logits");
                    ++checked;
                }
            }, true));
        require(checked > 0, "Faltan decisiones posteriores al calentamiento en la prueba");
    }
}

void sensitivity_rejects_nonfinite_logits_even_with_finite_probabilities() {
    PpoLearningOptions options;
    options.enabled = true;
    options.environments = 2;
    options.trading_field = 0;
    options.variant = "ppo_episodic";
    PpoTrainer trainer(learning_sources(), config(), hyperparameters(), "cpu", true, options);
    constexpr std::size_t memory_tail_fields = 4;
    const auto presence = static_cast<int64_t>(trainer.policy().observation_width() - memory_tail_fields);
    const auto bytes = edit_network(trainer.snapshot().policy_archive, [&](const auto& network) {
        for (std::size_t index = 0; index < network->type()->numAttributes(); ++index) {
            network->getSlot(index).toTensor().zero_();
        }
        auto first = network->getAttr("first_weight").toTensor();
        first.select(1, static_cast<int64_t>(observation_width)).fill_(1);
        first.select(1, presence).fill_(-1);
        network->getAttr("second_weight").toTensor().copy_(at::eye(first.size(0)));
        network->getAttr("output_weight").toTensor()[0].fill_(-std::numeric_limits<float>::max());
    });
    std::istringstream archive(bytes);
    const auto policy = PpoPolicy::load(archive);
    auto validation = learning_sources();
    for (auto& input : validation) {
        auto market = std::make_shared<MarketTape>(*input.tape);
        market->partition = "validation";
        input.tape = std::move(market);
    }
    const auto complete = evaluate_policy(policy, validation, 1, {}, options);
    require(!complete.paused && complete.incomplete == 0,
            "La trayectoria real del caso de desbordamiento debe permanecer finita");
    rejected([&] {
        static_cast<void>(evaluate_policy(policy, validation, 1, {}, options,
            [](std::span<const DecisionRecord>) {}, true));
    });
}

void recovered_context_must_match_the_confirmed_financial_observation() {
    PpoLearningOptions options;
    options.enabled = true;
    options.environments = 2;
    options.trading_field = 0;
    PpoTrainer trainer(learning_sources(), config(), hyperparameters(), "cpu", true, options);
    require(trainer.advance() && trainer.advance(), "Faltan observaciones antes de recuperar");
    const auto confirmed = trainer.snapshot();
    auto corrupted = confirmed;
    const auto encoded = [](torch::serialize::InputArchive& input) {
        torch::serialize::OutputArchive output;
        for (const auto& name : input.keys()) {
            c10::IValue value;
            input.read(name, value);
            if (value.isTensor()) output.write(name, value.toTensor(), true);
            else output.write(name, value);
        }
        std::ostringstream stream;
        output.save_to(stream);
        return std::move(stream).str();
    };
    torch::serialize::InputArchive adaptive;
    std::istringstream adaptive_source(corrupted.adaptive_archive);
    adaptive.load_from(adaptive_source, at::Device(at::kCPU));
    at::Tensor context_bytes;
    adaptive.read("context", context_bytes, true);
    std::string context_content(static_cast<std::size_t>(context_bytes.numel()), '\0');
    std::memcpy(context_content.data(), context_bytes.const_data_ptr<uint8_t>(), context_content.size());
    torch::serialize::InputArchive context_archive;
    std::istringstream context_source(context_content);
    context_archive.load_from(context_source, at::Device(at::kCPU));
    at::Tensor raw, history;
    context_archive.read("raw", raw, true);
    context_archive.read("history", history, true);
    // El archivo conserva su coherencia interna, pero describe otra observación financiera.
    raw[0][0].add_(1);
    history[0][-1][0].add_(1);
    const auto changed_context = encoded(context_archive);
    auto active = learning_sources();
    active.resize(options.environments);
    FinancialBatch batch(active);
    batch.restore(confirmed.environment);
    PolicyContext independent(active, options, config().seed, batch.observations());
    independent.restore({changed_context});
    require(independent.cursor(0) == batch.cursor(0) &&
                independent.observations()[0][0].item<float>() != batch.observations()[0],
            "La reproducción no conserva los cursores y la discrepancia de observación");
    torch::serialize::OutputArchive edited;
    for (const auto& name : adaptive.keys()) {
        at::Tensor value;
        adaptive.read(name, value, true);
        if (name == "context") {
            value = at::empty({static_cast<int64_t>(changed_context.size())}, at::kByte);
            std::memcpy(value.data_ptr<uint8_t>(), changed_context.data(), changed_context.size());
        }
        edited.write(name, value, true);
    }
    std::ostringstream destination;
    edited.save_to(destination);
    corrupted.adaptive_archive = std::move(destination).str();
    rejected([&] { trainer.restore(corrupted); });
    require(trainer.snapshot().adaptive_archive == confirmed.adaptive_archive &&
                at::equal(trainer.snapshot().rollout.observations, confirmed.rollout.observations),
            "La recuperación incompatible modificó el estado confirmado");
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
        adaptive_catalog_and_warmup_resume_exactly();
        recovered_episode_count_must_match_context_resets();
        recurrent_history_must_fit_the_rollout_budget_before_allocation();
        observed_decisions_must_match_complete_ticks_and_the_final_partial_batch();
        audit_sensitivity_keeps_policy_rng_and_actual_trajectory();
        sensitivity_preserves_argmax_when_softmax_rounds_nearly_tied_logits();
        sensitivity_rejects_nonfinite_logits_even_with_finite_probabilities();
        recovered_context_must_match_the_confirmed_financial_observation();
        std::cout << "Recorridos PPO, recuperación y evaluación separados comprobados\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
