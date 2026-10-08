#include "mars_titan/klpo_collection.hpp"
#include "mars_titan/klpo_terminal.hpp"
#include "mars_titan/ppo_objectives.hpp"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include <chrono>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <utility>

// Registros y logits fijos. No se actualizan parámetros.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
namespace {
using namespace mars_titan::learning;
template <class T>
concept temporary_records = requires(T&& value) { std::move(value).records(); };
template <class T>
concept temporary_policy = requires(T&& value) { std::move(value).policy(); };
static_assert(!temporary_records<KlpoTerminalCollector>);
static_assert(!temporary_policy<KlpoTerminalCollector>);
void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}
template <class F> void rejected(F&& f, const char* message) {
    try {
        f();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(message);
}

KlpoEpisodeBatch recorded() {
    KlpoEpisodeBatch batch;
    batch.fold = "fixed";
    batch.reference_sha256 = std::string(64, 'a');
    batch.observation_width = 2;
    batch.beta = .2;
    batch.gamma = .25;
    for (std::size_t lane = 0; lane < 2; ++lane) {
        KlpoEpisodeRecord record;
        record.spec.id = "episode-" + std::to_string(lane);
        record.spec.source_sha256 = std::string(64, 'b');
        record.spec.close_times = {10, 20, 30};
        record.spec.forced_prefix = lane == 0 ? 1 : 2;
        for (std::size_t cursor = 0; cursor < 2; ++cursor) {
            KlpoEpisodeStep step;
            step.cursor = cursor;
            step.decision_at = record.spec.close_times[cursor];
            step.outcome_at = record.spec.close_times[cursor + 1];
            step.observation = {.1F, .2F};
            step.sampled = cursor >= record.spec.forced_prefix;
            step.action = step.sampled ? uint8_t{2} : uint8_t{1};
            if (step.sampled) {
                step.behavior.fill(1.F / 6);
            }
            step.reward = cursor == 0 ? 3. : 4.;
            step.truncated = cursor == 1;
            record.steps.push_back(step);
        }
        batch.episodes.push_back(record);
    }
    return batch;
}

void whole_wave_denominator_keeps_forced_episodes() {
    PpoPolicy policy(2, {}, 42);
    const auto records = recorded();
    const auto result = klpo_episode_objective(policy, records);
    require(!result.no_policy_decisions && result.sampled_decisions == 1,
            "El registro perdió la decisión sorteada");
    require(result.per_episode.numel() == 2 && result.per_episode[1].item<double>() == 0,
            "Se eliminó el episodio forzado");
    const auto history = at::tensor({.1F, .2F, .1F, .2F}).reshape({2, 1, 2});
    const auto logits = policy.terminal_forward(history, at::tensor({2}, at::kLong)).logits[1];
    const auto logp =
        ppo_behavior_log_probabilities(logits.log_softmax(1).exp()).reshape({1, 1, 6});
    const auto q = ppo_behavior_log_probabilities(at::full({1, 6}, 1.F / 6)).reshape({1, 1, 6});
    const auto literal =
        klpo_terminal_full_loss(logp, q, at::full({1, 1}, 2, at::kLong),
                                at::tensor({4.}, at::kDouble), at::ones({1, 1}, at::kBool), .2);
    require(at::allclose(result.mean, literal.sum() / 2, 1e-5, 1e-7),
            "La media se normalizó solo por episodios con acciones");
    require(result.mean.requires_grad(), "Se desacopló el actor del núcleo KLPO");
    require(policy.optimizer_steps() == 0, "El consumidor ejecutó una actualización");
}

void no_decisions_is_an_explicit_no_update() {
    PpoPolicy policy(2, {}, 42);
    auto records = recorded();
    records.episodes[0].spec.forced_prefix = 2;
    auto& last = records.episodes[0].steps.back();
    last.sampled = false;
    last.action = 1;
    last.behavior.fill(0);
    const auto before = policy.random_state();
    const auto result = klpo_episode_objective(policy, records);
    require(result.no_policy_decisions && result.sampled_decisions == 0 &&
                result.per_episode.numel() == 2 && result.mean.item<double>() == 0 &&
                !result.mean.requires_grad(),
            "La oleada forzada no conserva el caso sin actualización");
    require(at::equal(before.sampling, policy.random_state().sampling) &&
                policy.optimizer_steps() == 0,
            "La oleada forzada alteró el muestreador");
    records.episodes[1].steps.pop_back();
    rejected([&] { static_cast<void>(klpo_episode_objective(policy, records)); },
             "Se ignoró una trayectoria parcial sin decisiones");
}

mars_titan::simulation::BatchInput input(char identity, std::size_t forced) {
    using namespace mars_titan::simulation;
    auto tape = std::make_shared<MarketTape>();
    tape->assets = {"A"};
    tape->open_times = {1, 11, 21};
    tape->close_times = {10, 20, 30};
    tape->prediction_times = tape->close_times;
    tape->prices = {10, 10, 10, 10, 1000, 10, 10, 10, 10, 1000, 10, 10, 10, 10, 1000};
    tape->scores = {.1, .1, .1};
    tape->currency = "USD";
    tape->domain = "synthetic";
    tape->partition = "train";
    tape->parent_id = "fixed-parent";
    tape->source_sha256 = std::string(64, identity);
    ContextTape context;
    context.source_sha256 = std::string(64, identity);
    context.fields = {{"may_trade", "boolean"}};
    for (std::size_t row = 0; row < 3; ++row) {
        context.values.push_back({row < forced ? 0.F : 1.F, true, tape->close_times[row]});
    }
    return {tape, {}, context};
}

KlpoCollectionOptions options(PpoNetworkKind kind) {
    KlpoCollectionOptions result;
    result.fold = "fixed";
    result.beta = .2;
    result.gamma = .25;
    result.architecture.kind = kind;
    result.context.enabled = true;
    result.context.variant = kind == PpoNetworkKind::gru ? "ppo_gru" : "ppo";
    result.context.fold = result.fold;
    result.context.environments = 2;
    result.context.trading_field = 0;
    return result;
}

void collector_restores_a_partial_wave_without_refilling(PpoNetworkKind kind) {
    const std::vector sources{input('a', 1), input('b', 2)};
    const auto config = options(kind);
    KlpoTerminalCollector run(sources, config);
    const auto rng = run.policy().random_state();
    require(run.collect_tick(), "No se registró el calentamiento");
    require(at::equal(rng.sampling, run.policy().random_state().sampling),
            "El calentamiento sorteó acciones");
    require(run.records().episodes[0].steps[0].valuation_valid &&
                !run.records().episodes[0].steps[0].sampled,
            "Se mezclaron valoración y elegibilidad");
    rejected([&] { static_cast<void>(run.objective()); }, "Se consumió la oleada parcial");
    const auto partial = run.snapshot();
    KlpoTerminalCollector recovered(sources, config);
    recovered.restore(partial);
    require(run.collect_tick() && recovered.collect_tick(),
            "No se completó el episodio recuperado");
    require(!run.collect_tick() && !recovered.collect_tick(), "Se rellenó un carril finalizado");
    require(run.phase() == KlpoCollectionPhase::ready &&
                serialize_klpo_batch(run.records()) == serialize_klpo_batch(recovered.records()),
            "La recuperación cambió IDs, acciones o recompensas");
    require(
        at::equal(run.policy().random_state().sampling, recovered.policy().random_state().sampling),
        "La recuperación cambió el RNG");
    require(run.objective().sampled_decisions == 1 && recovered.policy().optimizer_steps() == 0,
            "Se perdió el episodio forzado o se ejecutó una actualización");
    auto damaged = partial;
    damaged.metadata["identity"]["fold"] = "other";
    rejected([&] { recovered.restore(damaged); }, "Se mezclaron scopes de recuperación");
    require(recovered.phase() == KlpoCollectionPhase::ready,
            "El rechazo alteró el estado confirmado");
}

void checkpoint_store_keeps_the_same_partial_wave() {
    const std::vector sources{input('a', 1), input('b', 2)};
    KlpoTerminalCollector run(sources, options(PpoNetworkKind::mlp));
    const auto directory =
        std::filesystem::temp_directory_path() /
        ("klpo-checkpoint-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    {
        PpoCheckpointStore store(directory, run.identity());
        require(run.collect_tick(), "No se inició la oleada");
        static_cast<void>(run.save(store));
        const auto payload = store.load_latest();
        KlpoTerminalCollector restored(sources, options(PpoNetworkKind::mlp));
        restored.restore(payload);
        require(serialize_klpo_batch(restored.records()) == serialize_klpo_batch(run.records()),
                "El escritor cambió la oleada parcial");
    }
    std::filesystem::remove_all(directory);
}

void wave_valuation_failure_is_preserved_and_blocks_every_episode() {
    auto damaged = input('a', 0);
    auto tape = std::make_shared<mars_titan::simulation::MarketTape>(*damaged.tape);
    const auto missing = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t field = 0; field < 4; ++field) {
        tape->prices[10 + field] = missing;
    }
    damaged.tape = tape;
    auto following = input('b', 0);
    auto second = std::make_shared<mars_titan::simulation::MarketTape>(*following.tape);
    for (std::size_t field = 0; field < 4; ++field) {
        second->prices[10 + field] = missing;
    }
    following.tape = second;
    const std::vector sources{damaged, following};
    KlpoTerminalCollector run(sources, options(PpoNetworkKind::mlp));
    require(run.collect_tick(), "No se registró la primera decisión");
    require(run.records().episodes[0].steps[0].action >= 2 ||
                run.records().episodes[1].steps[0].action >= 2,
            "El fixture no abrió la posición necesaria");
    require(run.collect_tick(), "No se registró el fallo de valoración");
    require(run.phase() == KlpoCollectionPhase::blocked && !run.collect_tick(),
            "La falta de valoración no bloqueó toda la oleada");
    require(run.records().episodes.size() == 2, "Se descartó el resultado no valorable");
    rejected([&] { static_cast<void>(run.objective()); },
             "Se consumió una oleada con valoración ausente");
    const auto blocked = run.snapshot();
    KlpoTerminalCollector restored(sources, options(PpoNetworkKind::mlp));
    restored.restore(blocked);
    require(restored.phase() == KlpoCollectionPhase::blocked,
            "La recuperación admitió el fallo como terminal válido");
}

void failures_restore_sampler_and_only_adopt_complete_transitions() {
    const std::vector sources{input('a', 0), input('b', 0)};
    for (const auto boundary :
         {KlpoCollectionBoundary::before_commit, KlpoCollectionBoundary::committed}) {
        KlpoTerminalCollector run(sources, options(PpoNetworkKind::gru));
        const auto initial = run.snapshot();
        bool failed = false;
        try {
            run.collect_tick([&](auto actual) {
                if (actual == boundary) {
                    throw std::runtime_error("Interrupción del fixture");
                }
            });
        } catch (const std::runtime_error&) {
            failed = true;
        }
        require(failed, "No se ejercitó el punto de fallo");
        if (boundary == KlpoCollectionBoundary::before_commit) {
            const auto current = run.snapshot();
            require(current.metadata == initial.metadata &&
                        current.policy_archive == initial.policy_archive &&
                        current.rollout_archive == initial.rollout_archive,
                    "El fallo previo alteró estado confirmado");
        } else {
            rejected([&] { static_cast<void>(run.snapshot()); },
                     "Se publicó después de un fallo no confirmado");
        }
        run.restore(initial);
        KlpoTerminalCollector reference(sources, options(PpoNetworkKind::gru));
        require(run.collect_tick() && reference.collect_tick(), "No se recuperó el tick pendiente");
        require(serialize_klpo_batch(run.records()) == serialize_klpo_batch(reference.records()),
                "El corte duplicó o alteró una decisión");
    }
}

void another_store_identity_is_rejected_before_publication() {
    const std::vector sources{input('a', 1), input('b', 2)};
    KlpoTerminalCollector run(sources, options(PpoNetworkKind::mlp));
    auto foreign = run.identity();
    foreign["fold"] = "other";
    const auto directory =
        std::filesystem::temp_directory_path() /
        ("klpo-foreign-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    {
        PpoCheckpointStore store(directory, foreign);
        rejected([&] { static_cast<void>(run.save(store)); },
                 "Se escribió la oleada bajo otra identidad");
    }
    std::filesystem::remove_all(directory);
}

void coherent_shapes_do_not_hide_changed_probabilities_or_hidden_state() {
    const std::vector sources{input('a', 1), input('b', 2)};
    KlpoTerminalCollector run(sources, options(PpoNetworkKind::gru));
    require(run.collect_tick() && run.collect_tick(), "No se completó el fixture");
    const auto state = run.snapshot();
    auto changed = state;
    const auto bytes = state.metadata.at("record_bytes").get<std::size_t>();
    auto parsed = deserialize_klpo_batch(std::string_view(state.rollout_archive).substr(0, bytes));
    parsed.episodes[0].steps.back().behavior = {.1F, .1F, .2F, .2F, .2F, .2F};
    const auto alternative = serialize_klpo_batch(parsed);
    require(alternative.size() == bytes && alternative != state.rollout_archive.substr(0, bytes),
            "No se modificó la distribución del fixture");
    changed.rollout_archive = alternative + state.rollout_archive.substr(bytes);
    rejected([&] { run.restore(changed); }, "Se aceptó otra q bajo la referencia congelada");
    changed = state;
    changed.metadata["hidden"]["values"][0] = .5;
    rejected([&] { run.restore(changed); }, "Se aceptó un hidden que no procede del prefijo");
    require(serialize_klpo_batch(run.records()) == state.rollout_archive.substr(0, bytes),
            "Un rechazo de recuperación alteró los datos confirmados");
}

void a_complete_forced_wave_never_samples_or_updates() {
    const std::vector sources{input('a', 2), input('b', 2)};
    KlpoTerminalCollector run(sources, options(PpoNetworkKind::gru));
    const auto rng = run.policy().random_state();
    require(run.collect_tick() && run.collect_tick() && !run.collect_tick(),
            "La oleada forzada no conserva el horizonte");
    const auto loss = run.objective();
    require(loss.no_policy_decisions && loss.per_episode.numel() == 2 &&
                loss.mean.item<double>() == 0 && !loss.mean.requires_grad(),
            "La oleada forzada se convirtió en una actualización");
    require(at::equal(rng.sampling, run.policy().random_state().sampling) &&
                run.policy().optimizer_steps() == 0,
            "El calentamiento alteró el sampler o el optimizador");
}

void operator_precision_is_identified_without_changing_caller_flags() {
    auto& runtime = at::globalContext();
    const auto generic = runtime.float32Precision(at::Float32Backend::GENERIC, at::Float32Op::ALL);
    const auto conv = runtime.float32Precision(at::Float32Backend::CUDA, at::Float32Op::CONV);
    const auto rnn = runtime.float32Precision(at::Float32Backend::CUDA, at::Float32Op::RNN);
    runtime.setFloat32Precision(at::Float32Backend::GENERIC, at::Float32Op::ALL,
                                at::Float32Precision::IEEE);
    runtime.setFloat32Precision(at::Float32Backend::CUDA, at::Float32Op::CONV,
                                at::Float32Precision::TF32);
    runtime.setFloat32Precision(at::Float32Backend::CUDA, at::Float32Op::RNN,
                                at::Float32Precision::IEEE);
    const std::vector sources{input('a', 1), input('b', 2)};
    KlpoTerminalCollector run(sources, options(PpoNetworkKind::gru));
    const auto precision = run.identity().at("precision");
    require(precision.at("cudnn_conv_tf32") == true && precision.at("cudnn_rnn_tf32") == false &&
                runtime.allowTF32CuDNN(at::Float32Op::CONV) &&
                !runtime.allowTF32CuDNN(at::Float32Op::RNN),
            "No se conservó la precisión efectiva por operador");
    runtime.setFloat32Precision(at::Float32Backend::CUDA, at::Float32Op::RNN,
                                at::Float32Precision::TF32);
    rejected([&] { static_cast<void>(run.collect_tick()); },
             "Se cambió la precisión RNN durante la oleada");
    runtime.setFloat32Precision(at::Float32Backend::GENERIC, at::Float32Op::ALL, generic);
    runtime.setFloat32Precision(at::Float32Backend::CUDA, at::Float32Op::CONV, conv);
    runtime.setFloat32Precision(at::Float32Backend::CUDA, at::Float32Op::RNN, rnn);
}
} // namespace

int main() {
    try {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        whole_wave_denominator_keeps_forced_episodes();
        no_decisions_is_an_explicit_no_update();
        collector_restores_a_partial_wave_without_refilling(PpoNetworkKind::mlp);
        collector_restores_a_partial_wave_without_refilling(PpoNetworkKind::gru);
        checkpoint_store_keeps_the_same_partial_wave();
        wave_valuation_failure_is_preserved_and_blocks_every_episode();
        failures_restore_sampler_and_only_adopt_complete_transitions();
        another_store_identity_is_rejected_before_publication();
        coherent_shapes_do_not_hide_changed_probabilities_or_hidden_state();
        a_complete_forced_wave_never_samples_or_updates();
        operator_precision_is_identified_without_changing_caller_flags();
        std::cout << "Consumidor terminal contrastado sin optimización\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
