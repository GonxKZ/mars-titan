#include "mars_titan/klpo_learning.hpp"
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <torch/optim/adam.h>
#include <torch/serialize/archive.h>

// Estado, gradientes y transiciones sin decisiones. No se llama a update_ready.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
namespace {
using namespace mars_titan::learning;
void require(bool value, const char* message) {
    if (!value) {
        throw std::runtime_error(message);
    }
}
template <class F> void rejected(F f) {
    try {
        f();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("Se aceptó un estado incompatible");
}

std::vector<mars_titan::simulation::BatchInput> sources(bool forced) {
    using namespace mars_titan::simulation;
    auto tape = std::make_shared<MarketTape>();
    tape->assets = {"FIXTURE"};
    tape->open_times = {1, 11, 21};
    tape->close_times = {10, 20, 30};
    tape->prediction_times = tape->close_times;
    tape->prices = {10, 10, 10, 10, 10000, 10, 10, 10, 10, 10000, 10, 10, 10, 10, 10000};
    tape->scores = {.1, .1, .1};
    tape->currency = "USD";
    tape->domain = "synthetic";
    tape->partition = "train";
    tape->parent_id = "fixture";
    tape->source_sha256 = std::string(64, 'a');
    ContextTape context;
    context.fields = {{"may_trade", "boolean"}};
    context.source_sha256 = std::string(64, 'b');
    for (const auto time : tape->close_times) {
        context.values.push_back({forced ? 0.F : 1.F, true, time});
    }
    return {{tape, {}, context}};
}
KlpoLearningConfig settings() {
    KlpoLearningConfig result;
    result.collection.context.enabled = true;
    result.collection.context.variant = "ppo";
    result.collection.context.fold = result.collection.fold;
    result.collection.context.environments = 1;
    result.collection.context.trading_field = 0;
    result.adam = {.learning_rate = .002, .gradient_norm = 0};
    result.confirmed_updates_per_reference = 2;
    result.gradient_block_episodes = 1;
    return result;
}
std::filesystem::path directory() {
    return std::filesystem::temp_directory_path() /
           ("klpo-learning-fixture-" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
}

void cadence_uses_confirmed_updates_in_a_pure_counter_fixture() {
    KlpoLearningCounters state;
    state = klpo_after_consumption(state, 2, false);
    require(state.consumed_waves == 1 && state.confirmed_updates == 0,
            "Se contó una oleada vacía como update");
    state = klpo_before_collection(state, 2);
    state = klpo_after_consumption(state, 2, true);
    state = klpo_before_collection(state, 2);
    state = klpo_after_consumption(state, 2, false);
    state = klpo_before_collection(state, 2);
    state = klpo_after_consumption(state, 2, true);
    require(state.consumed_waves == 4 && state.confirmed_updates == 2 &&
                state.reference_version == 0 && state.updates_since_reference == 2,
            "Se refrescó q antes de confirmar la cadencia");
    const auto next = klpo_before_collection(state, 2);
    require(next.reference_version == 1 && next.updates_since_reference == 0 &&
                next.consumed_waves == 4,
            "El refresco confundió oleadas y actualizaciones");
    rejected([&] { static_cast<void>(klpo_after_consumption(state, 2, true)); });
}

void gradients_leave_actor_and_reference_unchanged() {
    KlpoLearningController run(sources(false), settings());
    rejected([&] { static_cast<void>(run.backward_ready()); });
    const auto actor = run.actor_fingerprint(), reference = run.reference_fingerprint();
    require(actor == reference && run.collect_tick() && run.collect_tick(),
            "No se preparó la oleada fijada");
    const auto before = run.snapshot();
    const auto summary = run.backward_ready();
    require(!summary.no_policy_decisions && summary.episodes == 1 && summary.sampled_decisions == 2,
            "El gradiente perdió las decisiones registradas");
    require(run.actor_fingerprint() == actor && run.reference_fingerprint() == reference &&
                run.counters() == KlpoLearningCounters{},
            "Se cambió el estado sin ejecutar un update");
    require(run.snapshot().metadata == before.metadata,
            "Se persistieron gradientes como un update");
    KlpoLearningController recovered(sources(false), settings());
    recovered.restore(before);
    require(recovered.backward_ready().surrogate == summary.surrogate,
            "La recuperación cambió el cálculo fijo");
    const auto first = run.gradient_snapshot(), second = recovered.gradient_snapshot();
    for (std::size_t index = 0; index < first.size(); ++index) {
        require(at::equal(first[index], second[index]), "El gradiente recuperado cambió");
    }
    auto corrupt = before;
    corrupt.metadata["counters"]["confirmed_updates"] = true;
    rejected([&] { recovered.restore(corrupt); });
    require(recovered.actor_fingerprint() == actor, "El rechazo alteró al actor confirmado");
}

void empty_waves_publish_without_steps_or_reference_refresh() {
    const auto root = directory();
    {
        KlpoLearningController run(sources(true), settings());
        PpoCheckpointStore store(root, run.identity());
        const auto actor = run.actor_fingerprint(), reference = run.reference_fingerprint();
        require(run.collect_tick() && run.collect_tick(), "No se registró la oleada forzada");
        require(run.backward_ready().no_policy_decisions, "Se inventó un gradiente de política");
        run.consume_without_update(store);
        require(run.phase() == KlpoLearningPhase::consumed && run.counters().consumed_waves == 1 &&
                    run.counters().confirmed_updates == 0,
                "La publicación contó un update inexistente");
        run.start_next_wave(store);
        require(run.phase() == KlpoLearningPhase::collecting &&
                    run.counters().reference_version == 0 && run.actor_fingerprint() == actor &&
                    run.reference_fingerprint() == reference,
                "La oleada vacía alteró al actor o refrescó q");
        KlpoLearningController recovered(sources(true), settings());
        recovered.restore(store.load_latest());
        require(recovered.counters() == run.counters(),
                "Se perdieron los contadores sin aprendizaje");
    }
    std::filesystem::remove_all(root);
}

void failures_around_empty_publication_recover_the_confirmed_phase() {
    for (const auto boundary :
         {KlpoLearningBoundary::before_publish, KlpoLearningBoundary::confirmed}) {
        const auto root = directory();
        {
            KlpoLearningController run(sources(true), settings());
            PpoCheckpointStore store(root, run.identity());
            static_cast<void>(run.save(store));
            require(run.collect_tick() && run.collect_tick(), "No se preparó el fixture");
            rejected([&] {
                run.consume_without_update(store, [&](auto at) {
                    if (at == boundary) {
                        throw std::runtime_error("Interrupción del fixture sin update");
                    }
                });
            });
            rejected([&] { static_cast<void>(run.snapshot()); });
            run.restore(store.load_latest());
            require(run.counters().confirmed_updates == 0, "El fallo introdujo un update");
            require(run.phase() == (boundary == KlpoLearningBoundary::confirmed
                                        ? KlpoLearningPhase::consumed
                                        : KlpoLearningPhase::collecting),
                    "La recuperación no siguió la publicación confirmada");
        }
        std::filesystem::remove_all(root);
    }
}

// Archivos etiquetados: los pesos, momentos y contadores se fijan a mano.
// No representan una ejecución del optimizador ni una mejora aprendida.
PpoCheckpointBundle labelled_state(PpoCheckpointBundle state, int64_t steps,
                                   bool changed_critic = false, bool changed_budget = false) {
    const auto split = state.metadata.at("actor_archive_bytes").get<std::size_t>();
    std::istringstream bytes(state.policy_archive.substr(0, split));
    torch::serialize::InputArchive source;
    source.load_from(bytes, at::Device(at::kCPU));
    c10::IValue network;
    source.read("network", network);
    std::vector<at::Tensor> parameters;
    {
        const at::NoGradGuard no_grad;
        network.toObject()->getAttr("first_weight").toTensor().add_(.01);
        if (changed_critic) {
            network.toObject()
                ->getAttr("output_bias")
                .toTensor()
                .select(0, ppo_action_count)
                .add_(1);
        }
    }
    for (const auto* name : {"first_weight", "first_bias", "second_weight", "second_bias",
                             "output_weight", "output_bias"}) {
        parameters.push_back(network.toObject()->getAttr(name).toTensor().detach().clone());
    }
    torch::optim::Adam fixture(parameters,
                               torch::optim::AdamOptions(settings().adam.learning_rate));
    for (const auto& parameter : parameters) {
        auto moment = std::make_unique<torch::optim::AdamParamState>();
        moment->step(steps);
        moment->exp_avg(at::zeros_like(parameter));
        moment->exp_avg_sq(at::zeros_like(parameter));
        fixture.state()[parameter.unsafeGetTensorImpl()] = std::move(moment);
    }
    torch::serialize::OutputArchive output, optimizer;
    fixture.save(optimizer);
    for (const auto& key : source.keys()) {
        if (key == "optimizer") {
            output.write(key, optimizer);
        } else if (key == "memory_budget" && changed_budget) {
            output.write(key, c10::IValue(static_cast<int64_t>(default_ppo_memory_bytes / 2)));
        } else {
            c10::IValue value;
            source.read(key, value);
            output.write(key, value);
        }
    }
    std::ostringstream altered;
    output.save_to(altered);
    state.policy_archive = altered.str() + state.policy_archive.substr(split);
    state.metadata["actor_archive_bytes"] = altered.str().size();
    state.metadata["counters"]["consumed_waves"] = steps;
    state.metadata["counters"]["confirmed_updates"] = steps;
    state.metadata["counters"]["updates_since_reference"] = steps;
    state.metadata["phase"] = "consumed";
    return state;
}

void labelled_states_keep_reference_until_confirmed_cadence() {
    KlpoLearningController original(sources(false), settings());
    require(original.collect_tick() && original.collect_tick(), "Falta el registro del fixture");
    const auto initial = original.snapshot();
    const auto fixed_q = original.reference_fingerprint();
    for (const int64_t steps : {1, 2}) {
        const auto root = directory();
        {
            KlpoLearningController candidate(sources(false), settings());
            candidate.restore(labelled_state(initial, steps));
            const auto fixed_actor = candidate.actor_fingerprint();
            require(fixed_actor != fixed_q && candidate.reference_fingerprint() == fixed_q,
                    "El fixture no separó actor y referencia");
            PpoCheckpointStore store(root, candidate.identity());
            candidate.start_next_wave(store);
            require(candidate.counters().reference_version == (steps == 2 ? 1 : 0) &&
                        candidate.reference_fingerprint() == (steps == 2 ? fixed_actor : fixed_q),
                    "El refresco no respetó las actualizaciones declaradas");
            KlpoLearningController recovered(sources(false), settings());
            recovered.restore(store.load_latest());
            require(recovered.counters() == candidate.counters() &&
                        recovered.actor_fingerprint() == fixed_actor &&
                        recovered.reference_fingerprint() == candidate.reference_fingerprint(),
                    "La recuperación perdió la versión de la referencia");
        }
        std::filesystem::remove_all(root);
    }
    rejected([&] { original.restore(labelled_state(initial, 1, false, true)); });
    rejected([&] { original.restore(labelled_state(initial, 1, true)); });
}

void mixed_episodes_keep_the_global_denominator_across_blocks() {
    auto inputs = sources(true);
    inputs.push_back(sources(false).front());
    auto single_config = settings();
    single_config.collection.context.environments = inputs.size();
    auto joined_config = single_config;
    joined_config.gradient_block_episodes = inputs.size();
    KlpoLearningController single(inputs, single_config), joined(inputs, joined_config);
    require(single.collect_tick() && single.collect_tick() && joined.collect_tick() &&
                joined.collect_tick(),
            "Falta la oleada de episodios mixtos");
    const auto a = single.backward_ready(), b = joined.backward_ready();
    require(a.episodes == 2 && b.episodes == 2 && a.sampled_decisions == 2 && a.blocks == 2 &&
                b.blocks == 1 && std::abs(a.surrogate - b.surrogate) < 1e-8,
            "El bloque cambió la población o el denominador");
    const auto first = single.gradient_snapshot(), second = joined.gradient_snapshot();
    for (std::size_t index = 0; index < first.size(); ++index) {
        require(at::allclose(first[index], second[index], 1e-5, 1e-6),
                "La acumulación por episodios cambió el gradiente");
    }
}

void recovery_rejects_foreign_identity_partial_consumption_and_invalid_counts() {
    {
        KlpoLearningController forced(sources(true), settings());
        require(forced.collect_tick(), "Falta el prefijo forzado incompleto");
        auto partial = forced.snapshot();
        partial.metadata["phase"] = "consumed";
        partial.metadata["counters"]["consumed_waves"] = 1;
        rejected([&] { forced.restore(partial); });
    }
    KlpoLearningController run(sources(false), settings());
    require(run.collect_tick(), "Falta el prefijo incompleto");
    const auto valid = run.snapshot();
    auto corrupt = valid;
    corrupt.metadata["phase"] = "consumed";
    corrupt.metadata["counters"]["consumed_waves"] = 1;
    rejected([&] { run.restore(corrupt); });
    for (const auto* field :
         {"confirmed_updates", "reference_version", "updates_since_reference"}) {
        corrupt = valid;
        corrupt.metadata["counters"][field] = 1;
        rejected([&] { run.restore(corrupt); });
    }
    corrupt = valid;
    corrupt.metadata["identity"]["confirmed_updates_per_reference"] = 1;
    rejected([&] { run.restore(corrupt); });
    corrupt = valid;
    corrupt.metadata["actor_archive_bytes"] = true;
    rejected([&] { run.restore(corrupt); });
    corrupt = valid;
    corrupt.rollout_archive.back() = static_cast<char>(corrupt.rollout_archive.back() ^ 1);
    rejected([&] { run.restore(corrupt); });
    require(run.snapshot().metadata.dump() == valid.metadata.dump(),
            "La recuperación fallida cambió el prefijo confirmado");
    KlpoLearningController resumed(sources(false), settings());
    resumed.restore(valid);
    require(run.collect_tick() && resumed.collect_tick() &&
                run.snapshot().rollout_archive == resumed.snapshot().rollout_archive,
            "El prefijo recuperado no continuó con el mismo sampler");
    const auto root = directory();
    {
        PpoCheckpointStore unrelated(root, {{"kind", "foreign"}});
        rejected([&] { static_cast<void>(run.save(unrelated)); });
    }
    std::filesystem::remove_all(root);
}

void invalid_configs_and_disabled_autograd_are_rejected() {
    for (const auto cadence : {std::size_t{0}, std::size_t{65}}) {
        auto config = settings();
        config.confirmed_updates_per_reference = cadence;
        rejected([&] { KlpoLearningController invalid(sources(false), config); });
    }
    for (const auto block : {std::size_t{0}, maximum_klpo_episodes + 1}) {
        auto config = settings();
        config.gradient_block_episodes = block;
        rejected([&] { KlpoLearningController invalid(sources(false), config); });
    }
    KlpoLearningController run(sources(false), settings());
    require(run.collect_tick() && run.collect_tick(), "Falta la oleada del fixture");
    const at::NoGradGuard disabled;
    rejected([&] { static_cast<void>(run.backward_ready()); });
}

void exhausted_counter_fixtures_reject_another_wave() {
    constexpr uint64_t waves = uint64_t{1} << 32;
    constexpr uint64_t updates = uint64_t{1} << 26;
    rejected([&] {
        static_cast<void>(klpo_after_consumption({updates, updates, updates / 2, 0}, 2, true));
    });
    rejected([&] { static_cast<void>(klpo_after_consumption({waves, 0, 0, 0}, 2, false)); });
    KlpoLearningController run(sources(true), settings());
    require(run.collect_tick() && run.collect_tick(), "Falta el fixture forzado completo");
    auto last = run.snapshot();
    last.metadata["phase"] = "consumed";
    last.metadata["counters"]["consumed_waves"] = waves;
    run.restore(last);
    const auto root = directory();
    {
        PpoCheckpointStore store(root, run.identity());
        rejected([&] { run.start_next_wave(store); });
        require(run.snapshot().metadata.dump() == last.metadata.dump(),
                "El rechazo del límite cambió el estado confirmado");
    }
    std::filesystem::remove_all(root);
    rejected([&] { static_cast<void>(klpo_before_collection({waves, 0, 0, 0}, 2)); });
}
} // namespace

int main() {
    try {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        cadence_uses_confirmed_updates_in_a_pure_counter_fixture();
        gradients_leave_actor_and_reference_unchanged();
        empty_waves_publish_without_steps_or_reference_refresh();
        failures_around_empty_publication_recover_the_confirmed_phase();
        mixed_episodes_keep_the_global_denominator_across_blocks();
        labelled_states_keep_reference_until_confirmed_cadence();
        recovery_rejects_foreign_identity_partial_consumption_and_invalid_counts();
        invalid_configs_and_disabled_autograd_are_rejected();
        exhausted_counter_fixtures_reject_another_wave();
        std::cout << "Controlador terminal contrastado sin pasos de optimizador\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
