#include "mars_titan/klpo_collection.hpp"
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <iostream>
#include <stdexcept>

// Fuentes pequeñas y actores fijados. Se muestrean acciones, sin optimizar.
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
    throw std::runtime_error("Se aceptó una referencia incompatible");
}

std::vector<mars_titan::simulation::BatchInput> sources() {
    auto tape = std::make_shared<mars_titan::simulation::MarketTape>();
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
    return {{tape, {}, {}}};
}

void imported_reference_replays_from_its_actual_start_rng(PpoNetworkKind kind) {
    const auto inputs = sources();
    KlpoCollectionOptions config;
    config.architecture.kind = kind;
    KlpoTerminalCollector original(inputs, config);
    require(original.snapshot().metadata.at("schema_version") == 1,
            "El modo técnico anterior cambió de contrato");
    PpoPolicy parent(original.policy().observation_width(), {}, 17, "cpu", default_ppo_memory_bytes,
                     config.architecture);
    static_cast<void>(
        parent.act_recurrent(at::zeros({1, static_cast<int64_t>(parent.observation_width())})));
    const auto start = parent.random_state();
    KlpoTerminalCollector run(inputs, config, parent.frozen_reference(start));
    require(run.records().reference_sha256 == parent.parameter_fingerprint() &&
                run.records().reference_sha256 != original.records().reference_sha256,
            "Se sustituyó la referencia recibida por el actor de la semilla");
    require(run.collect_tick(), "Falta el primer paso");
    const auto partial = run.snapshot();
    require(partial.metadata.at("schema_version") == 2, "Falta el contrato de referencia externa");
    auto corrupt = partial;
    corrupt.metadata["initial_rng"]["sampling"][0] = true;
    rejected(
        [&] { static_cast<void>(KlpoTerminalCollector::from_snapshot(inputs, config, corrupt)); });
    corrupt = partial;
    corrupt.metadata["identity"]["initial_rng_sha256"] = std::string(64, '0');
    rejected(
        [&] { static_cast<void>(KlpoTerminalCollector::from_snapshot(inputs, config, corrupt)); });
    corrupt = partial;
    corrupt.metadata["schema_version"] = true;
    rejected(
        [&] { static_cast<void>(KlpoTerminalCollector::from_snapshot(inputs, config, corrupt)); });
    auto factory = KlpoTerminalCollector::from_snapshot(inputs, config, partial);
    KlpoTerminalCollector recovered(inputs, config, parent.frozen_reference(start));
    recovered.restore(partial);
    require(run.collect_tick() && recovered.collect_tick(), "No se terminó la trayectoria");
    require(factory->collect_tick() &&
                serialize_klpo_batch(factory->records()) == serialize_klpo_batch(run.records()),
            "La fábrica de recuperación no conserva el registro");
    require(serialize_klpo_batch(run.records()) == serialize_klpo_batch(recovered.records()),
            "La recuperación perdió el origen real del sampler");
    require(
        at::equal(run.policy().random_state().sampling, recovered.policy().random_state().sampling),
        "La recuperación alteró el RNG del sampler");
    rejected([&] { original.restore(partial); });
    require(parent.optimizer_steps() == 0 && recovered.policy().optimizer_steps() == 0,
            "La prueba ejecutó un paso de optimizador");
}

void an_updatable_actor_is_not_a_collection_reference() {
    const auto inputs = sources();
    KlpoCollectionOptions config;
    KlpoTerminalCollector original(inputs, config);
    PpoPolicy actor(original.policy().observation_width(), {}, 17);
    actor.enable_terminal_adam({.learning_rate = default_ppo_learning_rate, .gradient_norm = 0});
    rejected([&] { KlpoTerminalCollector invalid(inputs, config, std::move(actor)); });
}
} // namespace

int main() {
    try {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        imported_reference_replays_from_its_actual_start_rng(PpoNetworkKind::mlp);
        imported_reference_replays_from_its_actual_start_rng(PpoNetworkKind::gru);
        an_updatable_actor_is_not_a_collection_reference();
        std::cout << "Referencias y RNG inicial recuperados sin actualizaciones\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
