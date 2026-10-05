#include "mars_titan/memory_stress.hpp"

#include <ATen/Parallel.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <utility>

namespace {
using namespace mars_titan::stress;
using namespace mars_titan::learning;
using Json = nlohmann::json;
constexpr uint64_t fixture_seed = 7;
constexpr uint64_t alternate_seed = 9;
constexpr uint64_t stream_length = 8;
constexpr double retained_priority = 10;
constexpr int64_t query_cutoff = 100;
constexpr int64_t next_confirmation = 5;
constexpr uint64_t replacement_id = 5;
constexpr uint64_t uniform_trials = 512;
constexpr unsigned minimum_evictions = 75;
constexpr unsigned maximum_evictions = 185;
constexpr double expected_cosine = 0.6;
constexpr double cosine_tolerance = 1e-7;
constexpr std::size_t small_steps = 320;
constexpr std::size_t small_capacity = 16;
constexpr std::size_t small_regime_length = 64;
constexpr std::size_t feedback_delay = 7;
constexpr std::size_t invalid_period = 19;
constexpr uint64_t expected_invalid = 16;
constexpr uint64_t expected_valid = 304;
constexpr std::size_t pause_cursor = 133;
constexpr uint64_t other_seed = 999;
constexpr double perturbed_truth = 1000;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}
template <typename Function> void rejected(Function&& function) {
    bool failed = false;
    try {
        std::forward<Function>(function)();
    } catch (const std::exception&) {
        failed = true;
    }
    require(failed, "Se aceptó una entrada inválida");
}

MemoryRecord record(uint64_t id) {
    MemoryRecord result;
    result.id = id;
    result.decision_at = static_cast<int64_t>(2 * id);
    result.available_at = result.decision_at;
    result.maturity_at = result.decision_at + 1;
    result.key[0] = 1;
    result.value[0] = static_cast<float>(id);
    result.label = static_cast<double>(id);
    result.label_valid = true;
    return result;
}

void policies_write_same_candidates_and_keep_different_history() {
    RetentionBank recent(Retention::recent, 2, fixture_seed);
    RetentionBank selective(Retention::selective, 2, fixture_seed);
    RetentionBank uniform(Retention::uniform, 2, fixture_seed);
    for (uint64_t id = 1; id <= stream_length; ++id) {
        const auto sample = record(id);
        for (auto* bank : {&recent, &selective, &uniform}) {
            bank->write(sample, sample.maturity_at, id == 1 ? retained_priority : 0);
        }
    }
    for (const auto* bank : {&recent, &selective, &uniform}) {
        require(bank->writes() == stream_length && bank->size() == 2,
                "La política alteró las escrituras o excedió la capacidad");
    }
    const auto latest = recent.query(record(1).key, query_cutoff);
    require(latest.count == 2 && latest.neighbors[0].record.id == stream_length - 1 &&
                latest.neighbors[1].record.id == stream_length,
            "La política reciente no conserva las dos últimas entradas");
    const auto retained = selective.query(record(1).key, query_cutoff);
    require(retained.count == 2 && retained.neighbors[0].record.id == 1 &&
                retained.neighbors[1].record.id == stream_length,
            "La política selectiva pierde la prioridad madura o rechaza al nuevo candidato");
}

void invalid_and_future_records_leave_bank_unchanged() {
    RetentionBank bank(Retention::recent, 4, alternate_seed);
    bank.write(record(1), 3, 1);
    const auto before = bank.checkpoint();
    rejected([&] { bank.write(record(2), 4, 0); });
    rejected([&] { bank.write(record(1), next_confirmation, 0); });
    rejected([&] { bank.write(record(2), next_confirmation, -1); });
    auto bad = record(2);
    bad.key.fill(0);
    rejected([&] { bank.write(bad, next_confirmation, 0); });
    bad = record(2);
    bad.label = std::numeric_limits<double>::quiet_NaN();
    rejected([&] { bank.write(bad, next_confirmation, 0); });
    bad = record(2);
    bad.value[3] = std::numeric_limits<float>::infinity();
    rejected([&] { bank.write(bad, next_confirmation, 0); });
    require(bank.checkpoint() == before, "Una entrada rechazada cambia memoria o RNG");
    require(bank.query(record(1).key, 2).count == 0 && bank.query(record(1).key, 3).count == 1,
            "La consulta admite una etiqueta antes de madurar");
    rejected([&] { static_cast<void>(bank.query(MemoryVector{}, 4)); });

    RetentionBank delayed(Retention::recent, 4, alternate_seed);
    auto later = record(1);
    constexpr int64_t late_maturity = 11;
    later.maturity_at = late_maturity;
    delayed.write(later, late_maturity, 0);
    require(delayed.query(later.key, next_confirmation).count == 0 &&
                delayed.query(later.key, late_maturity).count == 1,
            "La consulta intermedia revela una etiqueta antes de su maduración");
}

void uniform_eviction_is_not_a_fixed_slot() {
    std::array<unsigned, 4> evictions{};
    for (uint64_t seed = 0; seed < uniform_trials; ++seed) {
        RetentionBank bank(Retention::uniform, 4, seed);
        for (uint64_t id = 1; id <= replacement_id; ++id) {
            bank.write(record(id), static_cast<int64_t>(2 * id + 1), 0);
        }
        const auto query = bank.query(record(1).key, 20);
        for (uint64_t id = 1; id <= 4; ++id) {
            bool found = false;
            for (std::size_t index = 0; index < query.count; ++index) {
                found = found || query.neighbors.at(index).record.id == id;
            }
            if (!found) {
                ++evictions.at(id - 1);
            }
        }
    }
    require(std::all_of(evictions.begin(), evictions.end(),
                        [](unsigned value) {
                            return value > minimum_evictions && value < maximum_evictions;
                        }),
            "La expulsión uniforme presenta un sesgo incompatible con las semillas probadas");
}

void query_matches_hand_computed_cosines() {
    RetentionBank bank(Retention::recent, 4, alternate_seed);
    auto first = record(1);
    first.key[0] = 3;
    first.key[1] = 4;
    bank.write(first, 3, 0);
    auto second = record(2);
    second.key[0] = -1;
    bank.write(second, next_confirmation, 0);
    const auto result = bank.query(record(1).key, 6);
    require(result.neighbors[0].record.id == 1 &&
                std::abs(result.neighbors[0].similarity - expected_cosine) < cosine_tolerance &&
                result.neighbors[1].record.id == 2 && result.neighbors[1].similarity == -1,
            "La consulta no conserva cosenos y orden");
}

Config small_config() {
    Config config;
    config.steps = small_steps;
    config.capacity = small_capacity;
    config.regime_length = small_regime_length;
    config.delay = feedback_delay;
    config.invalid_every = invalid_period;
    return config;
}

Json comparable(const Experiment& experiment) {
    auto report = Json::parse(experiment.report());
    report.erase("measurements");
    return report;
}

Json recoverable_state(const Experiment& experiment) {
    auto state = Json::parse(experiment.checkpoint());
    state.erase("measurements");
    return state;
}

void interruption_restores_pending_predictions_rng_and_memory() {
    for (const auto scenario :
         {Scenario::recurrence, Scenario::persistent, Scenario::noise, Scenario::outliers}) {
        auto config = small_config();
        config.scenario = scenario;
        Experiment full(config);
        full.run_until(config.steps);
        for (const std::size_t stop : {std::size_t{1}, pause_cursor, config.steps}) {
            Experiment partial(config);
            partial.run_until(stop);
            const auto checkpoint = partial.checkpoint();
            require(read_config(checkpoint) == config, "El checkpoint pierde la configuración");
            Experiment resumed(config);
            resumed.restore(checkpoint);
            resumed.run_until(config.steps);
            require(comparable(full) == comparable(resumed),
                    "Reanudar cambia predicciones, pendientes, memoria o métricas");
            require(recoverable_state(full) == recoverable_state(resumed),
                    "Reanudar cambia claves, prioridades, RNG o estado confirmado");
        }
        const auto result = comparable(full);
        require(result.at("completed") == true && result.at("pending") == 0,
                "Quedan etiquetas pendientes tras completar el recorrido");
        require(result.at("invalid_observations") == expected_invalid,
                "No se registran observaciones inválidas");
        for (const auto& policy : result.at("policies")) {
            require(policy.at("writes") == expected_valid && policy.at("size") == small_capacity &&
                        policy.at("evaluated") == expected_valid,
                    "Las políticas no comparten candidatos, etiquetas y capacidad saturada");
        }
    }
}

void independent_streams_keep_data_fixed_when_retention_seed_changes() {
    auto config = small_config();
    Experiment first(config);
    first.run_until(config.steps);
    ++config.retention_seed;
    Experiment changed(config);
    changed.run_until(config.steps);
    const auto a = comparable(first);
    const auto b = comparable(changed);
    require(a.at("data_digest") == b.at("data_digest"),
            "La selección de memoria consume el RNG de datos");
    require(a.at("policies").at(1) == b.at("policies").at(1) &&
                a.at("policies").at(2) == b.at("policies").at(2),
            "La semilla de expulsión aleatoria altera controles deterministas");
    ++config.data_seed;
    Experiment different(config);
    different.run_until(config.steps);
    require(a.at("data_digest") != comparable(different).at("data_digest"),
            "Cambiar la semilla de datos conserva el mismo stream");
}

void checkpoints_reject_corruption_incompatible_config_and_future_state() {
    const auto config = small_config();
    Experiment experiment(config);
    experiment.run_until(pause_cursor);
    const auto before = experiment.checkpoint();
    rejected([&] { experiment.restore(before.substr(0, before.size() / 2)); });
    auto wrong = Json::parse(before);
    wrong["cursor"] = config.steps + 1;
    rejected([&] { experiment.restore(wrong.dump()); });
    wrong = Json::parse(before);
    wrong["config"]["data_seed"] = other_seed;
    rejected([&] { experiment.restore(wrong.dump()); });
    wrong = Json::parse(before);
    wrong["pending"][0]["record"]["maturity_at"] = 0;
    rejected([&] { experiment.restore(wrong.dump()); });
    wrong = Json::parse(before);
    wrong["banks"][0]["last_id"] = config.steps;
    rejected([&] { experiment.restore(wrong.dump()); });
    wrong = Json::parse(before);
    wrong["cursor"] = 0;
    wrong["invalid_observations"] = 0;
    rejected([&] { experiment.restore(wrong.dump()); });
    require(experiment.checkpoint() == before, "La recuperación fallida cambia el estado vigente");
    rejected([&] { experiment.run_until(pause_cursor - 1); });
    auto invalid = config;
    invalid.capacity = 0;
    rejected([&] { Experiment unusable(invalid); });
    invalid = config;
    invalid.noise = std::numeric_limits<double>::infinity();
    rejected([&] { Experiment unusable(invalid); });
}

void latent_evaluation_truth_never_changes_predictions_or_retention() {
    auto config = small_config();
    Experiment original(config);
    original.run_until(pause_cursor);
    auto snapshot = Json::parse(original.checkpoint());
    for (auto& pending : snapshot.at("pending")) {
        pending["truth_evaluation_only"] = perturbed_truth;
    }
    Experiment perturbed(config);
    perturbed.restore(snapshot.dump());
    original.run_until(config.steps);
    perturbed.run_until(config.steps);
    const auto expected = comparable(original);
    const auto observed = comparable(perturbed);
    for (std::size_t index = 0; index < 3; ++index) {
        const auto& a = expected.at("policies").at(index);
        const auto& b = observed.at("policies").at(index);
        for (std::string_view field : {"prediction_digest", "retained_digest", "mae_observed"}) {
            require(a.at(field) == b.at(field),
                    "La verdad latente contamina predicción o retención");
        }
        require(a.at("mae_latent_evaluation_only") != b.at("mae_latent_evaluation_only"),
                "La perturbación no ejercita la verdad reservada al evaluador");
    }
}

void noise_control_has_zero_latent_signal_and_predicts_before_feedback() {
    auto config = small_config();
    config.scenario = Scenario::noise;
    config.noise = 0;
    config.invalid_every = 0;
    Experiment experiment(config);
    experiment.run_until(config.steps);
    const auto report = comparable(experiment);
    for (const auto& policy : report.at("policies")) {
        require(policy.at("mae_observed") == 0 && policy.at("mae_latent_evaluation_only") == 0,
                "El control sin señal inventa señal latente");
    }
    Experiment prefix(small_config());
    prefix.run_until(1);
    const auto saved = Json::parse(prefix.checkpoint());
    require(saved.at("pending").at(0).at("predictions") == Json::array({0, 0, 0}),
            "La primera predicción consulta su propia etiqueta futura");
}
} // namespace

int main() {
    try {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        policies_write_same_candidates_and_keep_different_history();
        invalid_and_future_records_leave_bank_unchanged();
        uniform_eviction_is_not_a_fixed_slot();
        query_matches_hand_computed_cosines();
        interruption_restores_pending_predictions_rng_and_memory();
        independent_streams_keep_data_fixed_when_retention_seed_changes();
        checkpoints_reject_corruption_incompatible_config_and_future_state();
        latent_evaluation_truth_never_changes_predictions_or_retention();
        noise_control_has_zero_latent_signal_and_predicts_before_feedback();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Retención, saturación, ruido, cronología y recuperación comprobados\n";
}
