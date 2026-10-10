#include "mars_titan/klpo_learning.hpp"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>

#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

// Oleadas de los objetivos de grupo sobre el controlador de KLPO, sin update_ready ni pasos de
// Adam. Las cintas son fixtures escritos aquí para comprobar agrupación, ventajas, causalidad,
// invariancia por bloques y trazas. No sustituyen a las cintas reales de la etapa.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
namespace {
using namespace mars_titan::learning;
using mars_titan::simulation::BatchInput;

void require(bool value, std::string_view message) {
    if (!value) {
        throw std::runtime_error(std::string(message));
    }
}

template <class F> void rejected(F function, std::string_view message) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(std::string(message));
}

constexpr std::array<std::string_view, 4> identities{"grpo_outcome_v1", "dr_grpo_outcome_v1",
                                                     "dapo_outcome_static_v1", "gspo_outcome_v1"};

// Cinta de un activo con cierres dados. Cada sesión repite el cierre en apertura, máximo y
// mínimo, de modo que el resultado depende de la exposición elegida y de los costes.
BatchInput tape(char digest, const std::vector<double>& closes) {
    using namespace mars_titan::simulation;
    auto market = std::make_shared<MarketTape>();
    market->assets = {"FIXTURE"};
    for (std::size_t index = 0; index < closes.size(); ++index) {
        const auto base = static_cast<int64_t>(index) * 10;
        market->open_times.push_back(base + 1);
        market->close_times.push_back(base + 10);
        for (int field = 0; field < 4; ++field) {
            market->prices.push_back(closes[index]);
        }
        market->prices.push_back(10000);
        market->scores.push_back(.1);
    }
    market->prediction_times = market->close_times;
    market->currency = "USD";
    market->domain = "synthetic";
    market->partition = "train";
    market->parent_id = "fixture";
    market->source_sha256 = std::string(64, digest);
    ContextTape context;
    context.fields = {{"may_trade", "boolean"}};
    context.source_sha256 = std::string(64, 'b');
    for (const auto time : market->close_times) {
        context.values.push_back({1.F, true, time});
    }
    return {market, {}, context};
}

const std::vector<double> rising{10, 12, 9, 11, 13};
const std::vector<double> falling{10, 8, 9, 7, 6};

// Cuatro carriles sobre dos cintas, en el orden de KLPO (carril l, cinta l módulo 2).
std::vector<BatchInput> lanes() {
    return {tape('a', rising), tape('c', falling), tape('a', rising), tape('c', falling)};
}

KlpoLearningConfig settings(std::string_view objective, std::size_t block) {
    KlpoLearningConfig result;
    result.collection.context.enabled = true;
    result.collection.context.variant = "ppo";
    result.collection.context.fold = result.collection.fold;
    result.collection.context.environments = 4;
    result.collection.context.trading_field = 0;
    result.adam = {.learning_rate = .002, .gradient_norm = 0};
    result.gradient_block_episodes = block;
    if (!objective.empty()) {
        result.group = published_group_objective(objective);
    }
    return result;
}

void complete(KlpoLearningController& run) {
    while (run.phase() == KlpoLearningPhase::collecting) {
        static_cast<void>(run.collect_tick());
    }
    require(run.phase() == KlpoLearningPhase::ready, "La oleada del fixture no se completó");
}

void identities_keep_klpo_intact() {
    KlpoLearningController klpo(lanes(), settings({}, 4));
    const auto identity = klpo.identity();
    require(identity.at("kind") == "klpo_terminal_actor_updates" &&
                identity.at("algorithm") == "klpo_full_fresh_waves_v1" &&
                !identity.contains("objective"),
            "La identidad de KLPO cambió al admitir objetivos de grupo");
    for (const auto objective : identities) {
        KlpoLearningController group(lanes(), settings(objective, 4));
        const auto value = group.identity();
        require(value.at("kind") == "group_relative_actor_updates" &&
                    value.at("algorithm") == std::string(group_relative_controller) &&
                    value.at("objective").at("id") == std::string(objective) &&
                    value.at("initial_collection") == identity.at("initial_collection"),
                "El objetivo de grupo no conserva la recogida de KLPO o su identidad");
    }
    auto discounted = settings("grpo_outcome_v1", 4);
    discounted.collection.gamma = .99;
    rejected([&] { KlpoLearningController invalid(lanes(), discounted); },
             "Se aceptó un retorno descontado como resultado del episodio");
}

// Mismos episodios que KLPO, grupos por cinta y ventajas centradas en su grupo.
void waves_group_by_source_and_center_each_group() {
    KlpoLearningController klpo(lanes(), settings({}, 4));
    complete(klpo);
    for (const auto objective : identities) {
        const auto config = published_group_objective(objective);
        KlpoLearningController run(lanes(), settings(objective, 4));
        complete(run);
        // La recogida no depende del objetivo: mismo q, misma semilla y mismas cintas.
        const auto& records = run.wave();
        require(serialize_klpo_batch(records) == serialize_klpo_batch(klpo.wave()),
                "El objetivo de grupo cambió la recogida de la oleada");
        const auto wave = group_wave(records, config);
        require(wave.labels == std::vector<int64_t>{0, 1, 0, 1} &&
                    wave.sizes == std::vector<std::size_t>{2, 2},
                "Los grupos no siguen la cinta de origen");
        const auto returns = klpo_terminal_returns(records);
        for (const int64_t label : {0, 1}) {
            const auto first = static_cast<std::size_t>(label);
            const auto second = first + 2;
            const auto mean = (returns[first] + returns[second]) / 2;
            const auto spread = std::abs(returns[first] - returns[second]) / std::sqrt(2.);
            const auto scale =
                config.advantage() == GroupAdvantage::mean ? 1. : spread + config.advantage_epsilon;
            for (const auto index : {first, second}) {
                const auto expected = (returns[index] - mean) / scale;
                require(std::abs(wave.advantages[static_cast<int64_t>(index)].item<double>() -
                                 expected) <= 1e-12,
                        "La ventaja no está centrada y escalada dentro de su grupo");
            }
        }
    }
}

// El gradiente de la oleada no depende del tamaño de bloque y activar la traza no cambia bits.
void blocks_and_traces_do_not_change_the_gradient() {
    for (const auto objective : identities) {
        KlpoLearningController whole(lanes(), settings(objective, 4));
        KlpoLearningController split(lanes(), settings(objective, 1));
        KlpoLearningController traced(lanes(), settings(objective, 1));
        complete(whole);
        complete(split);
        complete(traced);
        const auto a = whole.backward_ready();
        const auto b = split.backward_ready();
        GroupWaveTrace trace;
        const auto c = traced.backward_ready(&trace);
        require(a.blocks == 1 && b.blocks == 4 && std::abs(a.surrogate - b.surrogate) <= 1e-12 &&
                    b.surrogate == c.surrogate && b.gradient_norm == c.gradient_norm,
                "El bloque o la traza cambiaron el sustituto de grupo");
        const auto first = whole.gradient_snapshot();
        const auto second = split.gradient_snapshot();
        const auto third = traced.gradient_snapshot();
        for (std::size_t index = 0; index < first.size(); ++index) {
            require(at::allclose(first[index], second[index], 1e-5, 1e-7),
                    "La acumulación por bloques cambió el gradiente de grupo");
            require(at::equal(second[index], third[index]),
                    "Activar la traza cambió un bit del gradiente");
        }
        require(trace.groups == 2 && trace.smallest_group == 2 && trace.largest_group == 2 &&
                    trace.objective.episodes == 4 &&
                    trace.objective.decisions == c.sampled_decisions &&
                    std::isfinite(trace.objective.entropy) && trace.objective.entropy > 0 &&
                    std::isfinite(trace.objective.k3_kl) && trace.objective.k3_kl >= 0 &&
                    trace.objective.ratio_min <= trace.objective.ratio_mean &&
                    trace.objective.ratio_mean <= trace.objective.ratio_max &&
                    std::abs(trace.objective.advantage_mean) <= 1e-12,
                "La traza de la oleada no describe la oleada recogida");
        // Al inicio actor y q coinciden, así que el cociente es uno y nada se recorta.
        // q guarda sus probabilidades en FP32, de ahí la tolerancia.
        require(std::abs(trace.objective.ratio_mean - 1) <= 1e-6 &&
                    trace.objective.clip_fraction == 0,
                "El primer cociente de grupo no es uno con actor y q iguales");
    }
    KlpoLearningController klpo(lanes(), settings({}, 1));
    complete(klpo);
    GroupWaveTrace trace;
    rejected([&] { static_cast<void>(klpo.backward_ready(&trace)); },
             "KLPO aceptó una traza de objetivo de grupo");
}

// Defensas adversariales: un grupo debe compartir estado inicial y tener línea base.
void groups_reject_mismatched_starts_and_singletons() {
    const auto config = published_group_objective("grpo_outcome_v1");
    KlpoLearningController run(lanes(), settings("grpo_outcome_v1", 4));
    complete(run);
    auto forged = run.wave();
    forged.episodes[2].steps.front().observation.front() += 1;
    rejected([&] { static_cast<void>(group_wave(forged, config)); },
             "Se aceptó un grupo con observaciones iniciales distintas");
    std::vector<BatchInput> single{tape('a', rising), tape('c', falling)};
    auto settings_single = settings("grpo_outcome_v1", 2);
    settings_single.collection.context.environments = 2;
    KlpoLearningController lonely(single, settings_single);
    complete(lonely);
    rejected([&] { static_cast<void>(lonely.backward_ready()); },
             "Se aceptó un grupo de un solo episodio");
}

// Causalidad: alterar la cinta después de la sesión k no cambia nada recogido hasta k. La
// ventaja de grupo usa el resultado completo, pero solo después de cerrar la oleada.
void collection_before_a_session_ignores_later_prices() {
    auto altered = rising;
    altered.back() = 3;
    const std::vector<BatchInput> base{tape('a', rising), tape('c', falling), tape('a', rising),
                                       tape('c', falling)};
    const std::vector<BatchInput> future{tape('a', altered), tape('c', falling), tape('a', altered),
                                         tape('c', falling)};
    KlpoLearningController first(base, settings("dapo_outcome_static_v1", 4));
    KlpoLearningController second(future, settings("dapo_outcome_static_v1", 4));
    rejected([&] { static_cast<void>(first.backward_ready()); },
             "Se calculó un objetivo de grupo con la oleada incompleta");
    complete(first);
    complete(second);
    const auto& left = first.wave();
    const auto& right = second.wave();
    bool changed = false;
    for (const std::size_t lane : {std::size_t{0}, std::size_t{2}}) {
        const auto& a = left.episodes[lane].steps;
        const auto& b = right.episodes[lane].steps;
        require(a.size() == b.size(), "La cinta alterada cambió la longitud del episodio");
        // El último paso es el único cuyo resultado depende del cierre alterado.
        for (std::size_t step = 0; step + 1 < a.size(); ++step) {
            require(a[step].observation == b[step].observation &&
                        a[step].behavior == b[step].behavior && a[step].action == b[step].action &&
                        a[step].reward == b[step].reward,
                    "Una decisión anterior al cambio vio precios posteriores");
        }
        require(a.back().observation == b.back().observation && a.back().action == b.back().action,
                "El último paso vio el cierre que todavía no se había producido");
        changed = changed || a.back().reward != b.back().reward;
    }
    require(changed, "El fixture no alteró ningún resultado posterior");
}
} // namespace

int main() {
    try {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        identities_keep_klpo_intact();
        waves_group_by_source_and_center_each_group();
        blocks_and_traces_do_not_change_the_gradient();
        groups_reject_mismatched_starts_and_singletons();
        collection_before_a_session_ignores_later_prices();
        std::cout << "Oleadas de grupo contrastadas sin pasos de optimizador\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
