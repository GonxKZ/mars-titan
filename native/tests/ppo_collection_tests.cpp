#include "mars_titan/ppo_training.hpp"

#include <ATen/ATen.h>

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>

// Recogida PPO hasta antes de su primera actualización, sin Adam ni pasos de optimizador.
// Una política independiente con la misma semilla recalcula cada paso del recorrido.
namespace {
using namespace mars_titan::learning;
using namespace mars_titan::simulation;
constexpr std::size_t short_sessions = 4;
constexpr std::size_t long_sessions = 7;
constexpr std::size_t diagnostic_transitions = 32;
constexpr double price_base = 10;
constexpr double price_step = 0.5;
constexpr double volume = 10000;
constexpr std::size_t digest_size = 64;
constexpr double score_scale = 0.01;
constexpr uint64_t seed = 7;

void require(bool condition, const char* reason) {
    if (!condition) {
        throw std::runtime_error(reason);
    }
}

std::shared_ptr<MarketTape> tape(std::size_t count, const std::string& source) {
    auto data = std::make_shared<MarketTape>();
    data->assets = {"FIC0"};
    data->currency = "USD";
    data->domain = "synthetic";
    data->partition = "train";
    data->source_sha256 = source;
    data->parent_id = "analytic-frozen";
    for (std::size_t at = 0; at < count; ++at) {
        data->open_times.push_back(static_cast<int64_t>(2 * at));
        data->close_times.push_back(static_cast<int64_t>(2 * at + 1));
        data->prediction_times.push_back(static_cast<int64_t>(2 * at + 1));
        // Precios y puntuaciones distintos en cada sesión para que cada observación cambie.
        const auto price = price_base + price_step * static_cast<double>(at * at);
        data->prices.insert(data->prices.end(), {price, price, price, price, volume});
        data->scores.push_back(score_scale * (static_cast<double>(at % 3) - 1));
    }
    return data;
}

PpoObjectiveConfig objective() {
    PpoObjectiveConfig result;
    result.kind = PpoObjectiveKind::clip_full_kl;
    return result;
}

// Logits, valores y pesos de cada paso con una política nueva de la misma semilla, y la
// secuencia de acciones que produce su generador con un muestreo por paso.
void collection_matches_an_independent_replay() {
    const std::vector<BatchInput> inputs{{tape(short_sessions, std::string(digest_size, 'a')), {}, {}},
                                         {tape(long_sessions, std::string(digest_size, 'b')), {}, {}}};
    PpoTrainingConfig config;
    config.total_transitions = diagnostic_transitions;
    config.rollout_transitions = diagnostic_transitions;
    config.seed = seed;
    const PpoHyperparameters parameters;
    PpoTrainer trainer(inputs, config, parameters, "cpu", true, {}, objective());
    const auto ticks = diagnostic_transitions / inputs.size() - 1;
    for (std::size_t tick = 0; tick < ticks; ++tick) {
        require(trainer.advance(), "La recogida terminó antes de lo previsto");
    }
    require(trainer.optimizer_steps() == 0 && trainer.policy().optimizer_steps() == 0 &&
                trainer.partial_ticks() == ticks,
            "La prueba alcanzó una actualización");
    const auto state = trainer.snapshot();
    const auto& rollout = state.rollout;
    require(state.episodes >= 4, "Las cintas cortas deben reiniciar varios episodios");
    const auto width = static_cast<std::size_t>(rollout.observations.size(2));
    PpoPolicy replay(width, parameters, seed, "cpu", default_ppo_memory_bytes, PpoArchitecture{}, objective());
    require(replay.parameter_fingerprint() == trainer.policy().parameter_fingerprint(),
            "La política independiente no parte de los mismos pesos");
    const auto lanes = static_cast<int64_t>(inputs.size());
    for (std::size_t tick = 0; tick < ticks; ++tick) {
        const auto time = static_cast<int64_t>(tick);
        const auto output = replay.infer(rollout.observations[time]);
        const auto chosen = replay.sample(output);
        const auto packed = chosen.packed.contiguous();
        require(at::equal(packed.select(1, 0).to(at::kLong), rollout.actions[time]),
                "La secuencia de acciones no coincide con un muestreo por paso");
        require(at::equal(packed.select(1, 1), rollout.old_log_probabilities[time]) &&
                    at::equal(packed.select(1, 2), rollout.old_values[time]) &&
                    at::equal(chosen.probabilities, rollout.old_action_weights[time]),
                "Logits o valores del paso no coinciden con un forward sobre su observación");
        if (tick + 1 == ticks) {
            continue;
        }
        for (int64_t lane = 0; lane < lanes; ++lane) {
            const bool boundary = rollout.terminated[time][lane].item<bool>() ||
                                  rollout.truncated[time][lane].item<bool>();
            if (!boundary) {
                require(rollout.next_values[time][lane].item<double>() ==
                            rollout.old_values[time + 1][lane].item<double>(),
                        "El bootstrap no coincide con el valor del paso siguiente");
            }
        }
    }
    require(at::equal(replay.random_state().sampling, trainer.policy().random_state().sampling),
            "La recogida consumió el generador de otra forma");
}

// Reanudar a mitad del recorrido no reutiliza un forward anterior a la restauración.
void restored_collection_continues_identically() {
    const std::vector<BatchInput> inputs{{tape(short_sessions, std::string(digest_size, 'a')), {}, {}},
                                         {tape(long_sessions, std::string(digest_size, 'b')), {}, {}}};
    PpoTrainingConfig config;
    config.total_transitions = diagnostic_transitions;
    config.rollout_transitions = diagnostic_transitions;
    config.seed = seed;
    PpoTrainer continuous(inputs, config, PpoHyperparameters{}, "cpu", true, {}, objective());
    PpoTrainer interrupted(inputs, config, PpoHyperparameters{}, "cpu", true, {}, objective());
    const auto ticks = diagnostic_transitions / inputs.size() - 1;
    for (std::size_t tick = 0; tick < ticks / 2; ++tick) {
        require(continuous.advance() && interrupted.advance(), "Faltan pasos antes de la pausa");
    }
    const auto saved = interrupted.snapshot();
    // Un paso más en el original deja su forward guardado con otra observación.
    require(interrupted.advance(), "Falta el paso posterior a la pausa");
    interrupted.restore(saved);
    for (std::size_t tick = ticks / 2; tick < ticks; ++tick) {
        require(continuous.advance() && interrupted.advance(), "Faltan pasos después de la pausa");
    }
    const auto first = continuous.snapshot().rollout;
    const auto second = interrupted.snapshot().rollout;
    require(at::equal(first.actions, second.actions) &&
                at::equal(first.old_log_probabilities, second.old_log_probabilities) &&
                at::equal(first.next_values, second.next_values) &&
                at::equal(first.old_action_weights, second.old_action_weights),
            "La restauración cambia la recogida");
}
} // namespace

int main() {
    try {
        collection_matches_an_independent_replay();
        restored_collection_continues_identically();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
