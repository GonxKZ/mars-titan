#include "mars_titan/policy_context.hpp"
#include "mars_titan/ppo_training.hpp"

#include <ATen/ATen.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Comprueba qué guarda el replay de las variantes de valor (Double DQN, QR-DQN y QR-DQN con
// CVaR) recorriendo el entrenador con PpoTrainer::advance. El presupuesto queda por debajo del
// calentamiento y la prueba exige cero pasos de Adam, pero advance es una entrada que puede
// llegar a un paso. Por eso vive aparte de qr_dqn_policy_tests y se registra con la etiqueta
// optimizer-steps: mientras rija el bloqueo de aprendizaje el lanzador la salta (#546, #567).
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
namespace {
using namespace mars_titan::learning;

void require(bool value, std::string_view message) {
    if (!value) {
        throw std::runtime_error(std::string(message));
    }
}

// Cintas de ajuste con `sessions` sesiones y su contexto. Desde `changed_from` cambian los
// precios, las puntuaciones y la señal del contexto, pero no el indicador de negociación,
// de modo que las dos versiones deciden en las mismas sesiones.
std::vector<mars_titan::simulation::BatchInput> learning_sources(std::size_t sessions,
                                                                 std::size_t changed_from) {
    using mars_titan::simulation::ContextTape;
    using mars_titan::simulation::MarketTape;
    std::vector<mars_titan::simulation::BatchInput> result;
    for (const char digest : {'a', 'b', 'c'}) {
        auto market = std::make_shared<MarketTape>();
        market->assets = {"FIC0"};
        market->currency = "USD";
        market->domain = "synthetic";
        market->partition = "train";
        market->source_sha256.assign(64, digest);
        market->parent_id = "analytic-frozen";
        ContextTape context;
        context.source_sha256.assign(64, digest);
        context.fields = {{"trading_enabled", "indicador"}, {"signal", "indicador"}};
        for (std::size_t session = 0; session < sessions; ++session) {
            const bool changed = session >= changed_from;
            const auto time = static_cast<int64_t>(2 * session + 1);
            market->open_times.push_back(time - 1);
            market->close_times.push_back(time);
            market->prediction_times.push_back(time);
            const auto price = (changed ? 25.0 : 10.0) + static_cast<double>(session);
            market->prices.insert(market->prices.end(), {price, price, price, price, 10000.0});
            market->scores.push_back(changed ? -0.04 : 0.01);
            context.values.push_back({session == 0 ? 0.F : 1.F, true, time});
            context.values.push_back({changed ? -7.F : static_cast<float>(session), true, time});
        }
        result.push_back({std::move(market), {}, std::move(context)});
    }
    return result;
}

// Defensa contra mirar más allá del paso emitido: cada transición del replay de valor guarda la
// observación, la acción, la recompensa de ese paso, la terminación y solo la observación
// siguiente. Si se alteran las sesiones desde la k, las transiciones de los pasos t <= k - 2,
// que solo leen hasta la sesión t + 1, deben guardarse igual bit a bit, y la del paso k - 1 debe
// cambiar su observación siguiente. Un retorno a varios pasos o un objetivo que leyera la sesión
// t + 2 rompería la primera condición. No se ejecuta ningún paso del optimizador: el
// presupuesto queda por debajo del calentamiento de 256 decisiones.
void replay_transitions_stop_at_the_next_observation() {
    constexpr std::size_t sessions = 6;
    constexpr std::size_t changed_from = 4;
    for (const auto* variant : {"double_dqn", "qr_dqn", "qr_dqn_cvar"}) {
        PpoLearningOptions options;
        options.enabled = true;
        options.environments = 2;
        options.trading_field = 0;
        options.variant = variant;
        PpoTrainingConfig config;
        config.total_transitions = 24;
        config.rollout_transitions = 8;
        PpoHyperparameters parameters;
        parameters.epochs = 2;
        parameters.minibatch_size = 4;
        struct Step {
            uint32_t lane;
            std::string world;
            uint64_t cursor;
            bool operator==(const Step&) const = default;
        };
        const auto run = [&](std::size_t from) {
            PpoTrainer trainer(learning_sources(sessions, from), config, parameters, "cpu", true,
                               options);
            std::vector<Step> stored;
            const auto observer = [&](std::span<const DecisionRecord> records) {
                for (const auto& record : records) {
                    // El replay guarda, en el mismo orden, las decisiones con aprendizaje permitido
                    // y recompensa válida.
                    if (record.learning_allowed && record.reward_valid) {
                        stored.push_back({record.lane, record.world_sha256, record.cursor});
                    }
                }
            };
            while (trainer.advance(observer)) {
            }
            require(trainer.optimizer_steps() == 0, "La prueba del replay ejecutó un paso de Adam");
            const auto replay = trainer.replay_snapshot();
            require(replay.has_value() && replay->store_next && replay->size == stored.size(),
                    "El replay de valor no corresponde a las decisiones registradas");
            return std::pair{*replay, stored};
        };
        const auto [original, steps] = run(sessions);
        const auto [changed, changed_steps] = run(changed_from);
        require(steps == changed_steps, "Las dos versiones no deciden en los mismos pasos");
        std::size_t kept = 0;
        std::size_t moved = 0;
        for (std::size_t row = 0; row < steps.size(); ++row) {
            const auto index = static_cast<int64_t>(row);
            const auto same = [&](const at::Tensor& left, const at::Tensor& right) {
                return at::equal(left[index], right[index]);
            };
            if (steps[row].cursor + 2 <= changed_from) {
                require(same(original.data.observations, changed.data.observations) &&
                            same(original.data.next_observations, changed.data.next_observations) &&
                            same(original.data.actions, changed.data.actions) &&
                            same(original.data.rewards, changed.data.rewards) &&
                            same(original.data.terminated, changed.data.terminated),
                        "Una transición del replay depende de sesiones posteriores a su paso");
                ++kept;
            } else if (steps[row].cursor + 1 == changed_from) {
                require(same(original.data.observations, changed.data.observations) &&
                            !same(original.data.next_observations, changed.data.next_observations),
                        "La observación siguiente no es la de la sesión t + 1");
                ++moved;
            }
        }
        require(kept > 0 && moved > 0, "La prueba no recorrió pasos anteriores y posteriores al cambio");
    }
}
} // namespace

int main() {
    try {
        replay_transitions_stop_at_the_next_observation();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
