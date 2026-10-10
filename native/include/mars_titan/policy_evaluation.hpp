#ifndef MARS_TITAN_POLICY_EVALUATION_HPP
#define MARS_TITAN_POLICY_EVALUATION_HPP

#include "mars_titan/policy_tapes.hpp"
#include "mars_titan/ppo_training.hpp"

#include <nlohmann/json.hpp>

#include <array>
#include <filesystem>
#include <functional>
#include <vector>

namespace mars_titan::learning {

// Costes de evaluación por omisión, en puntos básicos. La etapa puede declarar otros con
// --evaluation-cost, que forman parte de la identidad sellada de la evaluación.
inline constexpr std::array<double, 3> frozen_evaluation_costs{0, 10, 25};
inline constexpr std::size_t maximum_evaluation_costs = 16;
inline constexpr double maximum_evaluation_cost_bps = 1000;

// Costes declarados o los de omisión. Exige entre 1 y 16 costes finitos, distintos, no
// negativos, de como mucho 1000 pb y en orden creciente.
[[nodiscard]] std::vector<double> frozen_costs(const std::vector<double>& declared);

struct FrozenEvaluationRequest {
    std::filesystem::path output;
    bool resume = false;
    // Ejecución elegida, huella de la política, cintas y costes. Se sella antes de evaluar.
    nlohmann::json identity;
    std::vector<PolicyTape> tapes;
    PpoLearningOptions learning;
    std::size_t workers = 1;
    std::vector<double> costs{frozen_evaluation_costs.begin(), frozen_evaluation_costs.end()};
};

// Evalúa con argmax la política congelada en cada cinta y coste, sin aprendizaje ni
// muestreo, y publica un episodio por cinta y coste, también si falla o se arruina.
// Cada episodio conserva su patrimonio en cada cierre, reconstruido con las recompensas
// logarítmicas de la sesión (NaN tras un cierre ausente y cero tras la ruina).
// Una pausa no confirma episodios. Al reanudar se repite la evaluación completa.
[[nodiscard]] nlohmann::json run_frozen_evaluation(const PpoPolicy& policy,
                                                   const FrozenEvaluationRequest& request,
                                                   const std::function<bool()>& stop = {});

} // namespace mars_titan::learning
#endif
