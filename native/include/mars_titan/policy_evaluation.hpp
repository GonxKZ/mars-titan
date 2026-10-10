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

// Costes de evaluación declarados por la etapa de políticas, en puntos básicos.
inline constexpr std::array<double, 3> frozen_evaluation_costs{0, 10, 25};

struct FrozenEvaluationRequest {
    std::filesystem::path output;
    bool resume = false;
    // Ejecución elegida, huella de la política, cintas y costes. Se sella antes de evaluar.
    nlohmann::json identity;
    std::vector<PolicyTape> tapes;
    PpoLearningOptions learning;
    std::size_t workers = 1;
};

// Evalúa con argmax la política congelada en cada cinta y coste, sin aprendizaje ni
// muestreo, y publica un episodio por cinta y coste, también si falla o se arruina.
// Una pausa no confirma episodios. Al reanudar se repite la evaluación completa.
[[nodiscard]] nlohmann::json run_frozen_evaluation(const PpoPolicy& policy,
                                                   const FrozenEvaluationRequest& request,
                                                   const std::function<bool()>& stop = {});

} // namespace mars_titan::learning
#endif
