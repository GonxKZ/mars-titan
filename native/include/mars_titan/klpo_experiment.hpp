#ifndef MARS_TITAN_KLPO_EXPERIMENT_HPP
#define MARS_TITAN_KLPO_EXPERIMENT_HPP

#include "mars_titan/klpo_learning.hpp"
#include "mars_titan/ppo_experiment.hpp"

#include <nlohmann/json.hpp>

#include <functional>

namespace mars_titan::learning {

// KLPO terminal sobre cintas reconstruidas por ventana, con la interfaz de mars-titan-ppo.
// Recoge oleadas completas de un episodio por entorno, que recorre en ciclo las cintas de
// ajuste. Consume las oleadas enteras que caben en el presupuesto declarado, evalúa con
// argmax en validación según el criterio de cartera y conserva la mejor política. Con
// --audit-run evalúa la política elegida en cintas posteriores, sin aprendizaje.
// Comprueba la protección local del aprendizaje antes de leer cintas o crear salidas.
// Una pausa en una oleada completa se confirma antes de update_ready, nunca dentro.
[[nodiscard]] nlohmann::json run_klpo_experiment(const PpoExperimentOptions& options,
                                                 const std::function<bool()>& stop = {});

} // namespace mars_titan::learning
#endif
