#ifndef MARS_TITAN_POLICY_COMMAND_HPP
#define MARS_TITAN_POLICY_COMMAND_HPP

#include "mars_titan/ppo_experiment.hpp"

#include <nlohmann/json.hpp>

#include <functional>
#include <string_view>
#include <vector>

namespace mars_titan::learning {

using PolicyRunner =
    std::function<nlohmann::json(const PpoExperimentOptions&, const std::function<bool()>&)>;

// Orden de un experimento de política con la interfaz común de mars-titan-ppo: configuración,
// cintas, auditoría separada, dispositivo, pausa, reanudación y vigilancia del padre.
struct PolicyCommand {
    std::string_view name;
    std::string_view help;
    // Capacidades que la etapa comprueba con --capabilities, sin leer datos.
    std::vector<std::string_view> capabilities;
    PolicyRunner run;
};

// --capabilities publica nombre, capacidades e identidad de compilación en JSON.
[[nodiscard]] nlohmann::json policy_capabilities(const PolicyCommand& command);

// Devuelve 0 al terminar, 2 en pausa y 1 si falla, como mars-titan-ppo.
[[nodiscard]] int run_policy_command(int argc, char** argv, const PolicyCommand& command);

} // namespace mars_titan::learning
#endif
