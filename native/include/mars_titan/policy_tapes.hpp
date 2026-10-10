#ifndef MARS_TITAN_POLICY_TAPES_HPP
#define MARS_TITAN_POLICY_TAPES_HPP

#include "mars_titan/financial_batch.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <filesystem>
#include <span>
#include <string_view>

namespace mars_titan::learning {

// Papel de una cinta reconstruida en una política por ventana walk-forward. La evaluación
// usa la partición de validación del lector porque el test sellado permanece cerrado.
enum class PolicyTapeRole : uint8_t { train, validation, evaluation };

struct PolicyTape {
    simulation::BatchInput input;
    // Papel, manifiesto, Parquet, base histórica y límites temporales de la cinta.
    nlohmann::json identity;
};

[[nodiscard]] std::string_view policy_tape_role_name(PolicyTapeRole role) noexcept;

// Comprueba el manifiesto antes de abrir el Parquet: formato, test cerrado, dominio real y
// partición del papel. Devuelve los activos y sesiones declarados para presupuestar memoria.
void require_policy_tape_manifest(const nlohmann::json& manifest, PolicyTapeRole role);

// Lee una cinta reconstruida verificada por el lector nativo. Rechaza contextos observables,
// porque la etapa no los declara, y cualquier cambio del manifiesto durante la lectura.
[[nodiscard]] PolicyTape load_policy_tape(const std::filesystem::path& directory,
                                          PolicyTapeRole role,
                                          const simulation::Parameters& parameters);

// Las cintas de una política comparten activos, moneda, reglas y base histórica, no se repiten
// y cada una termina antes de que empiece la siguiente en el orden recibido.
void require_policy_sequence(std::span<const PolicyTape> tapes);

// La evaluación empieza después de la última sesión de las cintas que eligieron la política.
// `selection` contiene las identidades de ajuste y validación registradas por la ejecución.
void require_after_selection(const nlohmann::json& selection, const PolicyTape& evaluation);

} // namespace mars_titan::learning
#endif
