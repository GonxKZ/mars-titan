#ifndef MARS_TITAN_LEARNING_HOLD_HPP
#define MARS_TITAN_LEARNING_HOLD_HPP

#include <filesystem>
#include <string_view>

namespace mars_titan::learning {
// Misma protección local que src/mars_titan/training/learning_hold.py. La variable
// MARS_TITAN_TRAINING_HOLD sustituye a ~/.local/state/mars-titan/training-hold-2000.json.
[[nodiscard]] std::filesystem::path learning_hold_path();

// Un archivo ausente no bloquea. Uno presente bloquea salvo que training_allowed sea true.
// Un campo ausente o no booleano, o un documento inválido, lanza una excepción.
[[nodiscard]] bool learning_blocked(const std::filesystem::path& path);

// Se llama antes de leer fuentes, crear salidas o ejecutar pasos de optimizador.
void require_learning_allowed(std::string_view action);

// Cerrojo de solo evaluación. Las órdenes que no aprenden lo activan antes de leer fuentes y,
// desde entonces, el proceso no puede crear un entrenador ni ejecutar un paso de optimizador.
// La protección no se rebaja ni se consulta: lo que admite el modo lo garantiza el cerrojo.
// No se desactiva, porque la orden termina con el proceso.
void enter_evaluation_only() noexcept;
[[nodiscard]] bool evaluation_only() noexcept;

// Se llama al crear un entrenador o un optimizador con estado y al empezar cada paso.
void require_optimizer_steps_allowed(std::string_view action);
} // namespace mars_titan::learning
#endif
