#ifndef MARS_TITAN_LEARNING_GATE_HPP
#define MARS_TITAN_LEARNING_GATE_HPP

#include <filesystem>
#include <ostream>
#include <span>

namespace mars_titan::learning {
// Código con el que CTest marca como saltada una prueba que no puede aprender (SKIP_RETURN_CODE).
inline constexpr int learning_gate_skip = 77;
// Código de una orden que no se pudo lanzar, como en las shells POSIX.
inline constexpr int learning_gate_unlaunched = 127;

// Lanzador de las pruebas con pasos de optimizador. Si la protección bloquea, escribe el motivo y
// devuelve learning_gate_skip sin ejecutar nada. Si permite aprender, sustituye el proceso por
// `command` con execv, de modo que el código de salida es el de la prueba, y solo vuelve si no
// puede lanzarla. `command` empieza por la ruta del ejecutable de la prueba.
[[nodiscard]] int run_learning_gate(bool blocked, std::span<char* const> command,
                                    const std::filesystem::path& hold, std::ostream& messages);
} // namespace mars_titan::learning
#endif
