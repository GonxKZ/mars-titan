#include "mars_titan/learning_gate.hpp"
#include "mars_titan/learning_hold.hpp"

#include <cstddef>
#include <exception>
#include <iostream>
#include <span>

// Lanzador común de las pruebas de CTest con la etiqueta optimizer-steps (#567). Lee la misma
// protección que los binarios y, si bloquea, termina con el código de salto antes de ejecutar la
// prueba. Una protección que no se puede leer detiene la prueba como fallo, sin lanzarla.
int main(int argc, char** argv) {
    using namespace mars_titan::learning;
    const std::span<char* const> command(argv + 1, static_cast<std::size_t>(argc > 0 ? argc - 1 : 0));
    try {
        const auto hold = learning_hold_path();
        return run_learning_gate(learning_blocked(hold), command, hold, std::cerr);
    } catch (const std::exception& error) {
        std::cerr << "No se pudo leer la protección de aprendizaje: " << error.what() << '\n';
        return 1;
    }
}
