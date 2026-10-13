#include "mars_titan/learning_gate.hpp"

#include <cerrno>
#include <cstring>
#include <vector>

#include <unistd.h>

namespace mars_titan::learning {
int run_learning_gate(bool blocked, std::span<char* const> command,
                      const std::filesystem::path& hold, std::ostream& messages) {
    if (command.empty()) {
        messages << "Uso: mars-titan-learning-gate <prueba> [argumentos]\n";
        return 2;
    }
    if (blocked) {
        // CTest lee este código como salto. La prueba no llega a ejecutarse.
        messages << "Bloqueo de aprendizaje vigente: " << command.front()
                 << " da pasos de optimizador y no se ejecuta mientras " << hold.string()
                 << " no declare training_allowed verdadero\n";
        return learning_gate_skip;
    }
    std::vector<char*> arguments(command.begin(), command.end());
    arguments.push_back(nullptr);
    messages.flush();
    ::execv(arguments.front(), arguments.data());
    // execv solo vuelve si no pudo sustituir el proceso.
    messages << "No se pudo lanzar " << command.front() << ": " << std::strerror(errno) << '\n';
    return learning_gate_unlaunched;
}
} // namespace mars_titan::learning
