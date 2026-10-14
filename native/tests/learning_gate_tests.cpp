#include "mars_titan/learning_gate.hpp"

#include <array>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

// Lanzador de las pruebas con pasos de optimizador (#567). Ninguna comprobación ejecuta una prueba
// real ni escribe una protección que permita aprender. En su lugar se lanza este mismo programa
// con --sentinel, que solo deja una marca y termina con un código propio. El camino que permite
// aprender se comprueba llamando a run_learning_gate con la protección ya decidida.
namespace {
using mars_titan::learning::learning_gate_skip;
using mars_titan::learning::run_learning_gate;
constexpr int sentinel_code = 23;

void require(bool condition, const char* reason) {
    if (!condition) {
        throw std::runtime_error(reason);
    }
}

// Código de salida de un proceso hijo que ejecuta `body`, o -1 si no terminó con normalidad.
template <typename Body> int in_child(Body&& body) {
    std::cout.flush();
    std::cerr.flush();
    const pid_t child = ::fork();
    require(child >= 0, "No se pudo crear el proceso hijo");
    if (child == 0) {
        ::_exit(body());
    }
    int status = 0;
    require(::waitpid(child, &status, 0) == child, "No se pudo esperar al proceso hijo");
    return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

struct Sentinel {
    std::filesystem::path marker;
    std::vector<std::string> words;
    std::vector<char*> command;

    Sentinel(const std::filesystem::path& self, std::filesystem::path path)
        : marker(std::move(path)), words{self.string(), "--sentinel", marker.string()} {
        std::filesystem::remove(marker);
        for (auto& word : words) {
            command.push_back(word.data());
        }
    }
    [[nodiscard]] bool ran() const { return std::filesystem::exists(marker); }
};

// Con la protección que bloquea no se lanza nada y el código es el de salto, con su motivo. La
// llamada va en un proceso hijo para que un lanzador roto no sustituya a esta prueba.
void a_blocking_hold_skips_without_launching(const std::filesystem::path& self,
                                             const std::filesystem::path& root) {
    const Sentinel sentinel(self, root / "blocked.marker");
    const int code = in_child([&] {
        std::ostringstream messages;
        const int result = run_learning_gate(true, sentinel.command, "/hold.json", messages);
        const bool explained = messages.str().starts_with("Bloqueo de aprendizaje vigente: ") &&
                               messages.str().find("/hold.json") != std::string::npos;
        return explained ? result : 1;
    });
    require(!sentinel.ran(), "La protección que bloquea lanzó la prueba");
    require(code == learning_gate_skip,
            "La protección que bloquea no devolvió el código de salto con su motivo");
    std::ostringstream usage;
    require(run_learning_gate(false, {}, "/hold.json", usage) == 2 &&
                usage.str().starts_with("Uso:"),
            "El lanzador aceptó una orden vacía");
}

// Con una protección que permite aprender, decidida aquí y sin archivo, el lanzador sustituye el
// proceso por la prueba y su código de salida es el de la prueba.
void an_allowing_hold_yields_to_the_test(const std::filesystem::path& self,
                                          const std::filesystem::path& root) {
    const Sentinel sentinel(self, root / "allowed.marker");
    const int code = in_child([&] {
        std::ostringstream messages;
        return run_learning_gate(false, sentinel.command, "/hold.json", messages);
    });
    require(code == sentinel_code && sentinel.ran(),
            "El lanzador no cedió el control a la prueba ni conservó su código");
    std::string missing = (root / "missing").string();
    const std::array<char*, 1> absent{missing.data()};
    require(in_child([&] {
                std::ostringstream messages;
                return run_learning_gate(false, absent, "/hold.json", messages);
            }) == mars_titan::learning::learning_gate_unlaunched,
            "Un ejecutable ausente no devolvió 127");
}

// El programa del lanzador lee la protección de MARS_TITAN_TRAINING_HOLD. Una que bloquea da el
// código de salto que CMake declara en SKIP_RETURN_CODE y una ilegible detiene la prueba como
// fallo. En ningún caso se lanza la orden.
void the_launcher_reads_the_hold(const std::filesystem::path& self, const std::filesystem::path& gate,
                                 const std::filesystem::path& blocking, int declared_skip,
                                 const std::filesystem::path& root) {
    const auto launch = [&](const std::filesystem::path& hold, const Sentinel& sentinel) {
        return in_child([&] {
            ::setenv("MARS_TITAN_TRAINING_HOLD", hold.c_str(), 1);
            std::vector<std::string> words{gate.string(), self.string(), "--sentinel",
                                           sentinel.marker.string()};
            std::vector<char*> arguments;
            for (auto& word : words) {
                arguments.push_back(word.data());
            }
            arguments.push_back(nullptr);
            ::execv(arguments.front(), arguments.data());
            return 126;
        });
    };
    require(learning_gate_skip == declared_skip,
            "El código de salto del lanzador no coincide con el de OptimizerStepTests.cmake");
    const Sentinel blocked(self, root / "gate-blocked.marker");
    require(launch(blocking, blocked) == declared_skip && !blocked.ran(),
            "El lanzador no saltó la prueba con la protección que bloquea");
    const auto broken = root / "broken-hold.json";
    std::ofstream(broken) << "{\"training_allowed\": \"no\"}\n";
    const Sentinel unreadable(self, root / "gate-broken.marker");
    require(launch(broken, unreadable) == 1 && !unreadable.ran(),
            "El lanzador no se detuvo ante una protección sin booleano");
}
} // namespace

int main(int argc, char** argv) {
    const std::vector<std::string_view> arguments(argv, argv + argc);
    if (arguments.size() == 3 && arguments[1] == "--sentinel") {
        std::ofstream(std::string(arguments[2])) << "ejecutada\n";
        return sentinel_code;
    }
    try {
        require(arguments.size() == 4,
                "Uso: learning_gate_tests <lanzador> <protección que bloquea> <código de salto>");
        const auto self = std::filesystem::canonical("/proc/self/exe");
        const auto root = std::filesystem::temp_directory_path() /
                          ("mars-titan-learning-gate-" + std::to_string(::getpid()));
        std::filesystem::create_directories(root);
        a_blocking_hold_skips_without_launching(self, root);
        an_allowing_hold_yields_to_the_test(self, root);
        the_launcher_reads_the_hold(self, arguments[1], arguments[2],
                                    std::stoi(std::string(arguments[3])), root);
        std::filesystem::remove_all(root);
        std::cout << "Lanzador de las pruebas con pasos de optimizador comprobado sin lanzarlas\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
