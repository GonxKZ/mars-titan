#include "mars_titan/learning_hold.hpp"

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

#include <unistd.h>

namespace {
using mars_titan::learning::learning_blocked;
using mars_titan::learning::learning_hold_path;
using mars_titan::learning::require_learning_allowed;
namespace fs = std::filesystem;

void require(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <class Function> std::string failure(Function&& action) {
    try {
        std::forward<Function>(action)();
    } catch (const std::exception& error) {
        return error.what();
    }
    return {};
}

template <class Function> bool throws(Function&& action) {
    return !failure(std::forward<Function>(action)).empty();
}

fs::path write(const fs::path& path, std::string_view content) {
    fs::create_directories(path.parent_path());
    std::ofstream(path, std::ios::binary | std::ios::trunc) << content;
    return path;
}

void set_variable(const char* name, const fs::path& value) {
    require(::setenv(name, value.c_str(), 1) == 0, "No se pudo fijar la variable de prueba");
}

void absent_allowed_and_blocked(const fs::path& root) {
    require(!learning_blocked(root / "absent.json"), "Una protección ausente no debe bloquear");
    require(!learning_blocked(write(root / "allowed.json", R"({"training_allowed": true})")),
            "Una protección permitida no debe bloquear");
    require(learning_blocked(write(root / "blocked.json", R"({"training_allowed": false})")),
            "Una protección con training_allowed falso debe bloquear");
}

void ambiguous_holds_fail_fast(const fs::path& root) {
    int index = 0;
    for (const std::string_view content :
         {R"({"training_allowed": null})", R"({"training_allowed": "false"})",
          R"({"training_allowed": 0})", R"({})", R"([false])"}) {
        const auto path = write(root / ("ambiguous-" + std::to_string(index++) + ".json"), content);
        require(failure([&] { static_cast<void>(learning_blocked(path)); })
                        .find("no declara training_allowed como booleano") != std::string::npos,
                "Una protección ambigua debe detener la ejecución con su motivo");
    }
    const auto invalid = write(root / "invalid.json", R"({"training_allowed": )");
    require(throws([&] { static_cast<void>(learning_blocked(invalid)); }),
            "Un documento inválido debe detener la ejecución");
    fs::create_directories(root / "directory.json");
    require(throws([&] { static_cast<void>(learning_blocked(root / "directory.json")); }),
            "Una protección que no es un archivo regular debe detener la ejecución");
}

void variable_and_home_select_the_same_hold(const fs::path& root) {
    set_variable("MARS_TITAN_TRAINING_HOLD", root / "blocked.json");
    require(learning_hold_path() == root / "blocked.json", "La variable no selecciona la ruta");
    try {
        require_learning_allowed("la prueba nativa");
        throw std::logic_error("El bloqueo no detuvo la acción");
    } catch (const std::runtime_error& error) {
        const std::string message = error.what();
        require(message.find("Bloqueo de aprendizaje vigente: la prueba nativa") == 0,
                "El mensaje no identifica el bloqueo y la acción");
    }
    set_variable("MARS_TITAN_TRAINING_HOLD", root / "allowed.json");
    require_learning_allowed("la prueba nativa");
    set_variable("MARS_TITAN_TRAINING_HOLD", fs::path{});
    require(throws([] { static_cast<void>(learning_hold_path()); }),
            "Una variable vacía no debe equivaler a una protección ausente");

    require(::unsetenv("MARS_TITAN_TRAINING_HOLD") == 0, "No se pudo retirar la variable");
    set_variable("HOME", root / "home");
    const auto standard = root / "home/.local/state/mars-titan/training-hold-2000.json";
    require(learning_hold_path() == standard, "La ruta por defecto no coincide con Python");
    require_learning_allowed("la prueba nativa");
    write(standard, R"({"training_allowed": false})");
    require(throws([] { require_learning_allowed("la prueba nativa"); }),
            "La protección por defecto debe bloquear");
}
} // namespace

int main() {
    const auto root =
        fs::temp_directory_path() / ("mars-titan-learning-hold-" + std::to_string(::getpid()));
    try {
        fs::remove_all(root);
        fs::create_directories(root);
        absent_allowed_and_blocked(root);
        ambiguous_holds_fail_fast(root);
        variable_and_home_select_the_same_hold(root);
        fs::remove_all(root);
        std::cout << "Protección nativa del aprendizaje verificada\n";
        return 0;
    } catch (const std::exception& error) {
        fs::remove_all(root);
        std::cerr << "Error: " << error.what() << '\n';
        return 1;
    }
}
