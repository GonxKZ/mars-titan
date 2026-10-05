#include "mars_titan/memory_stress.hpp"

#include <cstddef>
#include <cstdint>
#include <exception>
#include <string_view>

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    // El lector interpreta los bytes como caracteres, sin modificar su representación.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
    const std::string_view input(reinterpret_cast<const char*>(data), size);
    try {
        const auto config = mars_titan::stress::read_config(input);
        mars_titan::stress::Experiment experiment(config);
        experiment.restore(input);
    } catch (const std::exception&) {
        // Un archivo inválido debe rechazarse sin alterar un estado confirmado.
    }
    return 0;
}
