#include "mars_titan/episodic_projection.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <string_view>
#include <vector>

// Tiempo de cada versión del núcleo con las formas del codec episódico: 127 salidas y los seis
// bloques de la vista v3.1 de la campaña (precios 6 × 64, noticias, gráficos, fundamentales,
// macro y presencia). Las versiones se alternan en cada repetición para repartir la deriva de
// la máquina y la salida es una línea JSON por forma y versión. Las filas cubren los dos
// recorridos del núcleo, directo por debajo de projection_packed_rows y por paneles desde ahí.
// El reparto entre hilos lo hace el enlace con el pool de ATen y se mide desde Python.
namespace {
using mars_titan::episodic::available_projection_isas;
using mars_titan::episodic::project_block;
using mars_titan::episodic::projection_isa_name;
using mars_titan::episodic::ProjectionIsa;
using Clock = std::chrono::steady_clock;

constexpr std::array<std::size_t, 6> widths{384, 384, 512, 78, 420, 5};
constexpr std::size_t outputs = 127;
constexpr std::array<std::size_t, 8> row_counts{1, 16, 31, 32, 64, 128, 278, 512};
constexpr std::size_t repetitions = 41;
constexpr std::size_t decile = 10;
constexpr std::size_t target_work = 4096; // Filas por medida, para que las pequeñas no queden
                                          // por debajo de la resolución del reloj.
constexpr std::uint64_t seed = 520;

struct Blocks {
    std::vector<std::vector<double>> inputs;
    std::vector<std::vector<double>> weights;
    std::vector<std::vector<double>> out;
};

Blocks blocks(std::size_t rows, std::mt19937_64& generator) {
    std::normal_distribution<double> normal;
    Blocks result;
    for (const std::size_t width : widths) {
        auto& input = result.inputs.emplace_back(rows * width);
        auto& weight = result.weights.emplace_back(width * outputs);
        std::ranges::generate(input, [&] { return normal(generator); });
        std::ranges::generate(weight, [&] { return normal(generator); });
        result.out.emplace_back(rows * outputs);
    }
    return result;
}

// Microsegundos por llamada de los seis bloques, como una llamada de encode.
double measure(Blocks& data, std::size_t rows, ProjectionIsa isa, std::size_t calls) {
    const auto began = Clock::now();
    for (std::size_t call = 0; call < calls; ++call) {
        for (std::size_t block = 0; block < widths.size(); ++block) {
            project_block(data.inputs[block], rows, widths.at(block), data.weights[block], outputs,
                          data.out[block], isa);
        }
    }
    const std::chrono::duration<double, std::micro> elapsed = Clock::now() - began;
    return elapsed.count() / static_cast<double>(calls);
}

void report(std::size_t rows, ProjectionIsa isa, std::vector<double> times) {
    std::ranges::sort(times);
    std::cout << R"({"rows": )" << rows << R"(, "isa": ")" << projection_isa_name(isa)
              << R"(", "repetitions": )" << times.size() << R"(, "median_us": )"
              << times[times.size() / 2] << R"(, "p10_us": )" << times[times.size() / decile]
              << R"(, "p90_us": )" << times[times.size() - 1 - times.size() / decile] << "}\n";
}
} // namespace

int main() {
    try {
        std::mt19937_64 generator(seed);
        const auto isas = available_projection_isas();
        for (const std::size_t rows : row_counts) {
            auto data = blocks(rows, generator);
            const std::size_t calls = std::max<std::size_t>(1, target_work / rows);
            std::vector<std::vector<double>> times(isas.size());
            for (const ProjectionIsa isa : isas) {
                measure(data, rows, isa, 1); // Calentamiento.
            }
            for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
                for (std::size_t position = 0; position < isas.size(); ++position) {
                    const std::size_t index =
                        repetition % 2 == 0 ? position : isas.size() - 1 - position;
                    times[index].push_back(measure(data, rows, isas[index], calls));
                }
            }
            for (std::size_t index = 0; index < isas.size(); ++index) {
                report(rows, isas[index], times[index]);
            }
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
