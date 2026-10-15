// Núcleo de 131d809e (espacio mars_titan_old) y núcleo de #628 alternados en cada repetición,
// con un hilo y los seis bloques del codec v3.1 por llamada, como una llamada de encode.
// Comprueba además que los dos dan los mismos bits. El reparto entre hilos es del enlace y se
// mide con projection_times.py.
//
// Uso: interleaved REPETICIONES FILAS... Escribe una línea JSON por número de filas.
#include "mars_titan/episodic_projection.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <span>
#include <vector>

namespace mars_titan_old::episodic {
void project_block(std::span<const double> block, std::size_t rows, std::size_t width,
                   std::span<const double> transposed, std::size_t outputs, std::span<double> out);
}

namespace {
namespace current = mars_titan::episodic;
constexpr std::array<std::size_t, 6> widths{384, 384, 512, 78, 420, 5};
constexpr std::size_t outputs = 127;
constexpr std::size_t target_rows = 2048; // Filas por medida, para no medir llamadas sueltas.

double median(std::vector<double> values) {
    std::ranges::sort(values);
    return values[values.size() / 2];
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Uso: interleaved REPETICIONES FILAS...\n");
        return 2;
    }
    const int repetitions = std::atoi(argv[1]);
    std::mt19937_64 generator(628);
    std::normal_distribution<double> normal;
    for (int argument = 2; argument < argc; ++argument) {
        const auto rows = static_cast<std::size_t>(std::atoi(argv[argument]));
        std::vector<std::vector<double>> inputs, weights, old_out, new_out;
        for (std::size_t block = 0; block < widths.size(); ++block) {
            const std::size_t width = widths.at(block);
            inputs.emplace_back(rows * width);
            weights.emplace_back(width * outputs);
            old_out.emplace_back(rows * outputs);
            new_out.emplace_back(rows * outputs);
            std::ranges::generate(inputs.back(), [&] { return normal(generator); });
            std::ranges::generate(weights.back(), [&] { return normal(generator); });
        }
        const std::size_t calls = std::max<std::size_t>(1, target_rows / rows);
        const auto run = [&](bool fresh) {
            const auto began = std::chrono::steady_clock::now();
            for (std::size_t call = 0; call < calls; ++call) {
                for (std::size_t block = 0; block < widths.size(); ++block) {
                    if (fresh) {
                        current::project_block(inputs[block], rows, widths.at(block),
                                               weights[block], outputs, new_out[block]);
                    } else {
                        mars_titan_old::episodic::project_block(inputs[block], rows,
                                                                widths.at(block), weights[block],
                                                                outputs, old_out[block]);
                    }
                }
            }
            const std::chrono::duration<double, std::micro> elapsed =
                std::chrono::steady_clock::now() - began;
            return elapsed.count() / static_cast<double>(calls);
        };
        run(false); // Calentamiento de los dos núcleos.
        run(true);
        std::vector<double> old_times, new_times, ratios;
        for (int repetition = 0; repetition < repetitions; ++repetition) {
            // El orden se alterna para repartir la deriva de la máquina entre los dos.
            const bool new_first = repetition % 2 == 1;
            const double first = run(new_first);
            const double second = run(!new_first);
            old_times.push_back(new_first ? second : first);
            new_times.push_back(new_first ? first : second);
            ratios.push_back(old_times.back() / new_times.back());
        }
        bool same = true;
        for (std::size_t block = 0; block < widths.size(); ++block) {
            same = same && std::memcmp(old_out[block].data(), new_out[block].data(),
                                       old_out[block].size() * sizeof(double)) == 0;
        }
        std::ranges::sort(ratios);
        const std::size_t decile = ratios.size() / 10;
        std::printf("{\"rows\": %zu, \"repetitions\": %d, \"old_median_us\": %.1f, "
                    "\"new_median_us\": %.1f, \"ratio_median\": %.3f, \"ratio_p10\": %.3f, "
                    "\"ratio_p90\": %.3f, \"same_bits\": %s}\n",
                    rows, repetitions, median(old_times), median(new_times),
                    ratios[ratios.size() / 2], ratios[decile], ratios[ratios.size() - 1 - decile],
                    same ? "true" : "false");
        std::fflush(stdout);
    }
    return 0;
}
