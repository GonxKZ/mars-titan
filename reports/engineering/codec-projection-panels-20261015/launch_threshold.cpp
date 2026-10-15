// Un hilo frente a dos con el pool inter-op de ATen, como reparte el enlace de Python: quien
// llama calcula la primera mitad de los grupos con project_block_groups y una tarea de
// at::launch la segunda. Sirve para fijar projection_parallel_work. En modo `warm` las llamadas
// van seguidas, como los seis bloques de un encode. En modo `cold` cada llamada espera antes
// 3 ms, como la primera proyección de un encode. Las dos variantes se alternan y se comprueban
// los bits.
//
// Uso: launch_threshold REPETICIONES warm|cold. Escribe una línea JSON por forma.
#include "mars_titan/episodic_projection.hpp"

#include <ATen/Parallel.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string_view>
#include <thread>
#include <vector>

namespace {
namespace episodic = mars_titan::episodic;
constexpr std::size_t outputs = 127;
constexpr std::size_t warm_target = 4096 * 384; // Filas · anchura de cada medida en caliente.
constexpr std::size_t cold_calls = 20;
constexpr auto idle = std::chrono::milliseconds(3);

void launched(const std::vector<double>& block, std::size_t rows, std::size_t width,
              const std::vector<double>& transposed, std::vector<double>& out) {
    const auto isa = episodic::default_projection_isa();
    const std::size_t groups = episodic::projection_groups(outputs);
    const std::size_t half = (groups + 1) / 2;
    std::atomic<int> done{0};
    at::launch([&] {
        episodic::project_block_groups(block, rows, width, transposed, outputs, out, isa, half,
                                       groups);
        done.store(1);
        done.notify_one();
    });
    episodic::project_block_groups(block, rows, width, transposed, outputs, out, isa, 0, half);
    // Aquí la variable vive hasta el final de main, así que basta con esperar a que cambie.
    done.wait(0);
}

double median(std::vector<double> values) {
    std::ranges::sort(values);
    return values[values.size() / 2];
}
} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "Uso: launch_threshold REPETICIONES warm|cold\n");
        return 2;
    }
    const int repetitions = std::atoi(argv[1]);
    const bool cold = std::string_view(argv[2]) == "cold";
    std::mt19937_64 generator(628);
    std::normal_distribution<double> normal;
    for (const std::size_t rows : {1UL, 8UL, 16UL, 31UL, 32UL, 64UL, 128UL, 278UL, 512UL,
                                   1000UL}) {
        for (const std::size_t width : {5UL, 78UL, 384UL, 512UL}) {
            std::vector<double> block(rows * width);
            std::vector<double> transposed(width * outputs);
            std::ranges::generate(block, [&] { return normal(generator); });
            std::ranges::generate(transposed, [&] { return normal(generator); });
            std::vector<double> single(rows * outputs);
            std::vector<double> two(rows * outputs);
            const std::size_t calls =
                cold ? cold_calls : std::max<std::size_t>(4, warm_target / (rows * width));
            const auto run = [&](bool parallel) {
                double total = 0.0;
                for (std::size_t call = 0; call < calls; ++call) {
                    if (cold) {
                        std::this_thread::sleep_for(idle);
                    }
                    const auto began = std::chrono::steady_clock::now();
                    if (parallel) {
                        launched(block, rows, width, transposed, two);
                    } else {
                        episodic::project_block(block, rows, width, transposed, outputs, single);
                    }
                    const std::chrono::duration<double, std::micro> elapsed =
                        std::chrono::steady_clock::now() - began;
                    total += elapsed.count();
                }
                return total / static_cast<double>(calls);
            };
            run(false); // Calentamiento.
            run(true);
            std::array<std::vector<double>, 2> times;
            for (int repetition = 0; repetition < repetitions; ++repetition) {
                const bool launch_first = repetition % 2 == 1;
                const double first = run(launch_first);
                const double second = run(!launch_first);
                times[0].push_back(launch_first ? second : first);
                times[1].push_back(launch_first ? first : second);
            }
            const bool same =
                std::memcmp(single.data(), two.data(), single.size() * sizeof(double)) == 0;
            std::printf("{\"mode\": \"%s\", \"rows\": %zu, \"width\": %zu, \"work\": %zu, "
                        "\"repetitions\": %d, \"one_thread_us\": %.1f, \"two_threads_us\": %.1f, "
                        "\"ratio\": %.3f, \"same_bits\": %s}\n",
                        cold ? "cold" : "warm", rows, width, rows * width * outputs, repetitions,
                        median(times[0]), median(times[1]), median(times[0]) / median(times[1]),
                        same ? "true" : "false");
            std::fflush(stdout);
        }
    }
    return 0;
}
