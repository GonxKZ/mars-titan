#include "mars_titan/episodic_projection.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using mars_titan::episodic::available_projection_isas;
using mars_titan::episodic::default_projection_isa;
using mars_titan::episodic::project_block;
using mars_titan::episodic::projection_isa_name;
using mars_titan::episodic::ProjectionIsa;

// Orden de einsum: bloques de ocho entradas recorridos del par 6-7 al par 0-1.
constexpr std::size_t chunk = 8;
constexpr std::size_t last_pair = 6;
constexpr std::size_t pair = 2;
constexpr int exponent_range = 300;
constexpr int value_kinds = 9;
constexpr double subnormal_multiple = 3.0;
constexpr std::uint64_t order_seed = 20261010;
constexpr std::uint64_t row_seed = 7;
constexpr std::size_t independent_rows = 11;
constexpr std::size_t independent_width = 37;
constexpr std::size_t codec_outputs = 127;

struct Shape {
    std::size_t rows;
    std::size_t width;
    std::size_t outputs;
};

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

template <class Function> void rejected(Function&& function, std::string_view message) {
    try {
        std::forward<Function>(function)();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(std::string(message));
}

// Referencia escalar escrita a partir del orden de einsum, independiente del núcleo. Incluye
// las dos sumas de cero que el núcleo omite: 0 · 0 en el carril impar y 0 + total.
std::vector<double> reference(const std::vector<double>& block,
                              const std::vector<double>& transposed, Shape shape) {
    std::vector<double> out(shape.rows * shape.outputs);
    for (std::size_t row = 0; row < shape.rows; ++row) {
        for (std::size_t output = 0; output < shape.outputs; ++output) {
            const auto term = [&](std::size_t index) {
                return block[row * shape.width + index] *
                       transposed[index * shape.outputs + output];
            };
            double even = 0.0;
            double odd = 0.0;
            std::size_t index = 0;
            for (; shape.width - index >= chunk; index += chunk) {
                for (std::size_t offset = last_pair;; offset -= pair) {
                    even = term(index + offset) + even;
                    odd = term(index + offset + 1) + odd;
                    if (offset == 0) {
                        break;
                    }
                }
            }
            for (; index < shape.width; index += pair) {
                even = term(index) + even;
                odd = (index + 1 < shape.width ? term(index + 1) : 0.0 * 0.0) + odd;
            }
            out[row * shape.outputs + output] = 0.0 + (even + odd);
        }
    }
    return out;
}

bool same_bits(std::span<const double> left, std::span<const double> right) {
    if (left.size() != right.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (std::bit_cast<std::uint64_t>(left[index]) !=
            std::bit_cast<std::uint64_t>(right[index])) {
            return false;
        }
    }
    return true;
}

std::vector<double> projected(const std::vector<double>& block,
                              const std::vector<double>& transposed, Shape shape,
                              ProjectionIsa isa) {
    std::vector<double> out(shape.rows * shape.outputs, std::numeric_limits<double>::quiet_NaN());
    project_block(block, shape.rows, shape.width, transposed, shape.outputs, out, isa);
    return out;
}

std::string failure(std::string_view message, ProjectionIsa isa) {
    return std::string(message) + " (" + std::string(projection_isa_name(isa)) + ")";
}

// Valores con signos, escalas muy distintas, ceros con signo y subnormales, para que un
// cambio en el orden de las sumas cambie el redondeo.
std::vector<double> values(std::mt19937_64& generator, std::size_t count, bool extreme) {
    std::normal_distribution<double> normal;
    std::uniform_int_distribution<int> exponent(-exponent_range, exponent_range);
    std::uniform_int_distribution<int> kind(0, value_kinds);
    std::vector<double> result(count);
    for (double& value : result) {
        value = normal(generator);
        if (extreme) {
            value = std::ldexp(value, exponent(generator));
            const int selected = kind(generator);
            if (selected == 0) {
                value = 0.0;
            } else if (selected == 1) {
                value = -0.0;
            } else if (selected == 2) {
                value = std::numeric_limits<double>::denorm_min() * subnormal_multiple;
            }
        }
    }
    return result;
}

void matches_the_declared_order_for_every_tile_and_remainder(ProjectionIsa isa) {
    std::mt19937_64 generator(order_seed);
    const std::vector<std::size_t> row_counts{0, 1, 2, 3, 4, 5, 6, 7, 9, 13};
    const std::vector<std::size_t> widths{1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 78, 320, 513};
    const std::vector<std::size_t> output_counts{1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 127};
    for (const bool extreme : {false, true}) {
        for (const std::size_t rows : row_counts) {
            for (const std::size_t width : widths) {
                for (const std::size_t outputs : output_counts) {
                    const Shape shape{rows, width, outputs};
                    const auto block = values(generator, rows * width, extreme);
                    const auto transposed = values(generator, width * outputs, extreme);
                    require(same_bits(projected(block, transposed, shape, isa),
                                      reference(block, transposed, shape)),
                            failure("La proyección nativa no conserva el orden declarado de las "
                                    "sumas",
                                    isa));
                }
            }
        }
    }
}

void every_row_is_independent_of_its_neighbours(ProjectionIsa isa) {
    std::mt19937_64 generator(row_seed);
    const Shape shape{independent_rows, independent_width, codec_outputs};
    const auto transposed = values(generator, shape.width * shape.outputs, false);
    const auto block = values(generator, shape.rows * shape.width, true);
    const auto together = projected(block, transposed, shape, isa);
    const std::span<const double> rows(block);
    const std::span<const double> results(together);
    for (std::size_t row = 0; row < shape.rows; ++row) {
        const auto single = rows.subspan(row * shape.width, shape.width);
        const auto alone = projected(std::vector<double>(single.begin(), single.end()), transposed,
                                     Shape{1, shape.width, shape.outputs}, isa);
        require(same_bits(alone, results.subspan(row * shape.outputs, shape.outputs)),
                failure("Una fila depende del mosaico en que se calcula", isa));
    }
}

void inconsistent_shapes_are_rejected_before_writing() {
    const std::vector<double> block(6, 1.0);
    const std::vector<double> transposed(6, 1.0);
    std::vector<double> out(4, -1.0);
    rejected([&] { project_block(block, 2, 3, transposed, 2, std::span<double>(out).first(3)); },
             "Una salida corta debe rechazarse");
    rejected([&] { project_block(block, 3, 3, transposed, 2, out); },
             "Un bloque con más filas que valores debe rechazarse");
    rejected([&] { project_block(block, 2, 3, transposed, 3, out); },
             "Una proyección con otra forma debe rechazarse");
    rejected([&] { project_block(block, 2, 0, transposed, 2, out); },
             "Una anchura nula debe rechazarse");
    rejected([&] { project_block(block, 2, 3, transposed, 0, out); },
             "Una proyección sin salidas debe rechazarse");
    rejected(
        [&] {
            project_block(block, std::numeric_limits<std::size_t>::max() / 2 + 1, 2, transposed, 2,
                          out);
        },
        "Un producto de dimensiones desbordado debe rechazarse");
    const auto isas = available_projection_isas();
    for (const ProjectionIsa isa :
         {ProjectionIsa::portable, ProjectionIsa::avx2, ProjectionIsa::avx512f}) {
        if (std::find(isas.begin(), isas.end(), isa) == isas.end()) {
            rejected([&] { project_block(block, 2, 3, transposed, 2, out, isa); },
                     "Una versión que el procesador no admite debe rechazarse");
        }
    }
    for (const double value : out) {
        require(value == -1.0, "Un rechazo no puede escribir en la salida");
    }
}

// La versión por defecto es la más ancha disponible y la portable siempre existe.
void default_dispatch_uses_the_widest_available_version() {
    const auto isas = available_projection_isas();
    require(!isas.empty() && isas.front() == ProjectionIsa::portable,
            "La versión portable debe estar siempre disponible");
    require(default_projection_isa() == isas.back(),
            "La versión por defecto debe ser la más ancha disponible");
    std::mt19937_64 generator(row_seed);
    const Shape shape{independent_rows, independent_width, codec_outputs};
    const auto block = values(generator, shape.rows * shape.width, true);
    const auto transposed = values(generator, shape.width * shape.outputs, true);
    std::vector<double> out(shape.rows * shape.outputs);
    project_block(block, shape.rows, shape.width, transposed, shape.outputs, out);
    require(same_bits(out, projected(block, transposed, shape, isas.back())),
            "La versión por defecto debe dar los mismos bits que la más ancha");
}
} // namespace

int main() {
    try {
        for (const ProjectionIsa isa : available_projection_isas()) {
            matches_the_declared_order_for_every_tile_and_remainder(isa);
            every_row_is_independent_of_its_neighbours(isa);
            std::cout << "Versión " << projection_isa_name(isa) << " comprobada\n";
        }
        inconsistent_shapes_are_rejected_before_writing();
        default_dispatch_uses_the_widest_available_version();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Proyección del codec con el orden de einsum comprobada\n";
    return 0;
}
