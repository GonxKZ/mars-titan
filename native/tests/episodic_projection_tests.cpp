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
#include <thread>
#include <utility>
#include <vector>

namespace {
using mars_titan::episodic::available_projection_isas;
using mars_titan::episodic::default_projection_isa;
using mars_titan::episodic::project_block;
using mars_titan::episodic::project_block_groups;
using mars_titan::episodic::projection_groups;
using mars_titan::episodic::projection_packed_rows;
using mars_titan::episodic::projection_parallel_work;
using mars_titan::episodic::projection_threads;
using mars_titan::episodic::projection_isa_name;
using mars_titan::episodic::projection_lanes;
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

// La misma proyección por tramos de grupos: [0, g/3), [g/3, g/3) vacío y [g/3, g).
std::vector<double> projected_by_groups(const std::vector<double>& block,
                                        const std::vector<double>& transposed, Shape shape,
                                        ProjectionIsa isa) {
    std::vector<double> out(shape.rows * shape.outputs, std::numeric_limits<double>::quiet_NaN());
    const std::size_t groups = projection_groups(shape.outputs);
    for (const auto [first, last] : {std::pair{std::size_t{0}, groups / 3},
                                     std::pair{groups / 3, groups / 3},
                                     std::pair{groups / 3, groups}}) {
        project_block_groups(block, shape.rows, shape.width, transposed, shape.outputs, out, isa,
                             first, last);
    }
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
    // Por debajo de projection_packed_rows, el recorrido directo con sus tres mosaicos. Desde
    // ahí, los paneles con mosaicos de cuatro, dos y una fila.
    constexpr std::size_t packed = projection_packed_rows;
    const std::vector<std::size_t> row_counts{
        0, 1, 2, 3, 4, 5, 6, 7, 9, 13, packed - 1, packed, packed + 1, packed + 2, packed + 3};
    const std::vector<std::size_t> widths{1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 78, 320, 513};
    const std::vector<std::size_t> output_counts{1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 127};
    for (const bool extreme : {false, true}) {
        for (const std::size_t rows : row_counts) {
            for (const std::size_t width : widths) {
                for (const std::size_t outputs : output_counts) {
                    const Shape shape{rows, width, outputs};
                    const auto block = values(generator, rows * width, extreme);
                    const auto transposed = values(generator, width * outputs, extreme);
                    const auto expected = reference(block, transposed, shape);
                    require(same_bits(projected(block, transposed, shape, isa), expected),
                            failure("La proyección nativa no conserva el orden declarado de "
                                    "las sumas",
                                    isa));
                    // Los paneles por tramos, también con menos filas que el corte.
                    require(same_bits(projected_by_groups(block, transposed, shape, isa),
                                      expected),
                            failure("Los tramos de grupos no conservan el orden declarado", isa));
                }
            }
        }
    }
}

// Una fila sola usa el recorrido directo. Dentro de un bloque usa el directo o los paneles,
// según sus filas, y debe dar los mismos bits en los dos casos.
void every_row_is_independent_of_its_neighbours(ProjectionIsa isa) {
    std::mt19937_64 generator(row_seed);
    for (const std::size_t count : {independent_rows, projection_packed_rows + independent_rows}) {
        const Shape shape{count, independent_width, codec_outputs};
        const auto transposed = values(generator, shape.width * shape.outputs, false);
        const auto block = values(generator, shape.rows * shape.width, true);
        const auto together = projected(block, transposed, shape, isa);
        const std::span<const double> rows(block);
        const std::span<const double> results(together);
        for (std::size_t row = 0; row < shape.rows; ++row) {
            const auto single = rows.subspan(row * shape.width, shape.width);
            const auto alone =
                projected(std::vector<double>(single.begin(), single.end()), transposed,
                          Shape{1, shape.width, shape.outputs}, isa);
            require(same_bits(alone, results.subspan(row * shape.outputs, shape.outputs)),
                    failure("Una fila depende del mosaico en que se calcula", isa));
        }
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
            project_block_groups(block, 2, 3, transposed, 2, out, default_projection_isa(), 1, 0);
        },
        "Un tramo de grupos invertido debe rechazarse");
    rejected(
        [&] {
            project_block_groups(block, 2, 3, transposed, 2, out, default_projection_isa(), 0, 2);
        },
        "Un tramo con grupos que no existen debe rechazarse");
    rejected(
        [&] {
            project_block_groups(block, 2, 3, transposed, 2, std::span<double>(out).first(3),
                                 default_projection_isa(), 0, 1);
        },
        "Un tramo con formas incoherentes debe rechazarse");
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

// Con las formas del codec, tramos disjuntos de grupos calculados a la vez desde varios hilos,
// sobre la misma salida, dan los bits de la proyección entera. Un tramo por grupo es el caso
// con más hilos escribiendo en las mismas filas.
void concurrent_groups_match_the_whole_block(ProjectionIsa isa) {
    std::mt19937_64 generator(order_seed + 1);
    for (const std::size_t rows : {projection_packed_rows - 1, projection_packed_rows + 5}) {
        for (const std::size_t width : {5UL, 78UL, 384UL, 512UL}) {
            const Shape shape{rows, width, codec_outputs};
            const auto block = values(generator, shape.rows * shape.width, true);
            const auto transposed = values(generator, shape.width * shape.outputs, true);
            const auto whole = projected(block, transposed, shape, isa);
            require(same_bits(whole, reference(block, transposed, shape)),
                    failure("La proyección entera no conserva el orden declarado", isa));
            const std::size_t groups = projection_groups(shape.outputs);
            for (const std::size_t parts : {2UL, 3UL, groups}) {
                std::vector<double> out(shape.rows * shape.outputs,
                                        std::numeric_limits<double>::quiet_NaN());
                {
                    std::vector<std::jthread> threads;
                    for (std::size_t part = 0; part < parts; ++part) {
                        threads.emplace_back([&, part] {
                            project_block_groups(block, shape.rows, shape.width, transposed,
                                                 shape.outputs, out, isa, groups * part / parts,
                                                 groups * (part + 1) / parts);
                        });
                    }
                }
                require(same_bits(out, whole),
                        failure("Los tramos calculados a la vez cambian la proyección", isa));
            }
        }
    }
}

// Un tramo escribe solo las columnas de sus grupos, con los dos recorridos: el primero, uno
// intermedio y el último, que tiene siete salidas.
void groups_write_only_their_columns(ProjectionIsa isa) {
    std::mt19937_64 generator(order_seed + 2);
    constexpr double untouched = 12345.0;
    for (const std::size_t rows : {projection_packed_rows - 1, projection_packed_rows + 5}) {
        const Shape shape{rows, independent_width, codec_outputs};
        const auto block = values(generator, shape.rows * shape.width, false);
        const auto transposed = values(generator, shape.width * shape.outputs, false);
        const auto whole = projected(block, transposed, shape, isa);
        const std::size_t groups = projection_groups(shape.outputs);
        for (const std::size_t group : {std::size_t{0}, groups / 2, groups - 1}) {
            std::vector<double> out(shape.rows * shape.outputs, untouched);
            project_block_groups(block, shape.rows, shape.width, transposed, shape.outputs, out,
                                 isa, group, group + 1);
            const std::size_t begin = group * projection_lanes;
            const std::size_t end = std::min(begin + projection_lanes, shape.outputs);
            for (std::size_t row = 0; row < shape.rows; ++row) {
                for (std::size_t column = 0; column < shape.outputs; ++column) {
                    const std::size_t index = row * shape.outputs + column;
                    const bool inside = column >= begin && column < end;
                    require(inside ? same_bits(std::span(out).subspan(index, 1),
                                               std::span(whole).subspan(index, 1))
                                   : out[index] == untouched,
                            failure("Un tramo escribe fuera de sus columnas", isa));
                }
            }
        }
    }
}

// El enlace reparte una llamada solo si llega al trabajo mínimo, con pocas o muchas filas.
void threads_need_enough_work() {
    constexpr std::size_t available = 4;
    for (const std::size_t rows : {1UL, 3UL, projection_packed_rows, 1000UL}) {
        const std::size_t per_row = (projection_parallel_work + rows - 1) / rows;
        require(projection_threads(rows, per_row, 1, available) == available,
                "Con el trabajo mínimo se usan los hilos disponibles");
        require(projection_threads(rows, per_row - 1, 1, available) == 1,
                "Por debajo del trabajo mínimo no se reparte la proyección");
        require(projection_threads(rows, per_row, 1, 0) == 1,
                "Sin hilos disponibles se usa el que llama");
    }
    require(projection_threads(0, projection_parallel_work, 1, available) == 1,
            "Un bloque sin filas no se reparte");
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
            concurrent_groups_match_the_whole_block(isa);
            groups_write_only_their_columns(isa);
            std::cout << "Versión " << projection_isa_name(isa) << " comprobada\n";
        }
        inconsistent_shapes_are_rejected_before_writing();
        threads_need_enough_work();
        default_dispatch_uses_the_widest_available_version();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Proyección del codec con el orden de einsum comprobada\n";
    return 0;
}
