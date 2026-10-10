#include "mars_titan/episodic_projection.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>

#if defined(__GNUC__) || defined(__clang__)
#define MARS_TITAN_PROJECTION_INLINE [[gnu::always_inline]] inline
#else
#define MARS_TITAN_PROJECTION_INLINE inline
#endif

namespace mars_titan::episodic {
namespace {
// einsum procesa dos dobles por registro y desenrolla cuatro registros: ocho entradas por bloque.
constexpr std::size_t pair = 2;
constexpr std::size_t chunk = 8;
// Filas por mosaico en el tramo principal. Los restos usan mosaicos de dos filas y de una, con
// más vectores de salida para conservar ocho cadenas de sumas independientes.
constexpr std::size_t band_rows = 4;
constexpr std::size_t pair_vectors = 2;
constexpr std::size_t single_vectors = 4;

#if defined(__GNUC__) || defined(__clang__)
// Ocho salidas en un vector del compilador. Cada carril repite las operaciones de una salida,
// así que el ancho de la instrucción no cambia el resultado.
using Lanes = double __attribute__((vector_size(projection_lanes * sizeof(double))));
#else
struct Lanes {
    std::array<double, projection_lanes> value{};

    friend Lanes operator*(double scale, const Lanes& lanes) {
        Lanes result;
        for (std::size_t lane = 0; lane < projection_lanes; ++lane) {
            result.value[lane] = scale * lanes.value[lane];
        }
        return result;
    }

    friend Lanes operator+(const Lanes& left, const Lanes& right) {
        Lanes result;
        for (std::size_t lane = 0; lane < projection_lanes; ++lane) {
            result.value[lane] = left.value[lane] + right.value[lane];
        }
        return result;
    }
};
#endif

// Formas de una llamada, ya comprobadas por project_block.
struct Problem {
    const double* block;
    std::size_t rows;
    std::size_t width;
    const double* transposed;
    std::size_t outputs;
    double* out;
};

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
// NOLINTBEGIN(cppcoreguidelines-pro-bounds-constant-array-index)
// Los índices del núcleo salen de las formas comprobadas y de los parámetros de las plantillas.
// El acceso directo evita una comprobación de límites en cada carga del bucle interno.

// Carga `kept` salidas consecutivas. Las que faltan en el último vector de una fila valen cero
// y su resultado no se escribe.
MARS_TITAN_PROJECTION_INLINE void load(Lanes& target, const double* source, std::size_t kept) {
    if (kept == projection_lanes) {
        std::memcpy(&target, source, sizeof(Lanes));
        return;
    }
    std::array<double, projection_lanes> values{};
    std::memcpy(values.data(), source, kept * sizeof(double));
    std::memcpy(&target, values.data(), sizeof(Lanes));
}

// Un mosaico de Rows filas y Vectors vectores de salida. lane0 suma las entradas pares de cada
// par y lane1 las impares, como los dos carriles del registro de einsum.
template <std::size_t Rows, std::size_t Vectors>
struct Tile {
    const double* block{};   // Primera fila del mosaico.
    std::size_t width{};     // Entradas por fila.
    const double* columns{}; // Primera salida del mosaico en la fila 0 de la transpuesta.
    std::size_t outputs{};   // Paso de la transpuesta y de la salida.
    std::size_t kept{};      // Salidas válidas del último vector.
    std::array<std::array<Lanes, Vectors>, Rows> lane0{};
    std::array<std::array<Lanes, Vectors>, Rows> lane1{};

    // Un par de entradas (index, index + 1). Si index + 1 no existe, einsum suma 0 · 0 al
    // carril impar, que no cambia su valor (ver la cabecera), así que ese carril no se toca.
    MARS_TITAN_PROJECTION_INLINE void step(std::size_t index, bool complete) {
        std::array<Lanes, Vectors> even{};
        std::array<Lanes, Vectors> odd{};
        for (std::size_t vector = 0; vector < Vectors; ++vector) {
            const std::size_t count = vector + 1 == Vectors ? kept : projection_lanes;
            const double* first = columns + index * outputs + vector * projection_lanes;
            load(even[vector], first, count);
            if (complete) {
                load(odd[vector], first + outputs, count);
            }
        }
        for (std::size_t row = 0; row < Rows; ++row) {
            const double first = block[row * width + index];
            for (std::size_t vector = 0; vector < Vectors; ++vector) {
                lane0[row][vector] = first * even[vector] + lane0[row][vector];
            }
            if (complete) {
                const double second = block[row * width + index + 1];
                for (std::size_t vector = 0; vector < Vectors; ++vector) {
                    lane1[row][vector] = second * odd[vector] + lane1[row][vector];
                }
            }
        }
    }

    // Recorre las entradas con el orden de einsum y escribe el mosaico desde `out`.
    MARS_TITAN_PROJECTION_INLINE void run(double* out) {
        std::size_t index = 0;
        for (; width - index >= chunk; index += chunk) {
            for (std::size_t offset = chunk; offset > 0; offset -= pair) {
                step(index + offset - pair, true);
            }
        }
        for (; index < width; index += pair) {
            step(index, index + 1 < width);
        }
        for (std::size_t row = 0; row < Rows; ++row) {
            for (std::size_t vector = 0; vector < Vectors; ++vector) {
                const Lanes total = lane0[row][vector] + lane1[row][vector];
                std::array<double, projection_lanes> values{};
                std::memcpy(values.data(), &total, sizeof(Lanes));
                const std::size_t count = vector + 1 == Vectors ? kept : projection_lanes;
                std::memcpy(out + row * outputs + vector * projection_lanes, values.data(),
                            count * sizeof(double));
            }
        }
    }
};

// Mosaicos completos de Vectors vectores y, al final de la fila, vectores sueltos con el
// último incompleto.
template <std::size_t Rows, std::size_t Vectors>
MARS_TITAN_PROJECTION_INLINE void project_rows(const Problem& problem, std::size_t row) {
    const double* block = problem.block + row * problem.width;
    double* out = problem.out + row * problem.outputs;
    constexpr std::size_t covered = Vectors * projection_lanes;
    std::size_t column = 0;
    for (; problem.outputs - column >= covered; column += covered) {
        Tile<Rows, Vectors> tile{block, problem.width, problem.transposed + column,
                                 problem.outputs, projection_lanes};
        tile.run(out + column);
    }
    for (; column < problem.outputs; column += projection_lanes) {
        Tile<Rows, 1> tile{block, problem.width, problem.transposed + column, problem.outputs,
                           std::min(problem.outputs - column, projection_lanes)};
        tile.run(out + column);
    }
}
// NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

// El mismo cálculo se compila para AVX-512, AVX2 y la línea base, y el cargador elige el
// primero que admite el procesador. Fuera de x86-64 con ELF se compila una sola versión.
#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__) && defined(__ELF__)
__attribute__((target_clones("avx512f", "avx2", "default")))
#endif
void project_all(const Problem& problem) {
    std::size_t row = 0;
    for (; problem.rows - row >= band_rows; row += band_rows) {
        project_rows<band_rows, 1>(problem, row);
    }
    for (; problem.rows - row >= pair; row += pair) {
        project_rows<pair, pair_vectors>(problem, row);
    }
    for (; row < problem.rows; ++row) {
        project_rows<1, single_vectors>(problem, row);
    }
}

bool shape_matches(std::size_t first, std::size_t second, std::size_t size) {
    return first <= std::numeric_limits<std::size_t>::max() / second && first * second == size;
}
} // namespace

void project_block(std::span<const double> block, std::size_t rows, std::size_t width,
                   std::span<const double> transposed, std::size_t outputs,
                   std::span<double> out) {
    if (width == 0 || outputs == 0) {
        throw std::invalid_argument("La proyección del codec necesita anchura y salidas");
    }
    if (!shape_matches(rows, width, block.size()) ||
        !shape_matches(width, outputs, transposed.size()) ||
        !shape_matches(rows, outputs, out.size())) {
        throw std::invalid_argument("Las formas de la proyección del codec no coinciden");
    }
    if (rows > 0) {
        project_all(Problem{block.data(), rows, width, transposed.data(), outputs, out.data()});
    }
}

} // namespace mars_titan::episodic
