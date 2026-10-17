#include "mars_titan/episodic_projection.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string_view>
#include <vector>

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
// Filas por mosaico. Con un vector de ocho salidas y dos carriles por fila, cuatro filas dan
// ocho cadenas de sumas independientes. Los restos de los paneles usan mosaicos de dos filas y
// de una. Los del recorrido directo usan más vectores de salida para conservar ocho cadenas.
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

// Formas de una llamada, ya comprobadas por `checked`.
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

// Recorre los pares de entradas de un mosaico con el orden de einsum: bloques de ocho entradas,
// dentro de cada uno del par 6-7 al par 0-1, y el resto por pares en orden creciente. Los dos
// recorridos de abajo lo comparten, así que suman en el mismo orden.
template <typename Tiles>
MARS_TITAN_PROJECTION_INLINE void add_in_einsum_order(Tiles& tile, std::size_t width) {
    std::size_t index = 0;
    for (; width - index >= chunk; index += chunk) {
        for (std::size_t offset = chunk; offset > 0; offset -= pair) {
            tile.step(index + offset - pair, true);
        }
    }
    for (; index < width; index += pair) {
        tile.step(index, index + 1 < width);
    }
}

// Recorrido directo de la transpuesta, para pocas filas. Es el núcleo de #520.

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

// Un mosaico de Rows filas y Vectors vectores de salida leídos de la transpuesta. lane0 suma
// las entradas pares de cada par y lane1 las impares, como los dos carriles de einsum.
template <std::size_t Rows, std::size_t Vectors> struct DirectTile {
    const double* block{};   // Primera fila del mosaico.
    std::size_t width{};     // Entradas por fila.
    const double* columns{}; // Primera salida del mosaico en la fila 0 de la transpuesta.
    std::size_t outputs{};   // Paso de la transpuesta y de la salida.
    std::size_t kept{};      // Salidas válidas del último vector.
    std::array<std::array<Lanes, Vectors>, Rows> lane0{};
    std::array<std::array<Lanes, Vectors>, Rows> lane1{};

    // Un par de entradas (index, index + 1). Si index + 1 no existe, el carril impar no se toca.
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

    // Suma las entradas con el orden de einsum y escribe el mosaico desde `out`.
    MARS_TITAN_PROJECTION_INLINE void run(double* out) {
        add_in_einsum_order(*this, width);
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

// Las salidas [begin, end) de las filas [row, row + Rows): mosaicos completos de Vectors
// vectores y, al final del tramo, vectores sueltos con el último incompleto.
template <std::size_t Rows, std::size_t Vectors>
MARS_TITAN_PROJECTION_INLINE void project_direct_rows(const Problem& problem, std::size_t row,
                                                      std::size_t begin, std::size_t end) {
    const double* block = problem.block + row * problem.width;
    double* out = problem.out + row * problem.outputs;
    constexpr std::size_t covered = Vectors * projection_lanes;
    std::size_t column = begin;
    for (; end - column >= covered; column += covered) {
        DirectTile<Rows, Vectors> tile{block, problem.width, problem.transposed + column,
                                       problem.outputs, projection_lanes};
        tile.run(out + column);
    }
    for (; column < end; column += projection_lanes) {
        DirectTile<Rows, 1> tile{block, problem.width, problem.transposed + column,
                                 problem.outputs, std::min(end - column, projection_lanes)};
        tile.run(out + column);
    }
}

// Los grupos [first, last) con todas las filas, leyendo la transpuesta directamente.
MARS_TITAN_PROJECTION_INLINE void project_direct(const Problem& problem, std::size_t first,
                                                 std::size_t last) {
    const std::size_t begin = first * projection_lanes;
    const std::size_t end = std::min(last * projection_lanes, problem.outputs);
    std::size_t row = 0;
    for (; problem.rows - row >= band_rows; row += band_rows) {
        project_direct_rows<band_rows, 1>(problem, row, begin, end);
    }
    for (; problem.rows - row >= pair; row += pair) {
        project_direct_rows<pair, pair_vectors>(problem, row, begin, end);
    }
    for (; row < problem.rows; ++row) {
        project_direct_rows<1, single_vectors>(problem, row, begin, end);
    }
}

// Recorrido por paneles.

// Panel de un grupo de ocho salidas: para cada entrada, sus ocho pesos contiguos. En la
// transpuesta están separados por `outputs` dobles y casi todas las cargas cruzaban dos líneas
// de caché. Si el último grupo tiene menos de ocho salidas, los carriles que sobran conservan
// lo que tuviera el panel y su resultado no se escribe.
MARS_TITAN_PROJECTION_INLINE void pack(const Problem& problem, std::size_t first,
                                       std::size_t kept, double* panel) {
    for (std::size_t index = 0; index < problem.width; ++index) {
        std::memcpy(panel + index * projection_lanes,
                    problem.transposed + index * problem.outputs + first, kept * sizeof(double));
    }
}

// Un mosaico de Rows filas y un grupo de ocho salidas. lane0 suma las entradas pares de cada
// par y lane1 las impares, como los dos carriles del registro de einsum.
template <std::size_t Rows> struct Tile {
    const double* block{}; // Primera fila del mosaico.
    std::size_t width{};   // Entradas por fila.
    const double* panel{}; // Panel del grupo.
    std::array<Lanes, Rows> lane0{};
    std::array<Lanes, Rows> lane1{};

    // Un par de entradas (index, index + 1). Si index + 1 no existe, einsum suma 0 · 0 al
    // carril impar, que no cambia su valor (ver la cabecera), así que ese carril no se toca.
    MARS_TITAN_PROJECTION_INLINE void step(std::size_t index, bool complete) {
        Lanes even;
        std::memcpy(&even, panel + index * projection_lanes, sizeof(Lanes));
        for (std::size_t row = 0; row < Rows; ++row) {
            lane0[row] = block[row * width + index] * even + lane0[row];
        }
        if (complete) {
            Lanes odd;
            std::memcpy(&odd, panel + (index + 1) * projection_lanes, sizeof(Lanes));
            for (std::size_t row = 0; row < Rows; ++row) {
                lane1[row] = block[row * width + index + 1] * odd + lane1[row];
            }
        }
    }

    // Suma las entradas con el orden de einsum y escribe las `kept` salidas de cada fila.
    MARS_TITAN_PROJECTION_INLINE void run(double* out, std::size_t outputs, std::size_t kept) {
        add_in_einsum_order(*this, width);
        for (std::size_t row = 0; row < Rows; ++row) {
            const Lanes total = lane0[row] + lane1[row];
            std::array<double, projection_lanes> values{};
            std::memcpy(values.data(), &total, sizeof(Lanes));
            std::memcpy(out + row * outputs, values.data(), kept * sizeof(double));
        }
    }
};

// Filas [row, row + Rows) del grupo que empieza en la salida `column`. Es una función y no una
// lambda porque el operador de una lambda no hereda el atributo `target` de la función que la
// contiene y se compilaría para la línea base.
template <std::size_t Rows>
MARS_TITAN_PROJECTION_INLINE void project_tile(const Problem& problem, std::size_t row,
                                               std::size_t column, std::size_t kept,
                                               const double* panel) {
    Tile<Rows> tile{problem.block + row * problem.width, problem.width, panel};
    tile.run(problem.out + row * problem.outputs + column, problem.outputs, kept);
}

// Los grupos [first, last) con todas las filas. Cada grupo se empaqueta en `panel`, de
// width · 8 dobles, y se recorre fila a fila mientras sigue en la caché.
MARS_TITAN_PROJECTION_INLINE void project_groups(const Problem& problem, std::size_t first,
                                                 std::size_t last, double* panel) {
    for (std::size_t group = first; group < last; ++group) {
        const std::size_t column = group * projection_lanes;
        const std::size_t kept = std::min(problem.outputs - column, projection_lanes);
        pack(problem, column, kept, panel);
        std::size_t row = 0;
        for (; problem.rows - row >= band_rows; row += band_rows) {
            project_tile<band_rows>(problem, row, column, kept, panel);
        }
        for (; problem.rows - row >= pair; row += pair) {
            project_tile<pair>(problem, row, column, kept, panel);
        }
        for (; row < problem.rows; ++row) {
            project_tile<1>(problem, row, column, kept, panel);
        }
    }
}
// NOLINTEND(cppcoreguidelines-pro-bounds-constant-array-index)
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

// Los grupos [first, last): con pocas filas por el recorrido directo, sin panel, y si no por
// paneles.
MARS_TITAN_PROJECTION_INLINE void project_work(const Problem& problem, std::size_t first,
                                               std::size_t last, double* panel) {
    if (problem.rows < projection_packed_rows) {
        project_direct(problem, first, last);
    } else {
        project_groups(problem, first, last, panel);
    }
}

// El mismo cálculo se compila para la línea base y, en x86-64, también para AVX2 y AVX-512.
// Se elige en tiempo de ejecución, sin ifunc, porque el resolutor de target_clones se ejecuta
// antes de iniciar ThreadSanitizer y lo hace fallar.
void project_portable(const Problem& problem, std::size_t first, std::size_t last,
                      double* panel) {
    project_work(problem, first, last, panel);
}

#if (defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
#define MARS_TITAN_PROJECTION_X86
__attribute__((target("avx2"))) void project_avx2(const Problem& problem, std::size_t first,
                                                   std::size_t last, double* panel) {
    project_work(problem, first, last, panel);
}

__attribute__((target("avx512f"))) void project_avx512f(const Problem& problem,
                                                         std::size_t first, std::size_t last,
                                                         double* panel) {
    project_work(problem, first, last, panel);
}
#endif

void project_range(const Problem& problem, ProjectionIsa isa, std::size_t first,
                   std::size_t last, double* panel) {
    switch (isa) {
#ifdef MARS_TITAN_PROJECTION_X86
    case ProjectionIsa::avx512f:
        project_avx512f(problem, first, last, panel);
        return;
    case ProjectionIsa::avx2:
        project_avx2(problem, first, last, panel);
        return;
#endif
    default:
        project_portable(problem, first, last, panel);
        return;
    }
}

bool supported(ProjectionIsa isa) {
#ifdef MARS_TITAN_PROJECTION_X86
    // Idempotente. Garantiza la detección aunque la primera llamada llegue desde un constructor
    // estático que se ejecute antes que el de la biblioteca de soporte del compilador.
    __builtin_cpu_init();
#endif
    switch (isa) {
    case ProjectionIsa::portable:
        return true;
#ifdef MARS_TITAN_PROJECTION_X86
    case ProjectionIsa::avx2:
        return __builtin_cpu_supports("avx2") != 0;
    case ProjectionIsa::avx512f:
        return __builtin_cpu_supports("avx512f") != 0;
#else
    case ProjectionIsa::avx2:
    case ProjectionIsa::avx512f:
        return false;
#endif
    }
    return false;
}

bool shape_matches(std::size_t first, std::size_t second, std::size_t size) {
    return first <= std::numeric_limits<std::size_t>::max() / second && first * second == size;
}

// Comprueba formas y versión antes de escribir nada en la salida.
Problem checked(std::span<const double> block, std::size_t rows, std::size_t width,
                std::span<const double> transposed, std::size_t outputs, std::span<double> out,
                ProjectionIsa isa) {
    if (width == 0 || outputs == 0) {
        throw std::invalid_argument("La proyección del codec necesita anchura y salidas");
    }
    if (!shape_matches(rows, width, block.size()) ||
        !shape_matches(width, outputs, transposed.size()) ||
        !shape_matches(rows, outputs, out.size())) {
        throw std::invalid_argument("Las formas de la proyección del codec no coinciden");
    }
    if (!supported(isa)) {
        throw std::invalid_argument("El procesador no admite la versión pedida de la proyección");
    }
    return Problem{block.data(), rows, width, transposed.data(), outputs, out.data()};
}
} // namespace

std::vector<ProjectionIsa> available_projection_isas() {
    std::vector<ProjectionIsa> result;
    for (const ProjectionIsa isa :
         {ProjectionIsa::portable, ProjectionIsa::avx2, ProjectionIsa::avx512f}) {
        if (supported(isa)) {
            result.push_back(isa);
        }
    }
    return result;
}

ProjectionIsa default_projection_isa() {
    static const ProjectionIsa widest = available_projection_isas().back();
    return widest;
}

std::string_view projection_isa_name(ProjectionIsa isa) {
    switch (isa) {
    case ProjectionIsa::portable:
        return "portable";
    case ProjectionIsa::avx2:
        return "avx2";
    case ProjectionIsa::avx512f:
        return "avx512f";
    }
    throw std::invalid_argument("Versión desconocida del núcleo de proyección");
}

void project_block(std::span<const double> block, std::size_t rows, std::size_t width,
                   std::span<const double> transposed, std::size_t outputs, std::span<double> out) {
    project_block(block, rows, width, transposed, outputs, out, default_projection_isa());
}

void project_block(std::span<const double> block, std::size_t rows, std::size_t width,
                   std::span<const double> transposed, std::size_t outputs, std::span<double> out,
                   ProjectionIsa isa) {
    project_block_groups(block, rows, width, transposed, outputs, out, isa, 0,
                         projection_groups(outputs));
}

void project_block_groups(std::span<const double> block, std::size_t rows, std::size_t width,
                          std::span<const double> transposed, std::size_t outputs,
                          std::span<double> out, ProjectionIsa isa, std::size_t first,
                          std::size_t last) {
    const Problem problem = checked(block, rows, width, transposed, outputs, out, isa);
    if (first > last || last > projection_groups(outputs)) {
        throw std::invalid_argument("Los grupos pedidos no existen en la proyección del codec");
    }
    if (rows == 0 || first == last) {
        return;
    }
    // Un panel propio por llamada, así que llamadas con grupos distintos pueden ir a la vez.
    std::vector<double> panel(rows < projection_packed_rows ? 0 : projection_panel_size(width));
    project_range(problem, isa, first, last, panel.data());
}

std::size_t projection_groups(std::size_t outputs) {
    return outputs / projection_lanes + (outputs % projection_lanes == 0 ? 0 : 1);
}

std::size_t projection_panel_size(std::size_t width) {
    if (width > std::numeric_limits<std::size_t>::max() / projection_lanes) {
        throw std::invalid_argument("El panel de la proyección no cabe en memoria");
    }
    return width * projection_lanes;
}

// La firma conserva el orden público de las dimensiones y el presupuesto de hilos. El enlace
// pasa outputs de la matriz y available de ATen. threads_need_enough_work contrasta ambos.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
std::size_t projection_threads(std::size_t rows, std::size_t width, std::size_t outputs,
                               std::size_t available) {
    // rows · width · outputs ≥ projection_parallel_work sin calcular el producto de los tres.
    const bool parallel =
        rows > 0 && width * outputs >= (projection_parallel_work + rows - 1) / rows;
    return parallel ? std::max<std::size_t>(available, 1) : 1;
}

} // namespace mars_titan::episodic
