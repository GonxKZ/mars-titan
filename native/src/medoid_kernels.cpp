#include "mars_titan/medoid_kernels.hpp"

#include "accurate_sum.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <vector>

namespace mars_titan::medoids {
namespace {
// Unidad de redondeo de float64, la de `_UNIT` en `cm/anchored_medoids.py`.
constexpr double unit = 0x1p-53;

bool shape_matches(std::size_t first, std::size_t second, std::size_t size) {
    return first <= std::numeric_limits<std::size_t>::max() / second && first * second == size;
}

bool finite(double value) { return std::isfinite(value); }

// La cota solo está demostrada para coordenadas que vienen de FP32. Se comprueba el rango antes
// de convertir, porque convertir a float un double fuera de rango no está definido.
bool from_float(double value) {
    constexpr auto largest = static_cast<double>(std::numeric_limits<float>::max());
    return std::isfinite(value) && std::fabs(value) <= largest &&
           static_cast<double>(static_cast<float>(value)) == value;
}

// Constantes de la cota de `_bounded_minimum`, con D dimensiones.
struct Bound {
    double relative;
    double absolute;
    double shrink;

    // Cota inferior de la distancia de referencia de un par con cuadrado calculado `square` y
    // suma `limit` de las raíces de las normas centradas. Repite las operaciones de NumPy.
    [[nodiscard]] double operator()(double square, double limit) const {
        double value = square - limit * limit * relative;
        value = std::sqrt(std::max(value, 0.0));
        value -= limit * absolute;
        return value * shrink;
    }
};

Bound distance_bound(std::size_t dimensions) {
    constexpr double rounding_slope = 8;
    constexpr double rounding_guard = 2048;
    return {.relative = 4 * (static_cast<double>(dimensions) + 4) * unit,
            .absolute = 4 * unit,
            .shrink = 1 - (rounding_slope * static_cast<double>(dimensions) + rounding_guard) * unit};
}
} // namespace

double background_bound(double square, double left, double right, std::size_t dimensions) {
    return distance_bound(dimensions)(square, left + right);
}

// NOLINTBEGIN(cppcoreguidelines-pro-bounds-pointer-arithmetic)
// Los índices salen de las formas comprobadas al entrar.
double reference_distance(const double* left, const double* right, std::size_t dimensions) {
    double distance = 0.0;
    for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
        distance = std::hypot(distance, left[dimension] - right[dimension]);
    }
    return distance;
}

void nearest_reference_block(std::span<double> products, std::span<const double> client_norms,
                             std::span<const double> center_norms,
                             std::span<const double> clients, std::span<const double> fixed,
                             std::size_t dimensions, std::span<double> nearest) {
    const std::size_t rows = nearest.size();
    const std::size_t centers = center_norms.size();
    if (dimensions == 0 || centers == 0) {
        throw std::invalid_argument("El fondo necesita dimensiones y centros fijos");
    }
    if (!shape_matches(rows, centers, products.size()) || client_norms.size() != rows ||
        !shape_matches(rows, dimensions, clients.size()) ||
        !shape_matches(centers, dimensions, fixed.size())) {
        throw std::invalid_argument("Las formas del fondo de medoids no coinciden");
    }
    if (!std::ranges::all_of(clients, from_float) || !std::ranges::all_of(fixed, from_float)) {
        throw std::invalid_argument("La cota del fondo necesita coordenadas FP32 finitas");
    }
    if (!std::ranges::all_of(client_norms, finite) || !std::ranges::all_of(center_norms, finite) ||
        !std::ranges::all_of(products, finite)) {
        throw std::invalid_argument("Los productos y las normas del fondo deben ser finitos");
    }
    std::vector<double> roots(centers);
    std::ranges::transform(center_norms, roots.begin(),
                           [](double value) { return std::sqrt(value); });
    const Bound bound = distance_bound(dimensions);
    for (std::size_t row = 0; row < rows; ++row) {
        const double norm = client_norms[row];
        const double root = std::sqrt(norm);
        // Cuadrado calculado de la distancia centrada con el orden de `_bounded_minimum`.
        double* square = products.data() + row * centers;
        for (std::size_t column = 0; column < centers; ++column) {
            square[column] = square[column] * -2 + norm + center_norms[column];
        }
        // El primer mínimo del cuadrado da la primera distancia de referencia, como en NumPy.
        const auto first =
            static_cast<std::size_t>(std::min_element(square, square + centers) - square);
        const double* client = clients.data() + row * dimensions;
        double best = reference_distance(client, fixed.data() + first * dimensions, dimensions);
        // El mínimo provisional baja con cada distancia calculada, y con él los pares que hay
        // que calcular.
        for (std::size_t column = 0; column < centers; ++column) {
            if (column != first && bound(square[column], root + roots[column]) <= best) {
                best = std::min(best, reference_distance(client, fixed.data() + column * dimensions,
                                                         dimensions));
            }
        }
        nearest[row] = best;
    }
}

bool min_column_sums(const StridedMatrix& distance, std::span<const double> base,
                     std::span<double> totals) {
    if (base.size() != distance.rows || totals.size() != distance.columns) {
        throw std::invalid_argument("Las formas de las sumas de los medoids no coinciden");
    }
    // Columnas por grupo: un acumulador de 1 KiB por columna en la pila y una pasada por fila.
    constexpr std::size_t group = 16;
    std::array<double, group> found{};
    std::vector<double> result(distance.columns);
    for (std::size_t first = 0; first < distance.columns; first += group) {
        const std::size_t width = std::min(group, distance.columns - first);
        std::array<simulation::AccurateSum, group> sums{};
        for (std::size_t row = 0; row < distance.rows; ++row) {
            const double limit = base[row];
            const double* values = distance.data + row * distance.row_stride;
            for (std::size_t column = 0; column < width; ++column) {
                const double value = values[(first + column) * distance.column_stride];
                // numpy.minimum propaga NaN. Una base NaN ya pasa por la comparación y la
                // rechaza add, como cualquier valor no finito.
                if (std::isnan(value) || !sums.at(column).add(value < limit ? value : limit)) {
                    return false;
                }
            }
        }
        for (std::size_t column = 0; column < width; ++column) {
            found.at(column) = sums.at(column).value();
            if (!std::isfinite(found.at(column))) {
                return false;
            }
        }
        std::copy_n(found.begin(), width, result.begin() + static_cast<std::ptrdiff_t>(first));
    }
    std::ranges::copy(result, totals.begin());
    return true;
}
// NOLINTEND(cppcoreguidelines-pro-bounds-pointer-arithmetic)

} // namespace mars_titan::medoids
