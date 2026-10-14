#include "mars_titan/medoid_kernels.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
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
using mars_titan::medoids::min_column_sums;
using mars_titan::medoids::nearest_reference_block;
using mars_titan::medoids::StridedMatrix;

constexpr std::uint64_t seed = 20261015;

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

// Mínimo por fuerza bruta de la distancia de `cm/medoids.py`, escrito aparte del núcleo:
// hypot encadenado por dimensión sobre todos los centros.
std::vector<double> brute_force(const std::vector<double>& clients,
                                const std::vector<double>& fixed, std::size_t dimensions) {
    const std::size_t count = clients.size() / dimensions;
    const std::size_t centers = fixed.size() / dimensions;
    std::vector<double> result(count, std::numeric_limits<double>::infinity());
    for (std::size_t row = 0; row < count; ++row) {
        for (std::size_t column = 0; column < centers; ++column) {
            double distance = 0.0;
            for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
                distance = std::hypot(distance, clients.at(row * dimensions + dimension) -
                                                    fixed.at(column * dimensions + dimension));
            }
            result.at(row) = std::min(result.at(row), distance);
        }
    }
    return result;
}

// Órdenes de suma del producto escalar. La cota admite cualquiera, también con FMA.
enum class Order : std::uint8_t { forward, backward, pairs, fused };

double dot(const double* left, const double* right, std::size_t size, Order order) {
    double total = 0.0;
    double odd = 0.0;
    for (std::size_t step = 0; step < size; ++step) {
        const std::size_t index = order == Order::backward ? size - 1 - step : step;
        if (order == Order::fused) {
            total = std::fma(left[index], right[index], total);
        } else if (order == Order::pairs && index % 2 == 1) {
            odd += left[index] * right[index];
        } else {
            total += left[index] * right[index];
        }
    }
    return total + odd;
}

// Coordenadas centradas, productos y normas como los prepara `_bounded_minimum`.
struct Prepared {
    std::vector<double> products;
    std::vector<double> client_norms;
    std::vector<double> center_norms;
};

Prepared prepare(const std::vector<double>& clients, const std::vector<double>& fixed,
                 std::size_t dimensions, const std::vector<double>& center, Order order) {
    const auto centered = [&](const std::vector<double>& points) {
        std::vector<double> result(points.size());
        for (std::size_t index = 0; index < points.size(); ++index) {
            result.at(index) = points.at(index) - center.at(index % dimensions);
        }
        return result;
    };
    const std::vector<double> left = centered(clients);
    const std::vector<double> right = centered(fixed);
    const std::size_t count = clients.size() / dimensions;
    const std::size_t centers = fixed.size() / dimensions;
    Prepared prepared{std::vector<double>(count * centers), std::vector<double>(count),
                      std::vector<double>(centers)};
    for (std::size_t row = 0; row < count; ++row) {
        const double* own = left.data() + row * dimensions;
        prepared.client_norms.at(row) = dot(own, own, dimensions, order);
        for (std::size_t column = 0; column < centers; ++column) {
            prepared.products.at(row * centers + column) =
                dot(own, right.data() + column * dimensions, dimensions, order);
        }
    }
    for (std::size_t column = 0; column < centers; ++column) {
        const double* own = right.data() + column * dimensions;
        prepared.center_norms.at(column) = dot(own, own, dimensions, order);
    }
    return prepared;
}

// Recorre los clientes por bloques de `block` filas con cada orden y dos centros: la media de
// los centros fijos, como `_bounded_minimum`, y el origen.
void same_bits(const std::vector<double>& clients, const std::vector<double>& fixed,
               std::size_t dimensions, std::string_view name) {
    const std::size_t count = clients.size() / dimensions;
    const std::size_t centers = fixed.size() / dimensions;
    const std::vector<double> expected = brute_force(clients, fixed, dimensions);
    std::vector<double> mean(dimensions, 0.0);
    for (std::size_t index = 0; index < fixed.size(); ++index) {
        mean.at(index % dimensions) += fixed.at(index) / static_cast<double>(centers);
    }
    for (const auto& center : {mean, std::vector<double>(dimensions, 0.0)}) {
        for (const Order order : {Order::forward, Order::backward, Order::pairs, Order::fused}) {
            Prepared prepared = prepare(clients, fixed, dimensions, center, order);
            std::vector<double> nearest(count, std::numeric_limits<double>::quiet_NaN());
            constexpr std::size_t block = 7;
            for (std::size_t start = 0; start < count; start += block) {
                const std::size_t rows = std::min(block, count - start);
                nearest_reference_block(
                    std::span(prepared.products).subspan(start * centers, rows * centers),
                    std::span(prepared.client_norms).subspan(start, rows), prepared.center_norms,
                    std::span(clients).subspan(start * dimensions, rows * dimensions), fixed,
                    dimensions, std::span(nearest).subspan(start, rows));
            }
            for (std::size_t row = 0; row < count; ++row) {
                if (std::bit_cast<std::uint64_t>(nearest.at(row)) !=
                    std::bit_cast<std::uint64_t>(expected.at(row))) {
                    throw std::runtime_error(std::string(name) + ": el cliente " +
                                             std::to_string(row) +
                                             " no coincide con la fuerza bruta");
                }
            }
        }
    }
}

// Coordenadas FP32 convertidas a double, como las claves del banco.
template <class Draw> std::vector<double> points(std::size_t size, Draw&& draw) {
    std::vector<double> result(size);
    for (double& value : result) {
        value = static_cast<double>(static_cast<float>(draw()));
    }
    return result;
}

// Nubes gaussianas y clientes alrededor de los centros.
void random_clouds_match_the_brute_force() {
    std::mt19937_64 generator(seed);
    std::normal_distribution<double> normal;
    for (const std::size_t dimensions : {1U, 3U, 64U, 65U}) {
        for (const std::size_t count : {1U, 7U, 8U, 30U}) {
            for (const std::size_t centers : {1U, 2U, 7U, 120U}) {
                const auto fixed = points(centers * dimensions, [&] { return normal(generator); });
                same_bits(points(count * dimensions, [&] { return normal(generator); }), fixed,
                          dimensions, "nube gaussiana");
                // Cada cliente cerca de un centro: muchos pares casi empatados con el mínimo.
                std::uniform_int_distribution<std::size_t> pick(0, centers - 1);
                std::vector<double> near(count * dimensions);
                for (std::size_t row = 0; row < count; ++row) {
                    const std::size_t column = pick(generator);
                    for (std::size_t dimension = 0; dimension < dimensions; ++dimension) {
                        near.at(row * dimensions + dimension) = static_cast<double>(
                            static_cast<float>(fixed.at(column * dimensions + dimension) +
                                               1e-3 * normal(generator)));
                    }
                }
                same_bits(near, fixed, dimensions, "clientes junto a los centros");
            }
        }
    }
}

// Clientes repetidos en los centros, empates exactos y coordenadas lejos del origen con
// diferencias de pocas ulp de FP32, donde q + p - 2 x'·y' cancela casi todo.
void ties_duplicates_and_cancellation_match_the_brute_force() {
    same_bits({0.0, 0.0, 1.0, 1.0, -1.0, 1.0}, {1.0, 1.0, -1.0, 1.0}, 2, "clientes repetidos");
    same_bits({0.0, 0.0, 0.0, 2.0}, {1.0, 0.0, -1.0, 0.0, 0.0, 1.0}, 2, "empates exactos");
    std::mt19937_64 generator(seed + 1);
    std::uniform_int_distribution<int> step(-8, 8);
    for (const float base : {1.0e6F, 1.0e30F, -3.0e38F}) {
        const auto wide = static_cast<double>(base);
        const double ulp = static_cast<double>(std::nextafter(base, 0.0F)) - wide;
        const auto draw = [&] { return wide + ulp * step(generator); };
        same_bits(points(40 * 16, draw), points(25 * 16, draw), 16,
                  "cancelación lejos del origen");
    }
    // Subnormales de FP32: siguen siendo normales en double.
    const auto tiny = static_cast<double>(std::numeric_limits<float>::denorm_min());
    const auto small = [&] { return tiny * step(generator); };
    same_bits(points(30 * 8, small), points(9 * 8, small), 8, "subnormales de FP32");
    // Escalas mezcladas en las mismas filas.
    std::uniform_int_distribution<int> exponent(-140, 120);
    const auto mixed = [&] {
        return std::ldexp(static_cast<double>(step(generator)), exponent(generator));
    };
    same_bits(points(50 * 12, mixed), points(20 * 12, mixed), 12, "escalas mezcladas");
}

// Argumentos válidos de nearest_reference_block, que cada caso de rechazo modifica.
struct Block {
    std::vector<double> products{0.0, 0.0};
    std::vector<double> client_norms{0.5, 4.5};
    std::vector<double> center_norms{0.0};
    std::vector<double> clients{0.0, 1.0, 2.0, 3.0};
    std::vector<double> fixed{1.0, 1.0};
    std::size_t dimensions = 2;

    void run(std::span<double> nearest) {
        nearest_reference_block(products, client_norms, center_norms, clients, fixed, dimensions,
                                nearest);
    }
};

void invalid_inputs_are_rejected_before_writing() {
    std::vector<double> nearest(2, -1.0);
    const auto refused = [&](const std::function<void(Block&)>& change, std::string_view message) {
        Block block;
        change(block);
        rejected([&] { block.run(nearest); }, message);
    };
    refused([](Block& block) { block.dimensions = 0; }, "Se aceptaron cero dimensiones");
    refused(
        [](Block& block) {
            block.products.clear();
            block.center_norms.clear();
            block.fixed.clear();
        },
        "Se aceptó un fondo sin centros");
    refused([](Block& block) { block.dimensions = 1; },
            "Se aceptaron coordenadas con otra anchura");
    refused([](Block& block) { block.products.pop_back(); },
            "Se aceptaron productos con otra forma");
    refused([](Block& block) { block.client_norms.pop_back(); },
            "Se aceptaron normas de clientes con otra forma");
    refused([](Block& block) { block.clients.resize(2); }, "Se aceptaron clientes con otra forma");
    refused([](Block& block) { block.center_norms.push_back(1.0); },
            "Se aceptaron centros con otra forma");
    for (const double value : {std::numeric_limits<double>::quiet_NaN(),
                               std::numeric_limits<double>::infinity(), 0.1, 1.0e300, 1.0e-310}) {
        refused([value](Block& block) { block.clients.at(3) = value; },
                "Se aceptó un cliente que no viene de FP32");
        refused([value](Block& block) { block.fixed.at(1) = value; },
                "Se aceptó un centro que no viene de FP32");
    }
    for (const double value :
         {std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity()}) {
        refused([value](Block& block) { block.products.at(1) = value; },
                "Se aceptó un producto no finito");
        refused([value](Block& block) { block.client_norms.at(0) = value; },
                "Se aceptó una norma de cliente no finita");
        refused([value](Block& block) { block.center_norms.at(0) = value; },
                "Se aceptó una norma de centro no finita");
    }
    require(nearest.at(0) == -1.0 && nearest.at(1) == -1.0, "Una llamada rechazada escribió");
    Block empty;
    empty.products.clear();
    empty.client_norms.clear();
    empty.clients.clear();
    empty.run({});
}

// Sumas de math.fsum con valores elegidos a mano: cancelación, empate al par, mínimo con la
// base y una vista con pasos sobre un búfer mayor, como las de `_Costs`.
void column_sums_take_the_minimum_and_round_once() {
    const double infinity = std::numeric_limits<double>::infinity();
    const double half = std::ldexp(1.0, -53);
    // Tres filas y cuatro columnas útiles dentro de un búfer de 3 × 10, con paso 2 por columna.
    const std::vector<std::vector<double>> columns{
        {1e100, 1.0, -1e100},       // La cancelación deja exactamente 1.
        {1.0, half, 0.0},           // Empate exacto: se redondea al par, 1.
        {1.0, half, std::ldexp(1.0, -80)}, // Por encima del empate: 1 + 2^-52.
        {5.0, 7.0, 9.0},            // Con la base 6 en las dos últimas filas: 5 + 6 + 6.
    };
    const std::vector<double> base{infinity, 6.0, 6.0};
    const std::size_t row_stride = 10;
    std::vector<double> buffer(3 * row_stride, std::numeric_limits<double>::quiet_NaN());
    for (std::size_t column = 0; column < columns.size(); ++column) {
        for (std::size_t row = 0; row < 3; ++row) {
            buffer.at(row * row_stride + 2 * column) = columns.at(column).at(row);
        }
    }
    const StridedMatrix view{buffer.data(), 3, 4, row_stride, 2};
    std::vector<double> totals(4, -1.0);
    require(min_column_sums(view, base, totals), "Las sumas exactas se rechazaron");
    const std::vector<double> expected{1.0, 1.0, 1.0 + std::ldexp(1.0, -52), 17.0};
    for (std::size_t column = 0; column < 4; ++column) {
        require(std::bit_cast<std::uint64_t>(totals.at(column)) ==
                    std::bit_cast<std::uint64_t>(expected.at(column)),
                "Una suma exacta no coincide con math.fsum");
    }
    // Un infinito por debajo de una base finita no llega a la suma, como en numpy.minimum.
    std::vector<double> clipped{infinity, 2.0};
    std::vector<double> one(1, -1.0);
    require(min_column_sums({clipped.data(), 2, 1, 1, 1}, std::vector<double>{3.0, 4.0}, one) &&
                one.at(0) == 5.0,
            "El mínimo con la base no recortó el infinito");
    // NaN en las distancias o en la base, infinitos sin recortar y desbordamientos: false sin
    // escribir.
    const double maximum = std::numeric_limits<double>::max();
    for (const auto& [values, limits] :
         std::vector<std::pair<std::vector<double>, std::vector<double>>>{
             {{std::numeric_limits<double>::quiet_NaN(), 1.0}, {infinity, infinity}},
             {{1.0, 1.0}, {std::numeric_limits<double>::quiet_NaN(), infinity}},
             {{infinity, 1.0}, {infinity, infinity}},
             {{maximum, maximum}, {infinity, infinity}},
         }) {
        std::vector<double> untouched(1, -1.0);
        require(!min_column_sums({values.data(), 2, 1, 1, 1}, limits, untouched) &&
                    untouched.at(0) == -1.0,
                "Una suma no finita se aceptó o escribió");
    }
    rejected([&] { (void)min_column_sums(view, std::vector<double>(2), totals); },
             "Se aceptó una base con otra forma");
    rejected([&] { (void)min_column_sums(view, base, std::span(totals).first(3)); },
             "Se aceptaron totales con otra forma");
}
} // namespace

int main() {
    try {
        random_clouds_match_the_brute_force();
        ties_duplicates_and_cancellation_match_the_brute_force();
        invalid_inputs_are_rejected_before_writing();
        column_sums_take_the_minimum_and_round_once();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Núcleos de los medoids iguales a sus referencias\n";
    return 0;
}
