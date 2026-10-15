#ifndef MARS_TITAN_MEDOID_KERNELS_HPP
#define MARS_TITAN_MEDOID_KERNELS_HPP

#include <cstddef>
#include <span>

namespace mars_titan::medoids {

// Mínimo por cliente de la distancia de referencia de `cm/medoids.py` a los centros fijos,
// para un bloque de clientes. La distancia empieza en +0 y, dimensión a dimensión y en orden,
// d = hypot(d, x - y), con las coordenadas originales.
//
// Es la parte de `_bounded_minimum` (`cm/anchored_medoids.py`) que sigue al producto de
// matrices. Con un centro c cualquiera, x' = fl(x - c) e y' = fl(y - c), el llamador da:
//   products      rows × centers, x'·y' en float64 con cualquier orden de suma. Se sobrescribe
//                 con el cuadrado calculado q + p - 2 x'·y'.
//   client_norms  rows, q = x'·x'.
//   center_norms  centers, p = y'·y'.
//   clients       rows × dimensions, coordenadas originales del bloque.
//   fixed         centers × dimensions, coordenadas originales de los centros.
// y nearest (rows) recibe el mínimo.
//
// El resultado tiene los mismos bits que el mínimo de la distancia de referencia sobre todos
// los centros, porque es el mínimo de distancias calculadas exactamente igual, y el mínimo de
// valores exactos no depende del orden. Solo se calcula la distancia de los pares que pueden
// ser el mínimo: los demás se descartan con la cota inferior de `_bounded_minimum`, con las
// mismas operaciones por par. La primera distancia es la del primer mínimo del cuadrado, y
// el mínimo provisional baja con cada distancia calculada.
//
// La cota está demostrada para coordenadas que vienen de FP32, como las claves del banco
// episódico, así que cada coordenada debe ser finita y representable en float sin redondeo.
// Las normas y los productos deben ser finitos. La distancia usa la hypot de la biblioteca de
// C, la misma que NumPy, y la compilación necesita -ffp-contract=off.
//
// Lanza std::invalid_argument, sin escribir en nearest, si las formas no encajan, si no hay
// dimensiones o centros, o si una entrada no cumple lo anterior. Reserva 8 · centers bytes.
void nearest_reference_block(std::span<double> products, std::span<const double> client_norms,
                             std::span<const double> center_norms,
                             std::span<const double> clients, std::span<const double> fixed,
                             std::size_t dimensions, std::span<double> nearest);

// Cota inferior que usa nearest_reference_block para descartar un par, con el cuadrado
// calculado `square`, las raíces `left` y `right` de las normas centradas y `dimensions`
// coordenadas. Con l = left + right y u = 2^-53, y en este orden de operaciones, como
// `_bounded_minimum`: (sqrt(max(square - l · l · 4(D + 4)u, 0)) - l · 4u) · (1 - (8D + 2048)u).
// Se publica para comprobar en las pruebas que el núcleo usa la cota demostrada.
[[nodiscard]] double background_bound(double square, double left, double right,
                                      std::size_t dimensions);

// Distancia de referencia entre dos puntos de `dimensions` coordenadas.
double reference_distance(const double* left, const double* right, std::size_t dimensions);

// Vista de una matriz float64 con pasos en elementos: el valor (fila, columna) está en
// data[fila · row_stride + columna · column_stride].
struct StridedMatrix {
    const double* data;
    std::size_t rows;
    std::size_t columns;
    std::size_t row_stride;
    std::size_t column_stride;
};

// Coste de cada candidato en un bloque de clientes de `_Costs.proposals` (`cm/medoids.py`):
// totals[c] = math.fsum(min(distance[r, c], base[r]) para cada fila r). La suma es la de
// parciales de Shewchuk que usa math.fsum (`AccurateSum`), redondeada una sola vez al par más
// cercano, así que da los mismos bits. El mínimo es el de numpy.minimum.
//
// Devuelve false, sin escribir en totals, si un valor es NaN o infinito, si una suma parcial
// desborda o si un total no es finito. El llamador repite entonces la suma de referencia, que
// conserva sus errores. Lanza std::invalid_argument si las formas no encajan.
[[nodiscard]] bool min_column_sums(const StridedMatrix& distance, std::span<const double> base,
                                   std::span<double> totals);

} // namespace mars_titan::medoids

#endif
