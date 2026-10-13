#ifndef MARS_TITAN_EPISODIC_PROJECTION_HPP
#define MARS_TITAN_EPISODIC_PROJECTION_HPP

#include <cstddef>
#include <cstdint>
#include <span>
#include <string_view>
#include <vector>

namespace mars_titan::episodic {

// Salidas que calcula cada vector del núcleo.
inline constexpr std::size_t projection_lanes = 8;

// Versiones del núcleo por conjunto de instrucciones. Todas hacen las mismas operaciones en el
// mismo orden, así que dan los mismos bits. Fuera de x86-64 con GCC o Clang solo existe la
// portable.
enum class ProjectionIsa : std::uint8_t { portable, avx2, avx512f };

// Versiones que admiten esta compilación y este procesador, de la más estrecha a la más ancha.
std::vector<ProjectionIsa> available_projection_isas();
// La versión que usa project_block si no se pide otra: la más ancha disponible. En el Ryzen 9
// 8945HS medido fue la más rápida con todas las formas del codec. Otro procesador necesita su
// propia medida, que da mars-titan-projection-benchmark.
ProjectionIsa default_projection_isa();
std::string_view projection_isa_name(ProjectionIsa isa);

// Proyección congelada del codec episódico: out[r, o] = Σ_i block[r, i] · transposed[i, o].
//
// Reproduce el orden de sumas de `numpy.einsum("bi,oi->bo", optimize=False)` sobre float64 en
// la línea base x86-64 de NumPy 2.5, para obtener los mismos bits. Cada salida tiene dos
// acumuladores, uno para las entradas pares y otro para las impares. Las entradas se recorren
// en bloques de ocho y, dentro de cada bloque, del par 6-7 al par 0-1. El resto se recorre
// por pares en orden creciente, y la salida es par + impar. Cada paso multiplica y después
// suma, sin FMA, así que la compilación necesita -ffp-contract=off o /fp:strict.
//
// einsum suma además 0 · 0 al acumulador impar cuando la anchura es impar y escribe 0 + total.
// Ninguna de las dos sumas cambia un valor finito: los acumuladores empiezan en +0 y, con
// redondeo al más cercano, una suma solo da -0 si los dos sumandos son -0, así que nunca lo
// valen. El núcleo las omite y la prueba compara con una referencia que sí las hace.
//
// block tiene rows × width valores contiguos, transposed width × outputs y out rows × outputs.
// Lanza std::invalid_argument si las formas no encajan o si la versión pedida no está
// disponible.
void project_block(std::span<const double> block, std::size_t rows, std::size_t width,
                   std::span<const double> transposed, std::size_t outputs, std::span<double> out);
void project_block(std::span<const double> block, std::size_t rows, std::size_t width,
                   std::span<const double> transposed, std::size_t outputs, std::span<double> out,
                   ProjectionIsa isa);

} // namespace mars_titan::episodic

#endif
