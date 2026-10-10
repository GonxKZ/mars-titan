#ifndef MARS_TITAN_GROUP_WAVES_HPP
#define MARS_TITAN_GROUP_WAVES_HPP

#include "mars_titan/group_relative.hpp"
#include "mars_titan/klpo_episodes.hpp"
#include "mars_titan/ppo_policy.hpp"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

namespace mars_titan::learning {
// Regla de grupos de las oleadas. Los carriles recorren las cintas de ajuste como en KLPO
// (carril l, cinta l módulo n), así que todos los episodios de una misma cinta parten del mismo
// estado real (primera sesión y caja inicial) y forman un grupo. El grupo se fija antes de
// recoger y nunca depende del resultado.
inline constexpr std::string_view group_wave_rule = "episodes_from_the_same_train_tape_v1";

// Ventajas y pesos de una oleada completa, calculados antes de dividirla en bloques para que
// el gradiente no dependa del tamaño de bloque. Todos los tensores son FP64 en CPU.
struct GroupWave {
    std::vector<int64_t> labels;
    std::vector<std::size_t> sizes;
    at::Tensor returns;
    at::Tensor advantages;
    at::Tensor weights;
    std::size_t sampled_decisions = 0;
};

// Agrupa los episodios por cinta de origen y exige que cada grupo tenga al menos dos episodios
// y la misma primera observación, comprobada bit a bit. El retorno es la suma de recompensas
// con el gamma de la oleada, que debe ser uno para que sea el resultado del episodio completo.
[[nodiscard]] GroupWave group_wave(const KlpoEpisodeBatch& wave,
                                   const GroupObjectiveConfig& config);

// Pérdida escalar con grafo del actor para los episodios [begin, begin + block.episodes.size())
// de la oleada. La suma de todos los bloques es la pérdida de la oleada. Devuelve un tensor
// indefinido si el bloque no tiene decisiones. No llama a backward.
[[nodiscard]] at::Tensor group_block_loss(const PpoPolicy& actor, const KlpoEpisodeBatch& block,
                                          std::size_t begin, const GroupWave& wave,
                                          const GroupObjectiveConfig& config,
                                          GroupObjectiveTrace* trace = nullptr);

// Diagnósticos de una oleada para las trazas de aprendizaje. Se rellenan solo si se piden y
// no intervienen en el gradiente.
struct GroupWaveTrace {
    GroupObjectiveTrace objective;
    std::size_t groups = 0;
    std::size_t smallest_group = 0;
    std::size_t largest_group = 0;
    // Grupos en los que todos los retornos coinciden, cuya ventaja es nula y no aportan señal.
    std::size_t flat_groups = 0;
    // Media de la desviación muestral de los retornos dentro de cada grupo.
    double return_spread = 0;
};
[[nodiscard]] GroupWaveTrace group_wave_trace(const GroupWave& wave);
} // namespace mars_titan::learning

#endif
