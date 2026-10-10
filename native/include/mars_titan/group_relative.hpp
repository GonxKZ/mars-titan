#ifndef MARS_TITAN_GROUP_RELATIVE_HPP
#define MARS_TITAN_GROUP_RELATIVE_HPP

#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace mars_titan::learning {
// Objetivos relativos al grupo. Varios episodios completos parten del mismo estado real de una
// cinta de ajuste y la media del grupo hace de línea base, sin crítico. Las cuatro identidades
// siguen a sus artículos (GRPO, Dr. GRPO, DAPO y GSPO) y solo difieren en cómo normalizan la
// ventaja, en qué nivel calculan el cociente, cómo agregan las decisiones y si usan término KL.
enum class GroupObjectiveKind : uint8_t { grpo, dr_grpo, dapo, gspo };
enum class GroupAdvantage : uint8_t { mean_std, mean };
enum class GroupRatio : uint8_t { token, sequence };
enum class GroupAggregation : uint8_t {
    sequence_mean_token_mean,
    sequence_mean_token_sum_constant,
    token_mean,
    sequence_mean
};

inline constexpr std::string_view group_relative_controller = "group_relative_fresh_waves_v1";
inline constexpr std::int64_t maximum_group_batch = 128;
inline constexpr std::int64_t maximum_group_length = 256;
inline constexpr std::size_t maximum_group_bytes = std::size_t{64} * 1024 * 1024;
// Recorte de PPO que DeepSeekMath no publica y que Dr. GRPO y DAPO fijan en 0,2.
inline constexpr double default_group_clip = 0.2;
inline constexpr double default_group_advantage_epsilon = 1e-6;

struct GroupObjectiveConfig {
    GroupObjectiveKind kind = GroupObjectiveKind::grpo;
    double clip_low = default_group_clip;
    double clip_high = default_group_clip;
    // Peso del estimador k3 de KL(pi_theta || q). Solo GRPO lo usa y en el resto vale cero.
    double kl_beta = 0;
    // Suelo que se suma a la desviación del grupo para no dividir por cero cuando todos los
    // retornos coinciden. Dr. GRPO no divide y lo deja en cero.
    double advantage_epsilon = default_group_advantage_epsilon;
    // Dr. GRPO divide la suma por una constante en lugar de por la longitud de cada episodio.
    // Aquí es el número máximo de decisiones que admite un episodio, como su MAX_TOKENS.
    std::size_t length_normalizer = static_cast<std::size_t>(maximum_group_length);
    void validate() const;
    [[nodiscard]] std::string id() const;
    [[nodiscard]] GroupAdvantage advantage() const noexcept;
    [[nodiscard]] GroupRatio ratio() const noexcept;
    [[nodiscard]] GroupAggregation aggregation() const noexcept;
    bool operator==(const GroupObjectiveConfig&) const noexcept = default;
};

[[nodiscard]] GroupObjectiveKind group_objective_kind(std::string_view id);

// Constantes publicadas de cada identidad, que no se leen de la configuración. GRPO usa la
// beta 0,04 de DeepSeekMath y un recorte 0,2 que el artículo no publica. Dr. GRPO quita la
// desviación y normaliza por una constante. DAPO recorta entre 0,2 y 0,28 y GSPO entre 3e-4 y
// 4e-4 sobre el cociente de secuencia. Ninguno salvo GRPO lleva término KL.
[[nodiscard]] GroupObjectiveConfig published_group_objective(std::string_view id);

// Ventajas [B] FP64 de cada episodio respecto a su grupo, identificado por un entero no negativo.
// mean_std divide por la desviación muestral (G-1) más epsilon, porque DeepSeekMath no fija el
// estimador, y mean solo centra como Dr. GRPO. Un grupo de un único episodio no tiene línea base
// y se rechaza en lugar de dar una ventaja nula engañosa.
[[nodiscard]] at::Tensor group_advantages(const at::Tensor& returns, const at::Tensor& groups,
                                          GroupAdvantage advantage, double epsilon);

// Pesos [B] FP64 que multiplican la suma de cada episodio para formar la pérdida de la oleada.
// decisions [B] cuenta las decisiones muestreadas de cada episodio. Así cada identidad reproduce
// su agregación (media por secuencia, suma por constante, media por decisión o por secuencia)
// aunque la oleada se procese en bloques.
[[nodiscard]] at::Tensor group_episode_weights(const at::Tensor& decisions,
                                               const GroupObjectiveConfig& config);

// Diagnósticos para las trazas de aprendizaje. Se calculan sin gradiente después de la pérdida
// y no intervienen en ella, de modo que activarlos no cambia ningún bit del resultado.
struct GroupObjectiveTrace {
    std::size_t episodes = 0;
    std::size_t decisions = 0;
    double entropy = 0;
    double full_kl = 0;
    double k3_kl = 0;
    double ratio_mean = 0;
    double ratio_min = 0;
    double ratio_max = 0;
    double clip_fraction = 0;
    double advantage_mean = 0;
    double advantage_std = 0;
    double advantage_min = 0;
    double advantage_max = 0;
};

// Pérdida por episodio [B] FP64 que se minimiza, cuya suma es la pérdida de la oleada. Recibe
// logp y logq [B,T,6], acciones int64 y máscara bool [B,T] con las decisiones seguidas de padding,
// y ventajas y pesos [B] fijos. Solo logp recibe gradiente. q es el muestreador que generó la
// oleada, así que hace a la vez de política antigua del cociente y de referencia del KL.
// Acumula los diagnósticos de un bloque en los de la oleada. Entropía, KL y cocientes por
// decisión se ponderan por decisiones, y los cocientes de secuencia de GSPO por episodios. Los
// extremos se combinan con mínimo y máximo. Las ventajas se describen aparte con la oleada.
void accumulate_group_trace(GroupObjectiveTrace& total, const GroupObjectiveTrace& block,
                            GroupRatio ratio);

[[nodiscard]] at::Tensor
group_relative_loss(const at::Tensor& logp, const at::Tensor& logq, const at::Tensor& actions,
                    const at::Tensor& advantages, const at::Tensor& weights,
                    const at::Tensor& policy_mask, const GroupObjectiveConfig& config,
                    GroupObjectiveTrace* trace = nullptr);
} // namespace mars_titan::learning

#endif
