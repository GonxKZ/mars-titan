#ifndef MARS_TITAN_QUANTILE_DQN_HPP
#define MARS_TITAN_QUANTILE_DQN_HPP

#include <ATen/core/Tensor.h>

#include <cstddef>
#include <cstdint>

namespace mars_titan::learning {
// Cabeza cuantílica de QR-DQN sobre las seis acciones de exposición.
inline constexpr std::int64_t maximum_quantiles = 256;
inline constexpr std::int64_t maximum_quantile_batch = 4096;

// Puntos medios tau_i = (2i+1)/(2N) [N] FP64 en el dispositivo pedido.
[[nodiscard]] at::Tensor quantile_midpoints(std::int64_t quantiles, const at::Device& device);

// Puntuación de cada acción [B,6] a partir de los cuantiles [B,6,N] ordenados por nivel tau.
// alpha = 1 da la media. alpha < 1 promedia los alpha*N primeros niveles, que aproxima
// CVaR_alpha por la regla del punto medio. alpha*N debe ser entero. Conserva el tipo de entrada.
[[nodiscard]] at::Tensor quantile_action_scores(const at::Tensor& quantiles, double risk_alpha);

// Objetivo distribucional [B,N] sin gradiente: r + gamma (1 - terminado) theta_objetivo(s', a*),
// con a* elegida por la red online sobre la misma puntuación de riesgo (selección doble).
// Descuento y nivel de riesgo van juntos en una estructura para que no puedan intercambiarse
// por error en la llamada, ya que ambos son reales entre cero y uno.
struct QuantileTargetOptions {
    double gamma = 0;
    double risk_alpha = 1;
};
[[nodiscard]] at::Tensor quantile_double_targets(const at::Tensor& rewards,
                                                 const at::Tensor& terminated,
                                                 const at::Tensor& online_next,
                                                 const at::Tensor& target_next,
                                                 QuantileTargetOptions options);

// Pérdida cuantílica de Huber por transición [B] FP64, según el algoritmo 1 de QR-DQN:
// sum_i mean_j |tau_i - 1{u_ij < 0}| L_kappa(u_ij), con u_ij = T_j - theta_i y sin dividir
// por kappa, como la ecuación 10 del artículo. Solo los cuantiles predichos reciben gradiente.
[[nodiscard]] at::Tensor quantile_huber_loss(const at::Tensor& predicted, const at::Tensor& targets,
                                             double kappa);
} // namespace mars_titan::learning

#endif
