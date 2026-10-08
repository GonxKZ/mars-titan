#ifndef MARS_TITAN_PPO_OBJECTIVES_HPP
#define MARS_TITAN_PPO_OBJECTIVES_HPP

#include <ATen/core/Tensor.h>

#include <cstdint>
#include <string_view>

namespace mars_titan::learning {
inline constexpr std::string_view ppo_kl_contract = "categorical_behavior_to_current_kl_v1";
inline constexpr std::int64_t maximum_ppo_kl_rows = 65536;

// KL(q histórica || p actual) por fila [B,6], calculada en FP64 sin actualizar parámetros.
// FP32/FP64, CPU/CUDA y filas normalizadas con tolerancia 1e-6. Se renormalizan
// en FP64 para absorber ese redondeo. -inf representa masa cero, sin añadir un piso.
// Se rechazan pérdida de soporte de q y KL infinita. El gradiente solo pasa por p.
[[nodiscard]] at::Tensor ppo_categorical_kl(const at::Tensor& log_probabilities,
                                           const at::Tensor& old_log_probabilities);
} // namespace mars_titan::learning

#endif
