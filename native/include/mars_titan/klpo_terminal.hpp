#ifndef MARS_TITAN_KLPO_TERMINAL_HPP
#define MARS_TITAN_KLPO_TERMINAL_HPP

#include <ATen/core/Tensor.h>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace mars_titan::learning {
inline constexpr std::string_view klpo_terminal_contract = "klpo_terminal_token_full_v1";
inline constexpr std::int64_t maximum_klpo_terminal_batch = 128;
inline constexpr std::int64_t maximum_klpo_terminal_length = 256;
inline constexpr std::size_t maximum_klpo_terminal_bytes = std::size_t{64} * 1024 * 1024;

struct KlpoTerminalBudget {
    std::size_t max_working_bytes = maximum_klpo_terminal_bytes;
};

// Sustituto Full-KL [B] FP64. Suma decisiones completas, sin promediar trayectorias.
// logp/logq [B,T,6], acciones int64 y máscara bool [B,T], retornos fijos [B].
// La máscara contiene decisiones seguidas de padding. Solo logp recibe gradiente.
// El presupuesto estima entradas, temporales y buffers propios de autograd,
// sin incluir el grafo previo del actor, el allocator ni el contexto del dispositivo.
[[nodiscard]] at::Tensor klpo_terminal_full_loss(const at::Tensor& logp, const at::Tensor& logq,
                                                 const at::Tensor& actions,
                                                 const at::Tensor& terminal_returns,
                                                 const at::Tensor& policy_mask, double beta,
                                                 KlpoTerminalBudget budget = {});
} // namespace mars_titan::learning

#endif
