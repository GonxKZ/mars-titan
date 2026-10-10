#ifndef MARS_TITAN_RECONSTRUCTED_TAPE_HPP
#define MARS_TITAN_RECONSTRUCTED_TAPE_HPP

#include "mars_titan/financial_session.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace mars_titan::simulation {

// Tramo walk-forward declarado por un recibo de ventana, en microsegundos UTC.
struct WalkForwardSegment {
    int64_t start = 0;
    int64_t end = 0;
    int64_t labels_used_until = 0;
};

// Auditoría de una cinta reconstruida (simulation/reconstructed_tape.py), leída antes del Parquet.
struct ReconstructedAudit {
    std::string market;
    std::vector<WalkForwardSegment> segments;
    std::vector<int64_t> prediction_fit_ends;
    int64_t dividend_payment_lag_sessions = 0;
    std::string basis;
};

// Exige el tratamiento declarado de la edición sin suavizar sus limitaciones, como
// MarketTape en Python. Falla antes de abrir el Parquet si falta algún campo del contrato.
[[nodiscard]] ReconstructedAudit read_reconstructed_audit(const nlohmann::json& identity,
                                                          std::string_view currency);

// Comprueba la cinta decodificada frente a su auditoría: cortes de cada sesión, cierres
// valorados, aperturas en rejilla, sesiones sin negociación, eventos con su decisión y
// reglas de acciones A en China. Solo entonces la marca como cinta real verificada.
void admit_reconstructed_tape(MarketTape& tape, const ReconstructedAudit& audit);

// Reglas esperadas de un activo A, con la misma tabla que market_rules.china_a_share_instrument.
[[nodiscard]] InstrumentRules china_a_share_rules(std::string_view asset);

} // namespace mars_titan::simulation

#endif
