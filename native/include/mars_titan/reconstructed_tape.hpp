#ifndef MARS_TITAN_RECONSTRUCTED_TAPE_HPP
#define MARS_TITAN_RECONSTRUCTED_TAPE_HPP

#include "mars_titan/financial_session.hpp"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mars_titan::simulation {

// Tramo walk-forward declarado por un recibo de ventana, en microsegundos UTC.
struct WalkForwardSegment {
    int64_t start = 0;
    int64_t end = 0;
    int64_t labels_used_until = 0;
};

// Salida acreditada de una baja: importe por acción, instante de cobro y huella de su fuente.
struct ExitRecord {
    double price = 0;
    int64_t pay_at = 0;
    std::string source_sha256;
};

// Baja de un activo en la apertura siguiente a la última sesión de su serie.
struct DelistingRecord {
    std::string last_session;
    std::optional<ExitRecord> exit;
};

// Estado acreditado de una acción A (simulation/listing_status.py), con fechas ISO de Pekín.
// Los tramos de advertencia de riesgo (ST) y de reforma accionarial pendiente (S) son
// [inicio, fin) y solo el último de cada lista puede no tener fin. Los días sin límite siguen
// a una reforma o a una reanudación de cotización.
struct ChinaStatus {
    std::string listed_on;
    std::optional<std::string> limit_free_until;
    std::vector<std::pair<std::string, std::optional<std::string>>> special_treatment;
    std::vector<std::pair<std::string, std::optional<std::string>>> share_reform_pending;
    std::vector<std::string> limit_free_days;
};

// Auditoría de una cinta reconstruida (simulation/reconstructed_tape.py), leída antes del Parquet.
struct ReconstructedAudit {
    std::string market;
    std::vector<WalkForwardSegment> segments;
    std::vector<int64_t> prediction_fit_ends;
    int64_t dividend_payment_lag_sessions = 0;
    std::string basis;
    // Activos del diseño fuera del universo del tramo, sin precios ni predicciones.
    std::vector<std::string> outside_universe;
    std::map<std::string, DelistingRecord> delistings;
    std::map<std::string, ChinaStatus> listing_status;
    std::string listing_status_sha256;
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

// Reglas del tablero con el estado acreditado del activo, como market_rules.status_price_limits:
// sin banda en los días sin límites tras la salida a bolsa, una reforma o una reanudación, y
// el 5 % en los tramos ST o S del tablero principal. Los tramos contiguos con la misma banda
// se unen.
[[nodiscard]] InstrumentRules china_a_share_rules(std::string_view asset,
                                                  const ChinaStatus& status);

} // namespace mars_titan::simulation

#endif
