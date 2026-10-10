#ifndef MARS_TITAN_FINANCIAL_SESSION_HPP
#define MARS_TITAN_FINANCIAL_SESSION_HPP

#include "mars_titan/simulation.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#if __has_cpp_attribute(clang::lifetimebound)
#define MARS_TITAN_LIFETIME_BOUND [[clang::lifetimebound]]
#elif __has_cpp_attribute(msvc::lifetimebound)
#define MARS_TITAN_LIFETIME_BOUND [[msvc::lifetimebound]]
#else
#define MARS_TITAN_LIFETIME_BOUND
#endif

namespace mars_titan::simulation {

inline constexpr std::size_t maximum_assets = 4096;
inline constexpr std::size_t maximum_sessions = 8192;
inline constexpr std::size_t maximum_rows = 1'048'576;
inline constexpr std::size_t maximum_actions = 65'536;
inline constexpr double default_capital = 10'000;
inline constexpr double default_cost_bps = 10;
inline constexpr double default_participation = 0.01;
inline constexpr double default_score_scale = 0.01;
inline constexpr double default_ruin_penalty = -20;

// Una baja con precio cambia la posición por su cobro y retira el activo. Una baja sin precio
// de salida deja sin valorar la posición abierta y solo retira el activo si no había posición.
enum class CorporateKind : uint8_t { split, dividend, writeoff, delisting, unpriced_delisting };

struct CorporateAction {
    std::string id;
    std::size_t asset = 0;
    CorporateKind kind = CorporateKind::split;
    int64_t effective_at = 0;
    double value = 0;
    std::optional<int64_t> pay_at;
    bool verified = false;
};

// Valores vigentes en [start, end), en microsegundos UTC, como Period en Python.
struct RulePeriod {
    int64_t start = 0;
    int64_t end = 0;
    double band = 0;
    double buy = 0;
    double sell = 0;
    bool operator==(const RulePeriod&) const = default;
};

// Reglas de un activo con la semántica de Instrument. Sin identidad solo conservan el lote.
struct InstrumentRules {
    std::string rules;
    double lot = 1;
    double minimum_order = 0;
    bool odd_lot_exit = false;
    std::vector<RulePeriod> price_limits;
    std::vector<RulePeriod> taxes;
    bool operator==(const InstrumentRules&) const = default;
};

inline constexpr std::string_view market_rules_contract = "mt_simulation_step_v2/mt_rules_v1";
// Única cinta real admitida: edición reconstruida con predicciones fuera de muestra (#390).
inline constexpr std::string_view reconstructed_tape_contract =
    "unadjusted_reconstructed_walk_forward_v1";

struct MarketTape {
    std::vector<std::string> assets;
    std::vector<int64_t> open_times;
    std::vector<int64_t> close_times;
    std::vector<int64_t> prediction_times;
    std::vector<double> prices;
    std::vector<double> scores;
    std::vector<CorporateAction> actions;
    // Vacío sin reglas declaradas. Si existe, una entrada por activo en el orden de assets.
    std::vector<InstrumentRules> instruments;
    std::string currency;
    std::string domain;
    std::string partition;
    std::string parent_id;
    std::string source_sha256;
    // En una cinta real, la partición es su papel en la política y los cortes salen de los
    // recibos walk-forward: el último dato de ajuste del predictor de cada sesión. El lector
    // solo marca historical_audit_verified después de comprobar el contrato reconstruido.
    std::vector<int64_t> prediction_fit_ends;
    // Mercado, edición y supuestos de la auditoría real. Vacío en las cintas sintéticas.
    std::string historical_basis;
    bool historical_audit_verified = false;

    void validate() const;
    void validate_delistings() const;
    [[nodiscard]] bool has_market_rules() const noexcept;
    [[nodiscard]] std::span<const double>
    frame(std::size_t session) const & MARS_TITAN_LIFETIME_BOUND;
    [[nodiscard]] std::span<const double>
    predictions(std::size_t session) const & MARS_TITAN_LIFETIME_BOUND;
    std::span<const double> frame(std::size_t session) const && = delete;
    std::span<const double> predictions(std::size_t session) const && = delete;
};

// Fuentes que una misma política puede combinar. Una cinta sintética exige el mismo predictor
// padre. En una cinta real cada ventana walk-forward reajusta el predictor y cambia su padre,
// así que se exige la misma base histórica: mercado, edición y plazo de pago declarados.
[[nodiscard]] bool same_policy_origin(const MarketTape& left, const MarketTape& right) noexcept;

struct Parameters {
    double capital = default_capital;
    double cost_bps = default_cost_bps;
    double participation = default_participation;
    double score_scale = default_score_scale;
    double ruin_penalty = default_ruin_penalty;
    bool operator==(const Parameters&) const = default;
};

struct Receivable {
    std::size_t action = 0;
    int64_t pay_at = 0;
    double amount = 0;
};

struct SessionSnapshot {
    std::string source_sha256;
    Parameters parameters;
    std::size_t cursor = 0;
    bool done = false;
    std::vector<mt_position_v1> positions;
    mt_account_v1 account{};
    std::vector<uint8_t> retired;
    std::vector<uint8_t> applied_actions;
    std::vector<Receivable> receivables;
    std::vector<std::size_t> held_order;
    double peak_nav = 0;
    double max_drawdown = 0;
    std::size_t unfilled_order_observations = 0;
};

struct StepOutcome {
    double reward = 0;
    bool reward_valid = false;
    bool terminated = false;
    bool truncated = false;
    std::vector<mt_trade_v1> trades;
    std::vector<std::size_t> unvalued;
};

struct FinancialMetrics {
    std::optional<double> net_return;
    // Retorno si se vendiera la cartera al último cierre pagando el coste y el impuesto de venta.
    std::optional<double> liquidated_net_return;
    std::optional<double> max_drawdown;
    double costs = 0;
    double turnover = 0;
    std::size_t steps = 0;
    bool completed = false;
    std::string invalid_reason;
};

/* Patrimonio tras vender todas las posiciones al cierre del cursor con el coste de la sesión
 * y el impuesto de venta de cada activo vigente en ese cierre. Devuelve NaN si falta un cierre
 * de una posición. No modifica la cartera. */
[[nodiscard]] double liquidated_nav(const SessionSnapshot& state, const MarketTape& tape);

/* Indica si el patrimonio desconocido procede de una posición abierta en una baja sin precio de
 * salida, el motivo `unpriced_exit`, y no de un cierre ausente. */
[[nodiscard]] bool unpriced_exit(const SessionSnapshot& state, const MarketTape& tape);

/* Cada sesión posee su estado. Las sesiones distintas comparten únicamente la cinta inmutable. */
class FinancialSession {
public:
    explicit FinancialSession(std::shared_ptr<const MarketTape> tape, Parameters parameters = {});
    FinancialSession(const FinancialSession&) = delete;
    FinancialSession& operator=(const FinancialSession&) = delete;
    FinancialSession(FinancialSession&&) noexcept = default;
    FinancialSession& operator=(FinancialSession&&) noexcept = default;
    ~FinancialSession() = default;

    [[nodiscard]] StepOutcome step(uint8_t action);
    [[nodiscard]] std::vector<float> observation() const;
    void observation_into(std::span<float> destination) const;
    [[nodiscard]] SessionSnapshot snapshot() const;
    void restore(const SessionSnapshot& snapshot);
    [[nodiscard]] FinancialMetrics metrics() const;
    [[nodiscard]] bool done() const noexcept;
    [[nodiscard]] std::size_t cursor() const noexcept;

private:
    friend class FinancialBatch;
    struct ValidatedTape {};
    FinancialSession(std::shared_ptr<const MarketTape> tape, Parameters parameters, ValidatedTape);
    [[nodiscard]] static std::shared_ptr<const MarketTape>
    validate_tape(std::shared_ptr<const MarketTape> tape);
    [[nodiscard]] StepOutcome prepare_step(uint8_t action);
    void commit_step() noexcept;
    void observe_state(const SessionSnapshot& state, std::span<float> destination) const;
    void prepare_rules(std::size_t following);
    std::shared_ptr<const MarketTape> tape_;
    Parameters parameters_;
    SessionSnapshot state_;
    SessionSnapshot staged_;
    std::vector<mt_position_v1> next_positions_;
    std::vector<mt_trade_v1> trades_;
    std::vector<uint32_t> currencies_;
    std::vector<double> lots_;
    // Vacío sin reglas. Se rellena en cada paso con la banda, el timbre y la referencia vigentes.
    std::vector<mt_rules_v1> rules_;
    std::vector<std::size_t> ranking_;
    std::vector<std::vector<std::size_t>> actions_by_session_;
};

enum class ReferencePolicy : uint8_t { cash, hold_initial, rebalance_25, rebalance_50, rebalance_75, rebalance_100 };
[[nodiscard]] uint8_t policy_action(ReferencePolicy policy, std::size_t step);
[[nodiscard]] std::string policy_name(ReferencePolicy policy);
[[nodiscard]] ReferencePolicy parse_policy(const std::string& name);

}

#undef MARS_TITAN_LIFETIME_BOUND

#endif
