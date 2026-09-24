#ifndef MARS_TITAN_FINANCIAL_SESSION_HPP
#define MARS_TITAN_FINANCIAL_SESSION_HPP

#include "mars_titan/simulation.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
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

enum class CorporateKind : uint8_t { split, dividend, writeoff };

struct CorporateAction {
    std::string id;
    std::size_t asset = 0;
    CorporateKind kind = CorporateKind::split;
    int64_t effective_at = 0;
    double value = 0;
    std::optional<int64_t> pay_at;
    bool verified = false;
};

struct MarketTape {
    std::vector<std::string> assets;
    std::vector<int64_t> open_times;
    std::vector<int64_t> close_times;
    std::vector<int64_t> prediction_times;
    std::vector<double> prices;
    std::vector<double> scores;
    std::vector<CorporateAction> actions;
    std::string currency;
    std::string domain;
    std::string partition;
    std::string parent_id;
    std::string source_sha256;
    bool historical_audit_verified = false;

    void validate() const;
    [[nodiscard]] std::span<const double>
    frame(std::size_t session) const & MARS_TITAN_LIFETIME_BOUND;
    [[nodiscard]] std::span<const double>
    predictions(std::size_t session) const & MARS_TITAN_LIFETIME_BOUND;
    std::span<const double> frame(std::size_t session) const && = delete;
    std::span<const double> predictions(std::size_t session) const && = delete;
};

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
    std::optional<double> max_drawdown;
    double costs = 0;
    double turnover = 0;
    std::size_t steps = 0;
    bool completed = false;
    std::string invalid_reason;
};

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
    [[nodiscard]] SessionSnapshot snapshot() const;
    void restore(const SessionSnapshot& snapshot);
    [[nodiscard]] FinancialMetrics metrics() const;
    [[nodiscard]] bool done() const noexcept;
    [[nodiscard]] std::size_t cursor() const noexcept;

private:
    std::shared_ptr<const MarketTape> tape_;
    Parameters parameters_;
    SessionSnapshot state_;
    SessionSnapshot staged_;
    std::vector<mt_position_v1> next_positions_;
    std::vector<mt_trade_v1> trades_;
    std::vector<uint32_t> currencies_;
    std::vector<double> lots_;
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
