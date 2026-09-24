#include "mars_titan/simulation.h"
#include "accurate_sum.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <type_traits>
#include <utility>

namespace {
constexpr std::size_t max_assets = 4096;
constexpr std::size_t max_accounts = 32;
constexpr std::size_t price_width = 5;
constexpr std::size_t observation_width = 6;
constexpr std::size_t close_column = 3;
constexpr std::size_t volume_column = 4;
constexpr std::size_t presence_column = 5;
constexpr int binary64_precision = 53;
constexpr std::size_t position_bytes = 32;
constexpr std::size_t account_bytes = 40;
constexpr std::size_t trade_bytes = 32;
constexpr double maximum_cost_rate = 0.1;
constexpr double observation_limit = 10;
constexpr double volume_log_scale = 20;
constexpr double quantity_tolerance = 1e-12;
constexpr double unknown = std::numeric_limits<double>::quiet_NaN();
static_assert(std::numeric_limits<double>::is_iec559);
static_assert(std::numeric_limits<double>::digits == binary64_precision);
static_assert(sizeof(mt_position_v1) == position_bytes && sizeof(mt_account_v1) == account_bytes);
static_assert(sizeof(mt_trade_v1) == trade_bytes);
static_assert(std::is_standard_layout_v<mt_position_v1>);
static_assert(std::is_trivially_copyable_v<mt_position_v1>);

int failure(int code, std::string_view message, char *error, std::size_t capacity) noexcept {
    if (error != nullptr && capacity != 0) {
        const std::span destination{error, capacity};
        const auto size = std::min(message.size(), capacity - 1);
        std::copy_n(message.begin(), size, destination.begin());
        destination[size] = '\0';
    }
    return code;
}

bool nonnegative(double value) noexcept { return std::isfinite(value) && value >= 0; }

bool price_or_missing(double value) noexcept {
    return std::isnan(value) || (std::isfinite(value) && value > 0);
}

using mars_titan::simulation::AccurateSum;
using mars_titan::simulation::CashMovements;

struct Execution {
    double quantity;
    double price;
};

bool fill(mt_position_v1 &position, mt_account_v1 &account, mt_trade_v1 &trade,
          CashMovements &movements, Execution execution, double rate) noexcept {
    const auto [quantity, price] = execution;
    const double notional = quantity * price;
    const double cost = std::abs(notional) * rate;
    if (!std::isfinite(notional) || !std::isfinite(cost) || !movements.add(-notional) ||
        !movements.add(-cost)) {
        return false;
    }
    position.quantity += quantity;
    if (std::abs(position.quantity) < quantity_tolerance) {
        position.quantity = 0;
    }
    account.costs += cost;
    account.turnover += std::abs(notional);
    trade.quantity = quantity;
    trade.price = price;
    trade.cost = cost;
    return nonnegative(position.quantity) && nonnegative(account.costs) &&
           nonnegative(account.turnover);
}

bool valid_positions(std::span<const mt_position_v1> positions,
                     std::span<const uint32_t> currencies, std::span<const double> lots,
                     std::span<const uint8_t> retired, std::size_t accounts,
                     int64_t previous_close) noexcept {
    for (std::size_t index = 0; index < positions.size(); ++index) {
        const auto &position = positions[index];
        const bool ordered = !std::isnan(position.target);
        if (currencies[index] >= accounts || !std::isfinite(lots[index]) || lots[index] <= 0 ||
            retired[index] > 1 || !nonnegative(position.quantity) ||
            (ordered && (!nonnegative(position.target) || position.decision_at < 0 ||
                         position.decision_at > previous_close)) ||
            (!std::isnan(position.capacity) && !nonnegative(position.capacity)) ||
            (retired[index] != 0 &&
             (position.quantity != 0 || (ordered && position.target != 0)))) {
            return false;
        }
    }
    return true;
}

bool valid_accounts(std::span<const mt_account_v1> accounts) noexcept {
    return std::all_of(accounts.begin(), accounts.end(), [](const auto &account) {
        return nonnegative(account.cash) && nonnegative(account.costs) &&
               nonnegative(account.turnover) && nonnegative(account.receivable) &&
               (std::isnan(account.nav) || nonnegative(account.nav));
    });
}

bool valid_prices(std::span<const double> prices) noexcept {
    for (std::size_t offset = 0; offset < prices.size(); offset += price_width) {
        if (!price_or_missing(prices[offset]) || !price_or_missing(prices[offset + close_column]) ||
            (!std::isnan(prices[offset + volume_column]) &&
             !nonnegative(prices[offset + volume_column]))) {
            return false;
        }
    }
    return true;
}

float clipped(double value) noexcept {
    return static_cast<float>(std::clamp(value, -observation_limit, observation_limit));
}
} // namespace

extern "C" mt_layout_v1 mt_simulation_layout_v1() {
    return {1, sizeof(mt_position_v1), sizeof(mt_account_v1), sizeof(mt_trade_v1)};
}

extern "C" int mt_simulation_step_v1(
    uint32_t asset_count, uint32_t account_count, const uint32_t *currencies, const double *lots,
    const uint8_t *retired, const double *prices, const mt_position_v1 *previous_positions,
    const mt_account_v1 *previous_accounts, double cost_rate, double participation,
    int64_t previous_close, int64_t open_at, int64_t close_at, mt_position_v1 *next_positions,
    mt_account_v1 *next_accounts, mt_trade_v1 *trades, char *error, std::size_t error_capacity) {
    if (asset_count == 0 || asset_count > max_assets || account_count == 0 ||
        account_count > max_accounts || currencies == nullptr || lots == nullptr ||
        retired == nullptr || prices == nullptr || previous_positions == nullptr ||
        previous_accounts == nullptr || next_positions == nullptr || next_accounts == nullptr ||
        trades == nullptr || previous_positions == next_positions ||
        previous_accounts == next_accounts || !nonnegative(cost_rate) ||
        cost_rate > maximum_cost_rate || !std::isfinite(participation) || participation <= 0 ||
        participation > 1 || previous_close < 0 || open_at <= previous_close ||
        close_at <= open_at) {
        return failure(MT_SIM_INVALID_ARGUMENT,
                       "Dimensiones, tiempos o parámetros contables inválidos", error,
                       error_capacity);
    }
    const std::span currency_view{currencies, asset_count};
    const std::span lot_view{lots, asset_count};
    const std::span retired_view{retired, asset_count};
    const std::span quote_view{prices, static_cast<std::size_t>(asset_count) * price_width};
    const std::span origin_positions{previous_positions, asset_count};
    const std::span origin_accounts{previous_accounts, account_count};
    std::span positions{next_positions, asset_count};
    std::span accounts{next_accounts, account_count};
    std::span operations{trades, asset_count};
    if (!valid_positions(origin_positions, currency_view, lot_view, retired_view, account_count,
                         previous_close) ||
        !valid_accounts(origin_accounts) || !valid_prices(quote_view)) {
        return failure(MT_SIM_INVALID_STATE, "Precios o estado contable inválidos", error,
                       error_capacity);
    }
    std::copy(origin_positions.begin(), origin_positions.end(), positions.begin());
    std::copy(origin_accounts.begin(), origin_accounts.end(), accounts.begin());
    std::array<CashMovements, max_accounts> movement_storage{};
    std::array<AccurateSum, max_accounts> request_storage{};
    const std::span movements{movement_storage};
    const std::span requested{request_storage};
    for (std::size_t currency = 0; currency < accounts.size(); ++currency) {
        if (!movements[currency].add(accounts[currency].cash)) {
            return failure(MT_SIM_NUMERICAL_ERROR, "El efectivo no admite una suma finita", error,
                           error_capacity);
        }
    }
    for (std::size_t index = 0; index < positions.size(); ++index) {
        auto &position = positions[index];
        auto &trade = operations[index];
        trade = {0, unknown, 0, MT_ORDER_COMPLETE, 0};
        if (std::isnan(position.target)) {
            continue;
        }
        const double price = quote_view[index * price_width];
        if (std::isnan(price)) {
            trade.reason = MT_ORDER_MISSING_OPEN;
            continue;
        }
        if (std::isnan(position.capacity)) {
            trade.reason = MT_ORDER_UNKNOWN_LIQUIDITY;
            continue;
        }
        const double delta = position.target - position.quantity;
        const double limit = std::min(std::abs(delta), position.capacity);
        const double quantity = std::floor(limit / lot_view[index]) * lot_view[index];
        const auto currency = currency_view[index];
        if (!std::isfinite(quantity)) {
            return failure(MT_SIM_NUMERICAL_ERROR, "La cantidad no admite su lote", error,
                           error_capacity);
        }
        if (delta < 0) {
            if (quantity != 0 &&
                !fill(position, accounts[currency], trade, movements[currency],
                      {.quantity = -std::min(position.quantity, quantity), .price = price},
                      cost_rate)) {
                return failure(MT_SIM_NUMERICAL_ERROR, "La venta excede el rango numérico", error,
                               error_capacity);
            }
        } else if (quantity != 0) {
            trade.quantity = quantity;
            trade.price = price;
            if (!requested[currency].add(quantity * price * (1 + cost_rate))) {
                return failure(MT_SIM_NUMERICAL_ERROR, "La compra excede el rango numérico", error,
                               error_capacity);
            }
        }
    }
    std::array<double, max_accounts> scale_storage{};
    const std::span scales{scale_storage};
    for (std::size_t currency = 0; currency < accounts.size(); ++currency) {
        const double required = requested[currency].value();
        const double available = movements[currency].reconciled();
        if (!nonnegative(available)) {
            return failure(MT_SIM_NUMERICAL_ERROR, "Las ventas no conservan efectivo válido", error,
                           error_capacity);
        }
        scales[currency] = required != 0 ? std::min(1.0, available / required) : 0;
    }
    for (std::size_t index = 0; index < positions.size(); ++index) {
        auto &position = positions[index];
        auto &trade = operations[index];
        const auto currency = currency_view[index];
        if (trade.quantity > 0) {
            const double quantity =
                std::floor(trade.quantity * scales[currency] / lot_view[index]) * lot_view[index];
            const double price = trade.price;
            trade.quantity = 0;
            trade.price = unknown;
            if (quantity != 0 && !fill(position, accounts[currency], trade, movements[currency],
                                       {.quantity = quantity, .price = price}, cost_rate)) {
                return failure(MT_SIM_NUMERICAL_ERROR, "Las compras no conservan importes finitos",
                               error, error_capacity);
            }
        }
        if (!std::isnan(position.target)) {
            if (std::abs(position.target - position.quantity) < quantity_tolerance) {
                position.target = unknown;
                position.capacity = unknown;
                position.decision_at = 0;
                trade.reason = MT_ORDER_COMPLETE;
            } else {
                if (trade.reason == MT_ORDER_COMPLETE) {
                    trade.reason = MT_ORDER_RESTRICTED;
                }
                position.capacity = quote_view[index * price_width + volume_column] * participation;
            }
        }
    }
    std::array<AccurateSum, max_accounts> invested_storage{};
    std::array<bool, max_accounts> unvalued_storage{};
    const std::span invested{invested_storage};
    const std::span unvalued{unvalued_storage};
    for (std::size_t index = 0; index < positions.size(); ++index) {
        if (positions[index].quantity == 0) {
            continue;
        }
        const auto currency = currency_view[index];
        const double close = quote_view[index * price_width + close_column];
        if (std::isnan(close)) {
            unvalued[currency] = true;
        } else if (!invested[currency].add(positions[index].quantity * close)) {
            return failure(MT_SIM_NUMERICAL_ERROR, "La valoración excede el rango numérico", error,
                           error_capacity);
        }
    }
    for (std::size_t currency = 0; currency < accounts.size(); ++currency) {
        auto &account = accounts[currency];
        account.cash = movements[currency].reconciled();
        account.nav = unvalued[currency]
                          ? unknown
                          : account.cash + invested[currency].value() + account.receivable;
    }
    if (!valid_accounts(accounts)) {
        return failure(MT_SIM_NUMERICAL_ERROR, "La ejecución deja una cuenta fuera de rango", error,
                       error_capacity);
    }
    return failure(MT_SIM_OK, {}, error, error_capacity);
}

extern "C" int mt_simulation_observation_v1(uint32_t asset_count, const double *prices,
                                            const double *previous_prices, const double *scores,
                                            const uint8_t *retired, const mt_position_v1 *positions,
                                            double nav, double cash, double score_scale,
                                            float *observation, char *error,
                                            std::size_t error_capacity) {
    if (asset_count == 0 || asset_count > max_assets || prices == nullptr ||
        previous_prices == nullptr || scores == nullptr || retired == nullptr ||
        positions == nullptr || observation == nullptr || !std::isfinite(score_scale) ||
        score_scale <= 0 || !nonnegative(cash) || (!std::isnan(nav) && !nonnegative(nav))) {
        return failure(MT_SIM_INVALID_ARGUMENT, "Observación financiera inválida", error,
                       error_capacity);
    }
    const auto quote_count = static_cast<std::size_t>(asset_count) * price_width;
    const std::span current{prices, quote_count};
    const std::span previous{previous_prices, quote_count};
    const std::span signals{scores, asset_count};
    const std::span excluded{retired, asset_count};
    const std::span holdings{positions, asset_count};
    const std::span output{observation,
                           static_cast<std::size_t>(asset_count) * observation_width + 2};
    if (!valid_prices(current) || !valid_prices(previous)) {
        return failure(MT_SIM_INVALID_STATE, "La observación contiene precios inválidos", error,
                       error_capacity);
    }
    if (std::isnan(nav) || nav == 0) {
        std::fill(output.begin(), output.end(), 0);
        return failure(MT_SIM_OK, {}, error, error_capacity);
    }
    double invested = 0;
    for (std::size_t index = 0; index < holdings.size(); ++index) {
        const auto &held = holdings[index];
        if (!nonnegative(held.quantity) ||
            (!std::isnan(held.target) && !nonnegative(held.target)) || std::isinf(signals[index]) ||
            excluded[index] > 1) {
            return failure(MT_SIM_INVALID_STATE,
                           "La observación contiene posiciones o señales inválidas", error,
                           error_capacity);
        }
        const double close = current[index * price_width + close_column];
        const double before = previous[index * price_width + close_column];
        const double volume = current[index * price_width + volume_column];
        const bool known = !std::isnan(close);
        const double weight = known ? held.quantity * close / nav : 0;
        invested += weight;
        const auto offset = index * observation_width;
        output[offset] = clipped(std::isnan(signals[index]) ? 0 : signals[index] / score_scale);
        output[offset + 1] = clipped(weight);
        output[offset + 2] = clipped(known && !std::isnan(before) ? close / before - 1 : 0);
        output[offset + 3] =
            clipped(std::isnan(volume) ? 0 : std::log1p(volume) / volume_log_scale);
        output[offset + 4] = clipped(
            known && !std::isnan(held.target) ? (held.target - held.quantity) * close / nav : 0);
        output[offset + presence_column] =
            known && !std::isnan(signals[index]) && excluded[index] == 0 ? 1 : 0;
    }
    const auto offset = static_cast<std::size_t>(asset_count) * observation_width;
    const double cash_fraction = cash / nav;
    output[offset] = clipped(cash_fraction);
    output[offset + 1] = clipped(std::max(0.0, 1 - cash_fraction - invested));
    return failure(MT_SIM_OK, {}, error, error_capacity);
}
