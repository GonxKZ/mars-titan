#include "mars_titan/simulation.h"
#include "accurate_sum.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <string_view>
#include <system_error>
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
constexpr std::size_t rules_bytes = 48;
constexpr std::size_t shortest_characters = 32;
constexpr uint64_t decimal_base = 10;
constexpr int cent_decimals = 2;
constexpr int maximum_band_decimals = 10;
constexpr int maximum_dropped_digits = 30;
constexpr uint64_t exact_integer_limit = uint64_t{1} << binary64_precision;
constexpr double cents_per_unit = 100;
static_assert(std::numeric_limits<double>::is_iec559);
static_assert(std::numeric_limits<double>::digits == binary64_precision);
static_assert(sizeof(mt_position_v1) == position_bytes && sizeof(mt_account_v1) == account_bytes);
static_assert(sizeof(mt_trade_v1) == trade_bytes && sizeof(mt_rules_v1) == rules_bytes);
// Producto exacto de hasta 28 cifras decimales para los precios límite.
__extension__ using Wide = unsigned __int128;
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

bool rate_below_one(double value) noexcept { return nonnegative(value) && value < 1; }

bool valid_rules(std::span<const mt_rules_v1> rules) noexcept {
    return std::all_of(rules.begin(), rules.end(), [](const auto &rule) {
        return nonnegative(rule.minimum_order) && !std::isinf(rule.reference) &&
               rate_below_one(rule.band) && rate_below_one(rule.buy_tax) &&
               rate_below_one(rule.sell_tax) && rule.odd_lot_exit <= 1 && rule.reserved == 0;
    });
}

// Valor exacto digits * 10^exponent.
struct DecimalNumber {
    uint64_t digits;
    int exponent;
};

// La representación más corta que recupera el mismo double coincide con repr de Python.
bool shortest_decimal(double value, DecimalNumber &result) noexcept {
    std::array<char, shortest_characters> buffer{};
    const auto written = std::to_chars(buffer.data(), buffer.data() + buffer.size(), value,
                                       std::chars_format::scientific);
    if (written.ec != std::errc{}) {
        return false;
    }
    const std::string_view text(buffer.data(),
                                static_cast<std::size_t>(written.ptr - buffer.data()));
    const auto marker = text.find('e');
    if (marker == std::string_view::npos) {
        return false;
    }
    uint64_t digits = 0;
    int fraction = 0;
    bool decimals = false;
    for (const char character : text.substr(0, marker)) {
        if (character == '.') {
            decimals = true;
            continue;
        }
        if (character < '0' || character > '9') {
            return false;
        }
        digits = digits * decimal_base + static_cast<uint64_t>(character - '0');
        fraction += decimals ? 1 : 0;
    }
    auto exponent_text = text.substr(marker + 1);
    if (!exponent_text.empty() && exponent_text.front() == '+') {
        exponent_text.remove_prefix(1);
    }
    int exponent = 0;
    const auto *end = exponent_text.data() + exponent_text.size();
    const auto parsed = std::from_chars(exponent_text.data(), end, exponent);
    if (parsed.ec != std::errc{} || parsed.ptr != end) {
        return false;
    }
    result = {digits, exponent - fraction};
    return true;
}

// coefficient * 10^exponent redondeado a céntimos por la mitad hacia arriba, como
// Decimal.quantize(Decimal("0.01"), ROUND_HALF_UP) seguido de float().
bool rounded_cents(Wide coefficient, int exponent, double &result) noexcept {
    const int shift = exponent + cent_decimals;
    Wide cents = coefficient;
    if (shift >= 0) {
        for (int step = 0; step < shift; ++step) {
            if (cents >= exact_integer_limit) {
                return false;
            }
            cents *= decimal_base;
        }
    } else if (-shift > maximum_dropped_digits) {
        // El coeficiente tiene menos de 29 cifras y queda por debajo de medio céntimo.
        cents = 0;
    } else {
        Wide divisor = 1;
        for (int step = 0; step < -shift; ++step) {
            divisor *= decimal_base;
        }
        cents = coefficient / divisor;
        if (2 * (coefficient % divisor) >= divisor) {
            ++cents;
        }
    }
    if (cents >= exact_integer_limit) {
        return false;
    }
    // Un entero exacto entre 100 da el double más cercano al decimal de dos cifras.
    result = static_cast<double>(static_cast<uint64_t>(cents)) / cents_per_unit;
    return true;
}

struct DailyLimits {
    double upper = unknown;
    double lower = unknown;
};

// Mismo resultado que Instrument.limits: Decimal(repr(reference)) * (1 ± Decimal(repr(band))).
// Con 17 cifras de referencia y 10 decimales de banda el producto cabe en las 28 cifras del
// contexto decimal de Python, que así no redondea antes de cuantizar.
bool daily_limits(double reference, double band, DailyLimits &limits) noexcept {
    limits = {};
    if (band == 0 || std::isnan(reference) || reference <= 0) {
        return true;
    }
    DecimalNumber base{};
    DecimalNumber width{};
    if (!std::isfinite(reference) || !shortest_decimal(reference, base) ||
        !shortest_decimal(band, width) || width.exponent >= 0 ||
        -width.exponent > maximum_band_decimals) {
        return false;
    }
    uint64_t scale = 1;
    for (int step = 0; step < -width.exponent; ++step) {
        scale *= decimal_base;
    }
    if (width.digits >= scale) {
        return false;
    }
    const int exponent = base.exponent + width.exponent;
    return rounded_cents(Wide{base.digits} * (scale + width.digits), exponent, limits.upper) &&
           rounded_cents(Wide{base.digits} * (scale - width.digits), exponent, limits.lower);
}

// Mayor venta admitida que no supera la deseada, como Instrument.sellable.
double sellable(double wanted, double held, double lot, const mt_rules_v1 &rule) noexcept {
    if (rule.odd_lot_exit == 0) {
        return std::min(held, std::floor(wanted / lot) * lot);
    }
    if (wanted >= held) {
        return held;
    }
    double best = std::floor(wanted / lot) * lot;
    if (best < std::max(rule.minimum_order, lot)) {
        best = 0;
    }
    const double odd = lot > 1 ? std::fmod(held, lot) : 0;
    if (odd != 0 && wanted >= odd) {
        best = std::max(best, odd + std::floor((wanted - odd) / lot) * lot);
    }
    return best;
}
} // namespace

extern "C" mt_layout_v1 mt_simulation_layout_v1() {
    return {1, sizeof(mt_position_v1), sizeof(mt_account_v1), sizeof(mt_trade_v1)};
}

extern "C" uint32_t mt_simulation_rules_size_v1() { return sizeof(mt_rules_v1); }

extern "C" int mt_simulation_price_limits_v1(double reference, double band, double *upper,
                                             double *lower, char *error,
                                             std::size_t error_capacity) {
    DailyLimits limits{};
    if (upper == nullptr || lower == nullptr || !rate_below_one(band) ||
        std::isinf(reference) || !daily_limits(reference, band, limits)) {
        return failure(MT_SIM_INVALID_ARGUMENT,
                       "La referencia o la banda no admiten un límite decimal exacto", error,
                       error_capacity);
    }
    *upper = limits.upper;
    *lower = limits.lower;
    return failure(MT_SIM_OK, {}, error, error_capacity);
}

namespace {
// Las reglas son opcionales. Sin ellas cada operación conserva la aritmética de v1.
template<std::size_t AccountCapacity> int step_accounts(
    uint32_t asset_count, uint32_t account_count, const uint32_t *currencies, const double *lots,
    const mt_rules_v1 *rules, const uint8_t *retired, const double *prices,
    const mt_position_v1 *previous_positions, const mt_account_v1 *previous_accounts,
    double cost_rate, double participation, int64_t previous_close, int64_t open_at,
    int64_t close_at, mt_position_v1 *next_positions, mt_account_v1 *next_accounts,
    mt_trade_v1 *trades, char *error, std::size_t error_capacity) {
    // Una sola ejecución por sesión, posterior al cierre de decisión, sostiene T+1: lo comprado
    // en esta apertura solo puede venderse en otra apertura posterior al siguiente cierre.
    if (asset_count == 0 || asset_count > max_assets || account_count == 0 ||
        account_count > AccountCapacity || currencies == nullptr || lots == nullptr ||
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
    const auto rule_view = rules == nullptr ? std::span<const mt_rules_v1>{}
                                            : std::span{rules, asset_count};
    const std::span retired_view{retired, asset_count};
    const std::span quote_view{prices, static_cast<std::size_t>(asset_count) * price_width};
    const std::span origin_positions{previous_positions, asset_count};
    const std::span origin_accounts{previous_accounts, account_count};
    std::span positions{next_positions, asset_count};
    std::span accounts{next_accounts, account_count};
    std::span operations{trades, asset_count};
    if (!valid_rules(rule_view)) {
        return failure(MT_SIM_INVALID_ARGUMENT, "Las reglas de mercado no son válidas", error,
                       error_capacity);
    }
    if (!valid_positions(origin_positions, currency_view, lot_view, retired_view, account_count,
                         previous_close) ||
        !valid_accounts(origin_accounts) || !valid_prices(quote_view)) {
        return failure(MT_SIM_INVALID_STATE, "Precios o estado contable inválidos", error,
                       error_capacity);
    }
    std::copy(origin_positions.begin(), origin_positions.end(), positions.begin());
    std::copy(origin_accounts.begin(), origin_accounts.end(), accounts.begin());
    std::array<CashMovements, AccountCapacity> movement_storage{};
    std::array<AccurateSum, AccountCapacity> request_storage{};
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
        const mt_rules_v1 *rule = rule_view.empty() ? nullptr : &rule_view[index];
        if (rule != nullptr) {
            DailyLimits limits{};
            if (!daily_limits(rule->reference, rule->band, limits)) {
                return failure(MT_SIM_INVALID_ARGUMENT,
                               "La referencia o la banda no admiten un límite decimal exacto",
                               error, error_capacity);
            }
            // Una apertura en el límite no se trata como ejecutable en esa dirección.
            if (delta > 0 && price >= limits.upper) {
                trade.reason = MT_ORDER_LIMIT_UP;
                continue;
            }
            if (delta < 0 && price <= limits.lower) {
                trade.reason = MT_ORDER_LIMIT_DOWN;
                continue;
            }
        }
        const double limit = std::min(std::abs(delta), position.capacity);
        const double lot = lot_view[index];
        const double quantity = std::floor(limit / lot) * lot;
        const auto currency = currency_view[index];
        if (!std::isfinite(quantity)) {
            return failure(MT_SIM_NUMERICAL_ERROR, "La cantidad no admite su lote", error,
                           error_capacity);
        }
        if (delta < 0) {
            // Solo se vende lo que ya se tenía al cierre de decisión.
            const double sold = rule == nullptr ? std::min(position.quantity, quantity)
                                                : sellable(limit, position.quantity, lot, *rule);
            const double rate = rule == nullptr ? cost_rate : cost_rate + rule->sell_tax;
            if (sold != 0 && !fill(position, accounts[currency], trade, movements[currency],
                                   {.quantity = -sold, .price = price}, rate)) {
                return failure(MT_SIM_NUMERICAL_ERROR, "La venta excede el rango numérico", error,
                               error_capacity);
            }
        } else if (quantity != 0) {
            trade.quantity = quantity;
            trade.price = price;
            // El impuesto de compra se reserva al dimensionar, en el mismo orden que Python.
            const double reserved = rule == nullptr ? quantity * price * (1 + cost_rate)
                                                    : quantity * price *
                                                          (1 + cost_rate + rule->buy_tax);
            if (!requested[currency].add(reserved)) {
                return failure(MT_SIM_NUMERICAL_ERROR, "La compra excede el rango numérico", error,
                               error_capacity);
            }
        }
    }
    std::array<double, AccountCapacity> scale_storage{};
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
            const mt_rules_v1 *rule = rule_view.empty() ? nullptr : &rule_view[index];
            double quantity =
                std::floor(trade.quantity * scales[currency] / lot_view[index]) * lot_view[index];
            if (rule != nullptr && quantity < rule->minimum_order) {
                quantity = 0;
            }
            const double price = trade.price;
            const double rate = rule == nullptr ? cost_rate : cost_rate + rule->buy_tax;
            trade.quantity = 0;
            trade.price = unknown;
            if (quantity != 0 && !fill(position, accounts[currency], trade, movements[currency],
                                       {.quantity = quantity, .price = price}, rate)) {
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
    std::array<AccurateSum, AccountCapacity> invested_storage{};
    std::array<bool, AccountCapacity> unvalued_storage{};
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

int step(uint32_t asset_count, uint32_t account_count, const uint32_t *currencies,
         const double *lots, const mt_rules_v1 *rules, const uint8_t *retired, const double *prices,
         const mt_position_v1 *previous_positions, const mt_account_v1 *previous_accounts,
         double cost_rate, double participation, int64_t previous_close, int64_t open_at,
         int64_t close_at, mt_position_v1 *next_positions, mt_account_v1 *next_accounts,
         mt_trade_v1 *trades, char *error, std::size_t error_capacity) {
    // La cartera habitual tiene una moneda. Evitar inicializar parciales de 31 cuentas ajenas.
    if (account_count == 1) {
        return step_accounts<1>(asset_count, account_count, currencies, lots, rules, retired,
            prices, previous_positions, previous_accounts, cost_rate, participation,
            previous_close, open_at, close_at, next_positions, next_accounts, trades, error,
            error_capacity);
    }
    return step_accounts<max_accounts>(asset_count, account_count, currencies, lots, rules,
        retired, prices, previous_positions, previous_accounts, cost_rate, participation,
        previous_close, open_at, close_at, next_positions, next_accounts, trades, error,
        error_capacity);
}
} // namespace

extern "C" int mt_simulation_step_v1(
    uint32_t asset_count, uint32_t account_count, const uint32_t *currencies, const double *lots,
    const uint8_t *retired, const double *prices, const mt_position_v1 *previous_positions,
    const mt_account_v1 *previous_accounts, double cost_rate, double participation,
    int64_t previous_close, int64_t open_at, int64_t close_at, mt_position_v1 *next_positions,
    mt_account_v1 *next_accounts, mt_trade_v1 *trades, char *error, std::size_t error_capacity) {
    return step(asset_count, account_count, currencies, lots, nullptr, retired, prices,
                previous_positions, previous_accounts, cost_rate, participation, previous_close,
                open_at, close_at, next_positions, next_accounts, trades, error, error_capacity);
}

extern "C" int mt_simulation_step_v2(
    uint32_t asset_count, uint32_t account_count, const uint32_t *currencies, const double *lots,
    const mt_rules_v1 *rules, const uint8_t *retired, const double *prices,
    const mt_position_v1 *previous_positions, const mt_account_v1 *previous_accounts,
    double cost_rate, double participation, int64_t previous_close, int64_t open_at,
    int64_t close_at, mt_position_v1 *next_positions, mt_account_v1 *next_accounts,
    mt_trade_v1 *trades, char *error, std::size_t error_capacity) {
    return step(asset_count, account_count, currencies, lots, rules, retired, prices,
                previous_positions, previous_accounts, cost_rate, participation, previous_close,
                open_at, close_at, next_positions, next_accounts, trades, error, error_capacity);
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
