#include "mars_titan/simulation.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>

namespace {
constexpr double unknown = std::numeric_limits<double>::quiet_NaN();
constexpr double tolerance = 1e-10;
constexpr std::size_t assets = 2;
constexpr std::size_t accounts_count = 2;
constexpr std::size_t quote_fields = 5;
constexpr std::size_t observation_fields = 6;
constexpr std::size_t error_capacity = 256;
constexpr double rate = 0.01;
constexpr double participation = 0.1;
constexpr double score_scale = 0.01;
constexpr double sale_price = 13;
constexpr double observation_cash = 800;
constexpr float expected_weight = 0.6F;
constexpr float expected_cash_fraction = 0.4F;
constexpr std::array<double, assets * quote_fields> first_prices{10, 12, 9,  12, 10000,
                                                                 20, 20, 20, 20, 10000};
constexpr std::array<mt_position_v1, assets> initial_positions{
    {{0, 100, 1000, 1}, {0, 100, 1000, 1}}};
constexpr std::array<mt_account_v1, accounts_count> initial_accounts{
    {{2000, 0, 0, 0, 2000}, {500, 0, 0, 0, 500}}};
constexpr std::array<double, assets> bought_quantities{100, 24};
constexpr std::array<mt_account_v1, accounts_count> bought_accounts{
    {{990, 10, 1000, 0, 2190}, {15.2, 4.8, 480, 0, 495.2}}};
constexpr mt_position_v1 sale_order{100, 0, 1000, 1};
constexpr mt_account_v1 sold_account{2277, 23, 2300, 0, 2277};
constexpr mt_position_v1 pending_order{100, 120, 1000, 1};
constexpr mt_position_v1 held_position{100, unknown, unknown, 0};
constexpr std::array<double, assets> scores{0.01, 0.02};
constexpr double limited_reference = 8;
constexpr double free_reference = 20;
constexpr double limit_band = 0.2;
constexpr double narrow_reference = 18.5;
constexpr double narrow_band = 0.05;

bool close(double left, double right) { return std::abs(left - right) <= tolerance; }

int fail(const char *message) {
    std::cerr << message << '\n';
    return 1;
}

// Los casos de redondeo y las rejillas de precios se leen mejor con sus valores literales.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
// Casos conocidos del redondeo decimal de Python y errores de la banda.
int price_limits_round_half_up_to_the_cent() {
    constexpr double band = 0.1;
    struct Case {
        double reference;
        double upper;
        double lower;
    };
    constexpr std::array cases{Case{10.15, 11.17, 9.14}, Case{10.05, 11.06, 9.05},
                               Case{2.675, 2.94, 2.41}, Case{0.004, 0.0, 0.0}};
    std::array<char, error_capacity> error{};
    for (const auto &[reference, upper, lower] : cases) {
        double high = 0;
        double low = 0;
        if (mt_simulation_price_limits_v1(reference, band, &high, &low, error.data(),
                                          error.size()) != MT_SIM_OK ||
            high != upper || low != lower) {
            return fail("Los límites no conservan el redondeo decimal de la referencia");
        }
    }
    double high = 0;
    double low = 0;
    if (mt_simulation_price_limits_v1(unknown, band, &high, &low, error.data(), error.size()) !=
            MT_SIM_OK ||
        !std::isnan(high) || !std::isnan(low) ||
        mt_simulation_price_limits_v1(10, 0, &high, &low, error.data(), error.size()) !=
            MT_SIM_OK ||
        !std::isnan(high) ||
        mt_simulation_price_limits_v1(10, 1, &high, &low, error.data(), error.size()) !=
            MT_SIM_INVALID_ARGUMENT ||
        mt_simulation_price_limits_v1(10, 1e-11, &high, &low, error.data(), error.size()) !=
            MT_SIM_INVALID_ARGUMENT) {
        return fail("Sin banda no hay límites y una banda no exacta se rechaza");
    }
    return 0;
}

// Una orden de un activo con reglas. Devuelve el motivo o -1 si el núcleo rechaza la llamada.
// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
int32_t limited_order(double price, double reference, double band, bool buy) {
    constexpr double held = 1000;
    constexpr double capacity = 10;
    constexpr double cash = 1e15;
    const uint32_t currency = 0;
    const double lot = 1;
    const uint8_t retired = 0;
    const std::array<double, quote_fields> quote{price, price, price, price, held};
    const mt_position_v1 position{held, buy ? held + capacity : 0, capacity, 1};
    const mt_account_v1 account{cash, 0, 0, 0, cash};
    const mt_rules_v1 rule{0, reference, band, 0, 0, 0, 0};
    mt_position_v1 next_position{};
    mt_account_v1 next_account{};
    mt_trade_v1 trade{};
    std::array<char, error_capacity> error{};
    if (mt_simulation_step_v2(1, 1, &currency, &lot, &rule, &retired, quote.data(), &position,
                              &account, rate, participation, 1, 2, 3, &next_position,
                              &next_account, &trade, error.data(), error.size()) != MT_SIM_OK) {
        return -1;
    }
    return trade.reason;
}

// El núcleo decide como el límite decimal exacto en la franja donde el producto binario duda,
// con referencias en céntimos y fuera de la rejilla y con bandas próximas a 1.
int limit_decisions_match_exact_limits() {
    constexpr int samples = 2000;
    constexpr int magnitudes = 10;
    constexpr int smallest_magnitude = -3;
    constexpr double spread = 9;
    constexpr double unit_interval = 0x1p-53;
    constexpr int mantissa_shift = 11;
    constexpr std::array bands{0.05, 0.1, 0.2, 0.3, 0.44, 0.6, 0.95, 0.9999999999};
    constexpr std::array offsets{0.0, 0.005, 0.0099, 0.01, 0.0101, 0.02, 0.5};
    uint64_t state = 0x9e3779b97f4a7c15U;
    std::array<char, error_capacity> error{};
    for (int sample = 0; sample < samples; ++sample) {
        state ^= state << 13U;
        state ^= state >> 7U;
        state ^= state << 17U;
        const double magnitude =
            std::pow(10.0, static_cast<double>(state % magnitudes) + smallest_magnitude);
        double reference =
            magnitude * (1 + static_cast<double>(state >> mantissa_shift) * unit_interval * spread);
        if (sample % 2 == 0 && reference >= 1) {
            reference = std::round(reference * 100) / 100;
        }
        for (const double band : bands) {
            double upper = 0;
            double lower = 0;
            if (mt_simulation_price_limits_v1(reference, band, &upper, &lower, error.data(),
                                              error.size()) != MT_SIM_OK) {
                return fail(error.data());
            }
            for (const double center : {upper, lower, reference * (1 + band),
                                        reference * (1 - band)}) {
                for (const double offset : offsets) {
                    for (const double price :
                         {center + offset, center - offset, std::nextafter(center, 0.0),
                          std::nextafter(center, HUGE_VAL)}) {
                        if (!(price > 0)) {
                            continue;
                        }
                        const auto buying = limited_order(price, reference, band, true);
                        const auto selling = limited_order(price, reference, band, false);
                        if ((buying == MT_ORDER_LIMIT_UP) != (price >= upper) ||
                            (selling == MT_ORDER_LIMIT_DOWN) != (price <= lower) ||
                            buying < 0 || selling < 0) {
                            std::cerr << reference << ' ' << band << ' ' << price << '\n';
                            return fail("La decisión del límite difiere del cálculo exacto");
                        }
                    }
                }
            }
        }
    }
    // Los importes fuera de 2^53 céntimos y las bandas sin decimal exacto se rechazan.
    constexpr double huge_reference = 1e14;
    constexpr double inexact_band = 0.30000000000000004;
    if (limited_order(1, huge_reference, 0.1, true) != -1 ||
        limited_order(1, 10, inexact_band, false) != -1) {
        return fail("El núcleo debe rechazar límites decimales no exactos");
    }
    return 0;
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
} // namespace

int main() {
    const auto layout = mt_simulation_layout_v1();
    if (layout.abi_version != 1 || layout.position_size != sizeof(mt_position_v1) ||
        layout.account_size != sizeof(mt_account_v1) || layout.trade_size != sizeof(mt_trade_v1)) {
        return fail("El contrato binario no coincide");
    }
    std::array<uint32_t, assets> currencies{0, 1};
    std::array<double, assets> lots{1, 1};
    std::array<uint8_t, assets> retired{0, 0};
    auto prices = first_prices;
    auto positions = initial_positions;
    auto accounts = initial_accounts;
    std::array<mt_position_v1, assets> next_positions{};
    std::array<mt_account_v1, accounts_count> next_accounts{};
    std::array<mt_trade_v1, assets> trades{};
    std::array<char, error_capacity> error{};
    const auto step = [&]() {
        return mt_simulation_step_v1(
            assets, accounts_count, currencies.data(), lots.data(), retired.data(), prices.data(),
            positions.data(), accounts.data(), rate, participation, 1, 2, 3, next_positions.data(),
            next_accounts.data(), trades.data(), error.data(), error.size());
    };
    if (step() != MT_SIM_OK) {
        return fail(error.data());
    }
    if (!close(next_positions[0].quantity, bought_quantities[0]) ||
        !close(next_accounts[0].cash, bought_accounts[0].cash) ||
        !close(next_accounts[0].nav, bought_accounts[0].nav) ||
        !close(next_accounts[0].costs, bought_accounts[0].costs) ||
        !close(next_positions[1].quantity, bought_quantities[1]) ||
        !close(next_accounts[1].cash, bought_accounts[1].cash) ||
        trades[1].reason != MT_ORDER_RESTRICTED) {
        return fail("Compras, costes o cuentas separados no concilian");
    }
    // Añadir una cuenta sin operaciones no cambia los movimientos de la primera.
    currencies.fill(0);
    if (step() != MT_SIM_OK) {
        return fail(error.data());
    }
    const auto padded_positions = next_positions;
    const auto padded_account = next_accounts[0];
    const auto padded_trades = trades;
    if (mt_simulation_step_v1(
            assets, 1, currencies.data(), lots.data(), retired.data(), prices.data(),
            positions.data(), accounts.data(), rate, participation, 1, 2, 3,
            next_positions.data(), next_accounts.data(), trades.data(), error.data(),
            error.size()) != MT_SIM_OK || next_accounts[0].cash != padded_account.cash ||
        next_accounts[0].nav != padded_account.nav ||
        next_accounts[0].costs != padded_account.costs ||
        next_accounts[0].turnover != padded_account.turnover ||
        next_positions[0].quantity != padded_positions[0].quantity ||
        next_positions[1].quantity != padded_positions[1].quantity ||
        trades[0].quantity != padded_trades[0].quantity ||
        trades[1].cost != padded_trades[1].cost) {
        return fail("La ruta de una cuenta difiere de la referencia con cuentas adicionales");
    }
    currencies = {0, 1};
    positions[0] = sale_order;
    accounts[0] = bought_accounts[0];
    prices[0] = sale_price;
    if (step() != MT_SIM_OK || !close(next_accounts[0].cash, sold_account.cash) ||
        !close(next_accounts[0].costs, sold_account.costs) ||
        !close(next_positions[0].quantity, 0)) {
        return fail("La venta no descuenta sus costes");
    }
    positions[0] = pending_order;
    prices[0] = unknown;
    prices[3] = unknown;
    if (step() != MT_SIM_OK || trades[0].reason != MT_ORDER_MISSING_OPEN ||
        !std::isnan(next_accounts[0].nav) ||
        !close(next_positions[0].quantity, held_position.quantity)) {
        return fail("Una cotización ausente no debe liquidar ni inventar valoración");
    }
    positions[0].decision_at = 4;
    if (step() != MT_SIM_INVALID_STATE || accounts[0].cash != bought_accounts[0].cash ||
        positions[0].quantity != held_position.quantity) {
        return fail("Una orden futura debe rechazarse sin cambiar el origen");
    }
    currencies[1] = accounts_count;
    if (step() == MT_SIM_OK) {
        return fail("Una cuenta fuera de rango no puede acceder a memoria");
    }
    std::array<float, assets * observation_fields + 2> observation{};
    positions[0] = held_position;
    prices = first_prices;
    if (mt_simulation_observation_v1(assets, prices.data(), prices.data(), scores.data(),
                                     retired.data(), positions.data(), initial_accounts[0].nav,
                                     observation_cash, score_scale, observation.data(),
                                     error.data(), error.size()) != MT_SIM_OK ||
        observation[0] != 1 || observation[1] != expected_weight ||
        observation[assets * observation_fields] != expected_cash_fraction) {
        return fail("La observación no conserva señal y exposición");
    }
    if (mt_simulation_rules_size_v1() != sizeof(mt_rules_v1) ||
        price_limits_round_half_up_to_the_cent() != 0 ||
        limit_decisions_match_exact_limits() != 0) {
        return fail("El contrato de reglas no coincide");
    }
    // Sin reglas, v2 reproduce v1 byte a byte.
    currencies = {0, 1};
    prices = first_prices;
    positions = initial_positions;
    accounts = initial_accounts;
    if (step() != MT_SIM_OK) {
        return fail(error.data());
    }
    const auto v1_positions = next_positions;
    const auto v1_accounts = next_accounts;
    const auto v1_trades = trades;
    const auto step_rules = [&](const mt_rules_v1 *rules) {
        return mt_simulation_step_v2(
            assets, accounts_count, currencies.data(), lots.data(), rules, retired.data(),
            prices.data(), positions.data(), accounts.data(), rate, participation, 1, 2, 3,
            next_positions.data(), next_accounts.data(), trades.data(), error.data(),
            error.size());
    };
    // La comparación de bytes es deliberada: v2 sin reglas debe reproducir v1 bit a bit.
    // NOLINTBEGIN(bugprone-suspicious-memory-comparison)
    if (step_rules(nullptr) != MT_SIM_OK ||
        std::memcmp(v1_positions.data(), next_positions.data(), sizeof(v1_positions)) != 0 ||
        std::memcmp(v1_accounts.data(), next_accounts.data(), sizeof(v1_accounts)) != 0 ||
        std::memcmp(v1_trades.data(), trades.data(), sizeof(v1_trades)) != 0) {
        return fail("v2 sin reglas no reproduce v1");
    }
    // NOLINTEND(bugprone-suspicious-memory-comparison)
    // La apertura de 10 supera el límite de 9,60 sobre un cierre previo de 8. La de 20 no
    // alcanza el de 24.
    std::array<mt_rules_v1, assets> rules{{{0, limited_reference, limit_band, 0, 0, 0, 0},
                                           {0, free_reference, limit_band, 0, 0, 0, 0}}};
    if (step_rules(rules.data()) != MT_SIM_OK || trades[0].reason != MT_ORDER_LIMIT_UP ||
        next_positions[0].quantity != 0 || trades[1].reason == MT_ORDER_LIMIT_UP) {
        return fail("Una compra en el límite superior no debe ejecutarse");
    }
    // Con una banda de 0,05 sobre 18,50 la apertura de 20 supera 19,43. Cada activo usa su banda
    // aunque el núcleo reutilice la lectura decimal de la anterior.
    rules[1] = {0, narrow_reference, narrow_band, 0, 0, 0, 0};
    if (step_rules(rules.data()) != MT_SIM_OK || trades[0].reason != MT_ORDER_LIMIT_UP ||
        trades[1].reason != MT_ORDER_LIMIT_UP) {
        return fail("Cada activo debe aplicar su propia banda");
    }
    rules[0].band = 1;
    if (step_rules(rules.data()) != MT_SIM_INVALID_ARGUMENT) {
        return fail("Una banda fuera de rango debe rechazarse");
    }
    std::cout << "Contabilidad, costes, monedas, datos ausentes y límites comprobados\n";
    return 0;
}
