#include "mars_titan/simulation.h"

#include <array>
#include <cmath>
#include <cstdint>
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

bool close(double left, double right) { return std::abs(left - right) <= tolerance; }

int fail(const char *message) {
    std::cerr << message << '\n';
    return 1;
}
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
    std::cout << "Contabilidad, costes, monedas, datos ausentes y límites comprobados\n";
    return 0;
}
