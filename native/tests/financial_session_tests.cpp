#include "mars_titan/financial_session.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using mars_titan::simulation::CorporateAction;
using mars_titan::simulation::CorporateKind;
using mars_titan::simulation::FinancialSession;
using mars_titan::simulation::MarketTape;
using mars_titan::simulation::Parameters;
using mars_titan::simulation::ReferencePolicy;
using mars_titan::simulation::SessionSnapshot;

constexpr std::size_t price_width = 5;
constexpr std::size_t volume_column = 4;
constexpr double unknown = std::numeric_limits<double>::quiet_NaN();
constexpr double price = 10;
constexpr double initial_capital = 1000;
constexpr double ordinary_volume = 1000;
constexpr double tolerance = 1e-10;
constexpr std::size_t digest_length = 64;
constexpr uint8_t full_exposure_action = 5;

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::runtime_error(std::string(message));
    }
}

void near(double actual, double expected, std::string_view message) {
    require(std::isfinite(actual) && std::abs(actual - expected) <= tolerance, message);
}

void near(std::optional<double> actual, double expected, std::string_view message) {
    if (!actual.has_value()) {
        throw std::runtime_error(std::string(message));
    }
    near(actual.value(), expected, message);
}

template <class Function>
void rejected(Function&& function, std::string_view message) {
    try {
        std::forward<Function>(function)();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error(std::string(message));
}

std::shared_ptr<MarketTape> tape(std::size_t sessions = 3,
                                 std::vector<std::string> assets = {"A"}) {
    auto result = std::make_shared<MarketTape>();
    result->assets = std::move(assets);
    result->currency = "USD";
    result->domain = "synthetic";
    result->partition = "train";
    result->parent_id = "referencia-de-prueba";
    result->source_sha256.assign(digest_length, 'a');
    for (std::size_t session = 0; session < sessions; ++session) {
        const auto opening = static_cast<int64_t>(session * 2);
        result->open_times.push_back(opening);
        result->close_times.push_back(opening + 1);
        result->prediction_times.push_back(opening + 1);
        for (std::size_t asset = 0; asset < result->assets.size(); ++asset) {
            result->prices.insert(result->prices.end(), {price, price, price, price,
                                                        ordinary_volume});
            result->scores.push_back(1);
        }
    }
    return result;
}

Parameters parameters() {
    Parameters result;
    result.capital = initial_capital;
    result.cost_bps = 0;
    result.participation = 1;
    return result;
}

bool same_number(double left, double right) {
    return left == right || (std::isnan(left) && std::isnan(right));
}

void same_state(const SessionSnapshot& left, const SessionSnapshot& right) {
    require(left.source_sha256 == right.source_sha256 && left.parameters == right.parameters &&
                left.cursor == right.cursor && left.done == right.done &&
                left.retired == right.retired && left.applied_actions == right.applied_actions &&
                left.held_order == right.held_order && left.peak_nav == right.peak_nav &&
                left.max_drawdown == right.max_drawdown &&
                left.unfilled_order_observations == right.unfilled_order_observations,
            "La recuperación cambia la identidad, curso o métricas");
    require(left.positions.size() == right.positions.size() &&
                left.receivables.size() == right.receivables.size(),
            "La recuperación cambia las dimensiones del estado");
    for (std::size_t index = 0; index < left.positions.size(); ++index) {
        const auto& first = left.positions[index];
        const auto& second = right.positions[index];
        require(first.quantity == second.quantity && same_number(first.target, second.target) &&
                    same_number(first.capacity, second.capacity) &&
                    first.decision_at == second.decision_at,
                "La recuperación cambia una posición u orden");
    }
    require(left.account.cash == right.account.cash && left.account.costs == right.account.costs &&
                left.account.turnover == right.account.turnover &&
                left.account.receivable == right.account.receivable &&
                same_number(left.account.nav, right.account.nav),
            "La recuperación cambia las cuentas");
    for (std::size_t index = 0; index < left.receivables.size(); ++index) {
        const auto& first = left.receivables[index];
        const auto& second = right.receivables[index];
        require(first.action == second.action && first.pay_at == second.pay_at &&
                    first.amount == second.amount,
                "La recuperación cambia un derecho de cobro");
    }
}

void purchases_and_sales_charge_both_sides() {
    constexpr double next_price = 12;
    constexpr double cost_bps = 100;
    constexpr double purchased = 99;
    constexpr double first_cash = 0.1;
    constexpr double first_cost = 9.9;
    constexpr double first_nav = 990.1;
    constexpr double final_nav = 1176.22;
    constexpr double total_cost = 21.78;
    constexpr double normalized_turnover = 2.178;
    constexpr double drawdown = 0.0099;
    constexpr double expected_return = 0.17622;
    auto data = tape();
    std::fill_n(data->prices.begin() + static_cast<std::ptrdiff_t>(2 * price_width),
                volume_column, next_price);
    auto options = parameters();
    options.cost_bps = cost_bps;
    FinancialSession session(data, options);
    const auto purchase = session.step(full_exposure_action);
    const auto bought = session.snapshot();
    near(bought.positions[0].quantity, purchased, "La compra debe reservar sus costes");
    near(bought.account.cash, first_cash, "El efectivo después de comprar no concilia");
    near(bought.account.costs, first_cost, "Falta el coste de compra");
    near(bought.account.nav, first_nav, "La valoración de la primera compra no concilia");
    require(purchase.reward_valid && !purchase.terminated && !purchase.truncated,
            "Una compra válida no debe terminar el episodio");
    const auto sale = session.step(1);
    const auto result = session.metrics();
    near(session.snapshot().account.cash, final_nav, "La venta debe descontar su coste");
    near(result.costs, total_cost, "Deben cobrarse ambos lados de la operación");
    near(result.turnover, normalized_turnover, "El caudal debe normalizarse por capital");
    near(result.max_drawdown, drawdown, "El drawdown debe conservar el mínimo anterior");
    near(result.net_return, expected_return, "El retorno debe descontar ambos costes");
    require(sale.truncated && !sale.terminated && result.completed && result.steps == 2,
            "El final de la cinta debe ser una truncación válida");
    rejected([&session] { static_cast<void>(session.step(0)); },
             "No se puede avanzar un episodio terminado");
}

void pending_orders_use_previous_volume() {
    constexpr double participation = 0.5;
    constexpr double following_volume = 6;
    constexpr double final_quantity = 6;
    auto data = tape(4);
    data->prices[volume_column] = 2;
    data->prices[price_width + volume_column] = following_volume;
    data->prices[2 * price_width] = unknown;
    data->prices[2 * price_width + volume_column] = price;
    auto options = parameters();
    options.participation = participation;
    FinancialSession session(data, options);
    static_cast<void>(session.step(full_exposure_action));
    near(session.snapshot().positions[0].quantity, 1, "Se debe usar el volumen previo");
    const auto absent = session.step(0);
    near(session.snapshot().positions[0].quantity, 1, "Sin apertura no se ejecuta la orden");
    require(absent.trades[0].reason == MT_ORDER_MISSING_OPEN,
            "La apertura ausente debe conservar su motivo");
    static_cast<void>(session.step(0));
    near(session.snapshot().positions[0].quantity, final_quantity,
         "La orden pendiente debe usar el volumen del último cierre");
}

void splits_and_dividends_preserve_payment_timing() {
    constexpr double split_price = 5;
    constexpr double dividend = 0.25;
    constexpr double dividend_total = 100;
    constexpr double target_nav = 1100;
    constexpr double purchased_after_payment = 210;
    constexpr double cash_after_close = 50;
    constexpr int64_t pay_at_open = 6;
    constexpr int64_t pay_at_close = 7;
    auto data = tape(4);
    for (std::size_t session = 2; session < 4; ++session) {
        std::fill_n(data->prices.begin() + static_cast<std::ptrdiff_t>(session * price_width),
                    volume_column, split_price);
    }
    data->actions = {
        CorporateAction{"split", 0, CorporateKind::split, 4, 2, std::nullopt, true},
        CorporateAction{"div-open", 0, CorporateKind::dividend, 4, dividend, pay_at_open, true},
        CorporateAction{"div-close", 0, CorporateKind::dividend, 4, dividend, pay_at_close, true}};
    FinancialSession session(data, parameters());
    static_cast<void>(session.step(full_exposure_action));
    static_cast<void>(session.step(1));
    const auto ex_date = session.snapshot();
    near(ex_date.account.cash, initial_capital, "El derecho no es efectivo en la fecha ex");
    near(ex_date.account.receivable, dividend_total, "El split precede a los dividendos");
    near(ex_date.account.nav, target_nav, "El patrimonio debe incluir los derechos");
    FinancialSession restored(data, parameters());
    restored.restore(ex_date);
    static_cast<void>(session.step(full_exposure_action));
    static_cast<void>(restored.step(full_exposure_action));
    same_state(session.snapshot(), restored.snapshot());
    near(session.snapshot().positions[0].quantity, purchased_after_payment,
         "El cobro al cierre no puede financiar la compra de apertura");
    near(session.snapshot().account.cash, cash_after_close, "Falta el cobro al cierre");
    require(session.snapshot().receivables.empty(), "Los dividendos pagados no deben repetirse");
}

void missing_close_and_writeoff_have_distinct_endings() {
    auto missing = tape();
    missing->prices[price_width + 3] = unknown;
    FinancialSession incomplete(missing, parameters());
    const auto absent = incomplete.step(full_exposure_action);
    require(!absent.reward_valid && absent.truncated && !absent.terminated &&
                !incomplete.metrics().net_return.has_value() &&
                incomplete.metrics().invalid_reason == "missing_close",
            "Un cierre ausente invalida la recompensa y trunca el episodio");
    auto writeoff = tape();
    writeoff->actions = {CorporateAction{"baja", 0, CorporateKind::writeoff, 4, 0,
                                         std::nullopt, true}};
    FinancialSession ruined(writeoff, parameters());
    static_cast<void>(ruined.step(full_exposure_action));
    const auto loss = ruined.step(0);
    require(loss.terminated && !loss.truncated && loss.reward_valid &&
                ruined.metrics().invalid_reason == "ruined",
            "La baja total debe terminar por ruina");
    near(ruined.metrics().net_return, -1, "La pérdida total debe ser del capital completo");
    const auto observation = ruined.observation();
    require(std::all_of(observation.begin(), observation.end(),
                        [](float value) { return value == 0; }),
            "La observación de ruina debe ser cero");
}

void corrupted_snapshots_are_atomic() {
    auto data = tape(4);
    FinancialSession session(data, parameters());
    static_cast<void>(session.step(full_exposure_action));
    const auto original = session.snapshot();
    auto wrong = original;
    wrong.account.nav += 1;
    rejected([&] { session.restore(wrong); }, "El NAV manipulado debe rechazarse");
    same_state(session.snapshot(), original);
    wrong = original;
    wrong.cursor = data->close_times.size();
    rejected([&] { session.restore(wrong); }, "El cursor fuera de la cinta debe rechazarse");
    wrong = original;
    wrong.done = true;
    rejected([&] { session.restore(wrong); }, "El final prematuro debe rechazarse");
    wrong = original;
    wrong.source_sha256.assign(digest_length, 'b');
    rejected([&] { session.restore(wrong); }, "El origen distinto debe rechazarse");
    same_state(session.snapshot(), original);
}

void failed_corporate_action_keeps_the_confirmed_state() {
    auto data = tape();
    data->actions = {CorporateAction{"split-extremo", 0, CorporateKind::split, 4,
                                    std::numeric_limits<double>::max(), std::nullopt, true}};
    FinancialSession session(data, parameters());
    static_cast<void>(session.step(full_exposure_action));
    const auto original = session.snapshot();
    rejected([&session] { static_cast<void>(session.step(0)); },
             "El split que desborda debe rechazarse");
    same_state(session.snapshot(), original);
}

void normalized_turnover_overflow_keeps_the_confirmed_state() {
    constexpr double tiny_capital = 1e-300;
    constexpr double large_price = 1e8;
    auto data = tape(4);
    std::fill_n(data->prices.begin(), volume_column, tiny_capital);
    data->prices[price_width] = tiny_capital;
    for (std::size_t session = 1; session < 4; ++session) {
        data->prices[session * price_width + 3] = large_price;
        if (session > 1) {
            data->prices[session * price_width] = large_price;
        }
    }
    auto options = parameters();
    options.capital = tiny_capital;
    FinancialSession session(data, options);
    static_cast<void>(session.step(full_exposure_action));
    static_cast<void>(session.step(1));
    const auto original = session.snapshot();
    rejected([&session] { static_cast<void>(session.step(full_exposure_action)); },
             "El turnover normalizado no puede publicarse como infinito");
    same_state(session.snapshot(), original);
}

void repeated_writeoff_keeps_the_confirmed_state() {
    constexpr int64_t later_open = 6;
    auto data = tape(4);
    data->actions = {
        CorporateAction{"primera-baja", 0, CorporateKind::writeoff, 4, 0, std::nullopt, true},
        CorporateAction{"segunda-baja", 0, CorporateKind::writeoff, later_open, 0,
                        std::nullopt, true}};
    FinancialSession session(data, parameters());
    static_cast<void>(session.step(3));
    static_cast<void>(session.step(0));
    const auto original = session.snapshot();
    rejected([&session] { static_cast<void>(session.step(0)); },
             "Un activo retirado no puede darse de baja una segunda vez");
    same_state(session.snapshot(), original);
    auto forged = original;
    forged.cursor = 3;
    forged.done = true;
    forged.applied_actions[1] = 1;
    rejected([&] { session.restore(forged); },
             "La recuperación no puede confirmar una segunda baja imposible");
    same_state(session.snapshot(), original);
}

void ties_follow_asset_identity_and_bad_tapes_fail() {
    auto reversed = tape(2, {"B", "A"});
    auto ordered = tape(2, {"A", "B"});
    FinancialSession first(reversed, parameters());
    FinancialSession second(ordered, parameters());
    static_cast<void>(first.step(full_exposure_action));
    static_cast<void>(second.step(full_exposure_action));
    near(first.snapshot().positions[1].quantity, initial_capital / price,
         "El desempate debe elegir A aunque cambie el orden de activos");
    near(second.snapshot().positions[0].quantity, initial_capital / price,
         "La permutación debe conservar la cantidad de A");
    near(first.snapshot().account.nav, second.snapshot().account.nav,
         "La permutación no debe alterar el patrimonio");
    reversed->assets[1] = "B";
    rejected([&] { reversed->validate(); }, "Los activos duplicados deben rechazarse");
    auto future = tape();
    future->prediction_times[1] = future->close_times[1] + 1;
    rejected([&] { future->validate(); }, "La predicción futura debe rechazarse");
    auto duplicate = tape();
    const CorporateAction split{"repetida", 0, CorporateKind::split, 2, 2, std::nullopt, true};
    duplicate->actions = {split, split};
    rejected([&] { duplicate->validate(); }, "Las acciones duplicadas deben rechazarse");
    require(mars_titan::simulation::policy_action(ReferencePolicy::hold_initial, 0) ==
                full_exposure_action &&
                mars_titan::simulation::policy_action(ReferencePolicy::hold_initial, 1) == 0 &&
                mars_titan::simulation::policy_action(
                    mars_titan::simulation::parse_policy("rebalance_50"), 0) == 3,
            "Las referencias deben conservar exposición y primer paso");
}
}

int main() {
    try {
        purchases_and_sales_charge_both_sides();
        pending_orders_use_previous_volume();
        splits_and_dividends_preserve_payment_timing();
        missing_close_and_writeoff_have_distinct_endings();
        corrupted_snapshots_are_atomic();
        failed_corporate_action_keeps_the_confirmed_state();
        normalized_turnover_overflow_keeps_the_confirmed_state();
        repeated_writeoff_keeps_the_confirmed_state();
        ties_follow_asset_identity_and_bad_tapes_fail();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Sesiones, contabilidad, eventos y recuperación comprobados\n";
    return 0;
}
