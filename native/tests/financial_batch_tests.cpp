#include "mars_titan/financial_batch.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string_view>

namespace {
using namespace mars_titan::simulation;
constexpr std::size_t sessions = 8;
constexpr std::size_t environments = 16;
constexpr double initial_price = 10;
constexpr double volume = 1000;
constexpr uint8_t full_action = 5;
constexpr std::size_t digest_length = 64;
constexpr std::size_t base_observation_width = 8;
constexpr std::size_t price_width = 5;
constexpr std::size_t close_column = 3;

void require(bool condition, std::string_view reason) {
    if (!condition) {
        throw std::runtime_error(std::string(reason));
    }
}

template<class Function> void rejected(Function&& function) {
    try {
        std::forward<Function>(function)();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("Se aceptó una entrada inválida");
}

void same_state(const SessionSnapshot& left, const SessionSnapshot& right) {
    const auto equal_number = [](double a, double b) {
        return a == b || (std::isnan(a) && std::isnan(b));
    };
    require(left.source_sha256 == right.source_sha256 && left.parameters == right.parameters &&
                left.cursor == right.cursor && left.done == right.done &&
                left.retired == right.retired && left.applied_actions == right.applied_actions &&
                left.held_order == right.held_order && left.peak_nav == right.peak_nav &&
                left.max_drawdown == right.max_drawdown &&
                left.unfilled_order_observations == right.unfilled_order_observations,
            "Cambió un campo del estado confirmado");
    require(left.account.cash == right.account.cash && left.account.costs == right.account.costs &&
                left.account.turnover == right.account.turnover &&
                left.account.receivable == right.account.receivable &&
                equal_number(left.account.nav, right.account.nav), "Cambió la cuenta confirmada");
    require(left.positions.size() == right.positions.size() &&
                left.receivables.size() == right.receivables.size(), "Cambió el tamaño del estado");
    for (std::size_t i = 0; i < left.positions.size(); ++i) {
        require(left.positions[i].quantity == right.positions[i].quantity &&
                    equal_number(left.positions[i].target, right.positions[i].target) &&
                    equal_number(left.positions[i].capacity, right.positions[i].capacity) &&
                    left.positions[i].decision_at == right.positions[i].decision_at,
                "Cambió una posición u orden confirmada");
    }
    for (std::size_t i = 0; i < left.receivables.size(); ++i) {
        require(left.receivables[i].action == right.receivables[i].action &&
                    left.receivables[i].pay_at == right.receivables[i].pay_at &&
                    left.receivables[i].amount == right.receivables[i].amount,
                "Cambió un derecho de cobro confirmado");
    }
}

std::shared_ptr<MarketTape> tape() {
    auto result = std::make_shared<MarketTape>();
    result->assets = {"FIC0"};
    result->currency = "USD";
    result->domain = "synthetic";
    result->partition = "train";
    result->parent_id = "analytic";
    result->source_sha256.assign(digest_length, 'a');
    for (std::size_t at = 0; at < sessions; ++at) {
        const auto time = static_cast<int64_t>(at) * 2;
        result->open_times.push_back(time);
        result->close_times.push_back(time + 1);
        result->prediction_times.push_back(time + 1);
        const auto price = initial_price + static_cast<double>(at);
        result->prices.insert(result->prices.end(), {price, price, price, price, volume});
        result->scores.push_back(default_score_scale);
    }
    return result;
}

void parity_and_restore(std::size_t workers) {
    const auto data = tape();
    FinancialBatch batch(std::vector<BatchInput>(environments, {data, {}, {}}), workers);
    FinancialSession reference(data);
    std::vector<uint8_t> actions(environments, full_action);
    for (std::size_t at = 0; at + 1 < sessions; ++at) {
        const auto expected = reference.step(actions.front());
        const auto& got = batch.step(actions);
        const auto observation = reference.observation();
        require(batch.observation_width() == observation.size(), "Anchura distinta");
        for (std::size_t lane = 0; lane < environments; ++lane) {
            require(got.rewards[lane] == expected.reward &&
                        got.terminated[lane] == expected.terminated &&
                        got.truncated[lane] == expected.truncated &&
                        got.reward_valid[lane] == expected.reward_valid,
                    "La transición paralela difiere de la referencia");
            require(std::equal(observation.begin(), observation.end(),
                               batch.observations().begin() +
                                   static_cast<std::ptrdiff_t>(lane * observation.size())),
                    "El lote cambia la observación");
        }
        const auto saved = batch.snapshot();
        batch.restore(saved);
    }
    const auto final_observation = std::vector<float>(batch.observations().begin(),
                                                      batch.observations().end());
    rejected([&] { static_cast<void>(batch.step(actions)); });
    require(std::equal(final_observation.begin(), final_observation.end(),
                       batch.observations().begin()), "El fallo alteró el lote final");
    batch.reset(std::array<std::size_t, 1>{0});
    require(batch.snapshot().sessions[0].cursor == 0 &&
                batch.snapshot().sessions[1].cursor == sessions - 1,
            "El reset debe afectar solo a los entornos seleccionados");
}

void different_actions_stay_in_lane_order(std::size_t workers) {
    constexpr std::size_t uneven_batch_size = 17;
    constexpr std::size_t actions_count = 6;
    const auto data = tape();
    FinancialBatch batch(std::vector<BatchInput>(uneven_batch_size, {data, {}, {}}), workers);
    std::vector<FinancialSession> references;
    references.reserve(uneven_batch_size);
    for (std::size_t lane = 0; lane < uneven_batch_size; ++lane) {
        references.emplace_back(data);
    }
    std::vector<uint8_t> actions(uneven_batch_size);
    for (std::size_t at = 0; at + 1 < sessions; ++at) {
        for (std::size_t lane = 0; lane < uneven_batch_size; ++lane) {
            actions[lane] = static_cast<uint8_t>((lane + at) % actions_count);
        }
        const auto& result = batch.step(actions);
        const auto saved = batch.snapshot();
        for (std::size_t lane = 0; lane < uneven_batch_size; ++lane) {
            const auto expected = references[lane].step(actions[lane]);
            const auto observed = references[lane].observation();
            same_state(saved.sessions[lane], references[lane].snapshot());
            require(result.rewards[lane] == expected.reward &&
                        std::equal(observed.begin(), observed.end(),
                            batch.observations().begin() +
                            static_cast<std::ptrdiff_t>(lane * observed.size())),
                    "Se mezclaron acciones, recompensas u observaciones entre entornos");
        }
    }
}

void failed_lane_does_not_commit_any_lane() {
    auto broken = tape();
    broken->actions = {{"split-overflow", 0, CorporateKind::split, 4,
                         std::numeric_limits<double>::max(), std::nullopt, true}};
    const auto good = tape();
    FinancialBatch batch({{good, {}, {}}, {broken, {}, {}}}, 2);
    const auto initial = batch.snapshot();
    const std::array<uint8_t, 2> actions{full_action, full_action};
    static_cast<void>(batch.step(actions));
    const auto before = batch.snapshot();
    rejected([&] { static_cast<void>(batch.step(actions)); });
    const auto after = batch.snapshot();
    for (std::size_t lane = 0; lane < after.sessions.size(); ++lane) {
        same_state(after.sessions[lane], before.sessions[lane]);
    }
    auto corrupt = before;
    corrupt.sessions[0] = initial.sessions[0];
    corrupt.sessions[1].account.nav += 1;
    rejected([&] { batch.restore(corrupt); });
    require(batch.snapshot().sessions[0].cursor == before.sessions[0].cursor,
            "La recuperación corrupta debe ser atómica");
    // El mismo pool sigue siendo utilizable después del fallo de un trabajador.
    batch.reset(std::array<std::size_t, 2>{0, 1});
    static_cast<void>(batch.step(actions));
    require(batch.snapshot().sessions[0].cursor == 1, "El pool no se recuperó del error");
}

void mixed_endings_keep_final_observations() {
    auto missing = tape();
    missing->prices[2 * price_width + close_column] = std::numeric_limits<double>::quiet_NaN();
    auto ruined = tape();
    ruined->actions = {{"writeoff", 0, CorporateKind::writeoff, 4, 0, std::nullopt, true}};
    auto short_tape = tape();
    constexpr std::size_t short_sessions = 3;
    short_tape->open_times.resize(short_sessions);
    short_tape->close_times.resize(short_sessions);
    short_tape->prediction_times.resize(short_sessions);
    short_tape->prices.resize(short_sessions * price_width);
    short_tape->scores.resize(short_sessions);
    Parameters parameters;
    constexpr double exact_purchase_capital = 1100;
    parameters.capital = exact_purchase_capital;
    parameters.cost_bps = 0;
    parameters.participation = 1;
    FinancialBatch batch({{missing, parameters, {}}, {ruined, parameters, {}},
                          {short_tape, parameters, {}}, {tape(), parameters, {}}}, 4);
    const std::array<uint8_t, 4> buy{full_action, full_action, full_action, full_action};
    static_cast<void>(batch.step(buy));
    const auto& final = batch.step(std::array<uint8_t, 4>{});
    require(final.truncated == std::vector<uint8_t>({1, 0, 1, 0}) &&
                final.terminated == std::vector<uint8_t>({0, 1, 0, 0}) &&
                final.reward_valid == std::vector<uint8_t>({0, 1, 1, 1}),
            "La ausencia, ruina y truncamiento necesitan máscaras distintas");
    require(batch.snapshot().sessions[2].cursor == 2,
            "No se debe reiniciar automáticamente antes del bootstrap");
    batch.reset(std::array<std::size_t, 3>{0, 1, 2});
    static_cast<void>(batch.step(buy));
    require(batch.snapshot().sessions[3].cursor == short_sessions,
            "Un reinicio ajeno ha alterado el entorno que continuaba");
}

void context_is_causal_and_bounded() {
    const auto data = tape();
    ContextTape context;
    context.source_sha256.assign(digest_length, 'b');
    context.fields = {{"external_event", "standardized"}};
    context.values.resize(sessions);
    for (std::size_t at = 0; at < sessions; ++at) {
        context.values[at] = {static_cast<float>(at), true, data->close_times[at]};
    }
    FinancialBatch batch({{data, {}, context}}, 1);
    constexpr std::size_t base_width = base_observation_width;
    require(batch.observation_width() == base_width + 3,
            "El contexto necesita valor, máscara y antigüedad");
    require(batch.observations()[base_width + 1] == 1, "Falta la máscara externa");
    auto changed = context;
    changed.values.back().value += 1;
    FinancialBatch future({{data, {}, changed}}, 1);
    require(std::equal(batch.observations().begin(), batch.observations().end(),
                       future.observations().begin()), "El futuro altera el prefijo");
    context.values[0].available_at += 1;
    rejected([&] { FinancialBatch invalid({{data, {}, context}}, 1); });
    rejected([&] { FinancialBatch invalid({{data, {}, {}}}, 3); });
    rejected([&] { FinancialBatch invalid({{data, {}, {}}}, 1, 1); });
    rejected([&] { batch.reset(std::array<std::size_t, 2>{0, 0}); });
    auto snapshot = batch.snapshot();
    snapshot.context_sources[0].assign(digest_length, 'c');
    rejected([&] { batch.restore(snapshot); });
    context.values[0] = {};
    FinancialBatch absent({{data, {}, context}}, maximum_batch_workers);
    require(absent.workers() == 1 && absent.observations()[base_width + 1] == 0,
            "Una ausencia no es un cero observado y un entorno no necesita ocho trabajadores");
    context.values[0].value = 1;
    rejected([&] { FinancialBatch invalid({{data, {}, context}}, 1); });
    context.values[0] = {std::numeric_limits<float>::infinity(), true, 0};
    rejected([&] { FinancialBatch invalid({{data, {}, context}}, 1); });
    rejected([] { FinancialBatch empty({}); });
    auto other_partition = tape();
    other_partition->partition = "validation";
    rejected([&] { FinancialBatch mixed({{data, {}, {}}, {other_partition, {}, {}}}); });
    const std::array<uint8_t, 1> bad_action{6};
    rejected([&] { static_cast<void>(batch.step(bad_action)); });
    rejected([&] { static_cast<void>(batch.step({})); });
    rejected([&] { batch.reset(std::array<std::size_t, 1>{1}); });
    FinancialSession reference(data);
    std::array<float, 1> too_short{};
    rejected([&] { reference.observation_into(too_short); });
}
}

int main() {
    try {
        for (const auto workers : {std::size_t{1}, std::size_t{2}, std::size_t{4}, maximum_batch_workers}) {
            parity_and_restore(workers);
            different_actions_stay_in_lane_order(workers);
        }
        failed_lane_does_not_commit_any_lane();
        mixed_endings_keep_final_observations();
        context_is_causal_and_bounded();
        std::cout << "Lotes, contexto temporal y recuperación comprobados\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
