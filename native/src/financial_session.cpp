#include "mars_titan/financial_session.hpp"

#include "accurate_sum.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

namespace mars_titan::simulation {
namespace {
constexpr std::size_t price_width = 5;
constexpr std::size_t close_column = 3;
constexpr std::size_t volume_column = 4;
constexpr std::size_t observation_width = 6;
constexpr std::size_t maximum_asset_name = 96;
constexpr std::size_t maximum_action_id = 128;
constexpr std::size_t maximum_parent_id = 256;
constexpr std::size_t digest_length = 64;
constexpr std::size_t error_capacity = 512;
constexpr std::size_t quartile_denominator = 4;
constexpr double maximum_target = 1e12;
constexpr double quantity_tolerance = 1e-12;
constexpr double maximum_cost_bps = 1000;
constexpr double basis_point_denominator = 10'000;
constexpr double quarter = 0.25;
constexpr double half = 0.5;
constexpr double three_quarters = 0.75;
constexpr double unknown = std::numeric_limits<double>::quiet_NaN();
constexpr int64_t validation_start = 1'672'531'200'000'000;
constexpr int64_t final_test_start = 1'704'067'200'000'000;
constexpr std::array<double, 6> exposures{0, 0, quarter, half, three_quarters, 1};
static_assert(std::is_nothrow_swappable_v<SessionSnapshot>);

bool nonnegative(double value) noexcept {
    return std::isfinite(value) && value >= 0;
}

bool same_number(double left, double right) noexcept {
    return left == right || (std::isnan(left) && std::isnan(right));
}

bool valid_digest(std::string_view value) {
    return value.size() == digest_length &&
           std::all_of(value.begin(), value.end(), [](char character) {
               return (character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f');
           });
}

void validate_parameters(const Parameters& parameters) {
    if (!nonnegative(parameters.capital) || parameters.capital == 0 ||
        !nonnegative(parameters.cost_bps) || parameters.cost_bps > maximum_cost_bps ||
        !std::isfinite(parameters.participation) || parameters.participation <= 0 ||
        parameters.participation > 1 || !std::isfinite(parameters.score_scale) ||
        parameters.score_scale <= 0 || !std::isfinite(parameters.ruin_penalty) ||
        parameters.ruin_penalty >= 0) {
        throw std::invalid_argument(
            "Los parámetros financieros no son finitos o están fuera de rango");
    }
}

void checked_sum(AccurateSum& total, double value) {
    if (!total.add(value)) {
        throw std::overflow_error("El importe agregado excede el rango numérico");
    }
}

void check_amount(double value) {
    if (!nonnegative(value)) {
        throw std::overflow_error("La operación produce una cantidad negativa o no finita");
    }
}

void check_native(int status, std::span<const char> error) {
    if (status != MT_SIM_OK) {
        const auto end = std::find(error.begin(), error.end(), '\0');
        throw std::runtime_error(std::string(error.begin(), end));
    }
}

double pending_value(const SessionSnapshot& state) {
    AccurateSum total;
    for (const auto& entry : state.receivables) {
        checked_sum(total, entry.amount);
    }
    return total.value();
}

double valuation(const SessionSnapshot& state, std::span<const double> prices) {
    AccurateSum total;
    for (std::size_t asset = 0; asset < state.positions.size(); ++asset) {
        const double quantity = state.positions[asset].quantity;
        if (quantity == 0) {
            continue;
        }
        const double close = prices[asset * price_width + close_column];
        if (std::isnan(close)) {
            return unknown;
        }
        checked_sum(total, quantity * close);
    }
    const double nav = state.account.cash + total.value() + state.account.receivable;
    check_amount(nav);
    return nav;
}

void settle(SessionSnapshot& state, int64_t at) {
    std::erase_if(state.receivables, [&state, at](const Receivable& entry) {
        if (entry.pay_at > at) {
            return false;
        }
        state.account.cash += entry.amount;
        check_amount(state.account.cash);
        return true;
    });
    state.account.receivable = pending_value(state);
}

void apply_actions(SessionSnapshot& state, const MarketTape& tape,
                   std::span<const std::size_t> actions) {
    for (const auto index : actions) {
        const auto& action = tape.actions[index];
        if (state.applied_actions[index] != 0) {
            throw std::invalid_argument("La acción corporativa ya se había aplicado");
        }
        auto& position = state.positions[action.asset];
        switch (action.kind) {
        case CorporateKind::split:
            position.quantity *= action.value;
            check_amount(position.quantity);
            if (!std::isnan(position.target)) {
                position.target *= action.value;
                check_amount(position.target);
                if (!std::isnan(position.capacity)) {
                    position.capacity *= action.value;
                    check_amount(position.capacity);
                }
            }
            break;
        case CorporateKind::dividend:
            if (!action.pay_at.has_value()) {
                throw std::invalid_argument("El dividendo no declara una fecha de pago");
            }
            if (position.quantity != 0) {
                const double amount = position.quantity * action.value;
                check_amount(amount);
                state.receivables.push_back({index, action.pay_at.value(), amount});
            }
            break;
        case CorporateKind::writeoff:
            if (state.retired[action.asset] != 0) {
                throw std::invalid_argument("El activo ya se había dado de baja");
            }
            position = {0, unknown, unknown, 0};
            state.retired[action.asset] = 1;
            break;
        }
        state.applied_actions[index] = 1;
    }
    state.account.receivable = pending_value(state);
}

void submit(SessionSnapshot& state, const MarketTape& tape, const Parameters& parameters,
            uint8_t action, std::vector<std::size_t>& ranking) {
    if (action == 0) {
        return;
    }
    const auto prices = tape.frame(state.cursor);
    const auto scores = tape.predictions(state.cursor);
    ranking.clear();
    for (std::size_t asset = 0; asset < state.positions.size(); ++asset) {
        auto& position = state.positions[asset];
        position.target = position.quantity > 0 ? 0 : unknown;
        position.capacity = unknown;
        position.decision_at = 0;
        if (std::isfinite(scores[asset]) && scores[asset] > 0 &&
            std::isfinite(prices[asset * price_width + close_column]) &&
            state.retired[asset] == 0) {
            ranking.push_back(asset);
        }
    }
    std::sort(ranking.begin(), ranking.end(), [&tape, scores](std::size_t left, std::size_t right) {
        return scores[left] == scores[right] ? tape.assets[left] < tape.assets[right]
                                            : scores[left] > scores[right];
    });
    const auto selected = (ranking.size() + quartile_denominator - 1) / quartile_denominator;
    const auto levels = std::span{exposures};
    for (std::size_t index = 0; index < selected; ++index) {
        const auto asset = ranking[index];
        const double target = state.account.nav * levels[action] / static_cast<double>(selected) /
                              prices[asset * price_width + close_column];
        if (!nonnegative(target) || target > maximum_target) {
            throw std::invalid_argument("La cantidad objetivo excede el presupuesto");
        }
        state.positions[asset].target = target;
    }
    for (std::size_t asset = 0; asset < state.positions.size(); ++asset) {
        auto& position = state.positions[asset];
        if (std::isnan(position.target)) {
            continue;
        }
        if (std::abs(position.target - position.quantity) < quantity_tolerance) {
            position.target = unknown;
            continue;
        }
        position.capacity = prices[asset * price_width + volume_column] * parameters.participation;
        position.decision_at = tape.close_times[state.cursor];
    }
}

void validate_snapshot(const SessionSnapshot& state, const MarketTape& tape,
                       const Parameters& parameters) {
    const auto assets = tape.assets.size();
    const auto sessions = tape.close_times.size();
    if (state.source_sha256 != tape.source_sha256 || state.parameters != parameters ||
        state.cursor >= sessions || state.positions.size() != assets ||
        state.retired.size() != assets || state.applied_actions.size() != tape.actions.size() ||
        state.receivables.size() > tape.actions.size() || state.held_order.size() > assets ||
        state.unfilled_order_observations > assets * state.cursor ||
        !nonnegative(state.account.cash) || !nonnegative(state.account.costs) ||
        !nonnegative(state.account.turnover) || !nonnegative(state.account.receivable) ||
        !std::isfinite(state.account.turnover / parameters.capital) ||
        (!std::isnan(state.account.nav) &&
         (!nonnegative(state.account.nav) ||
          !std::isfinite(state.account.nav / parameters.capital - 1))) ||
        !nonnegative(state.peak_nav) || state.peak_nav < parameters.capital ||
        !nonnegative(state.max_drawdown) || state.max_drawdown > 1) {
        throw std::invalid_argument("El checkpoint no conserva origen, parámetros o dimensiones");
    }
    const auto prices = tape.frame(state.cursor);
    const auto close = tape.close_times[state.cursor];
    std::vector<uint8_t> expected_retired(assets, 0);
    for (std::size_t index = 0; index < tape.actions.size(); ++index) {
        const auto& action = tape.actions[index];
        const bool occurred = action.effective_at <= tape.open_times[state.cursor];
        if (state.applied_actions[index] != static_cast<uint8_t>(occurred)) {
            throw std::invalid_argument("Las acciones aplicadas no corresponden al cursor");
        }
        if (occurred && action.kind == CorporateKind::writeoff) {
            if (expected_retired[action.asset] != 0) {
                throw std::invalid_argument(
                    "El checkpoint aplica una segunda baja del mismo activo");
            }
            expected_retired[action.asset] = 1;
        }
    }
    if (state.retired != expected_retired) {
        throw std::invalid_argument("Las bajas no corresponden a las acciones confirmadas");
    }
    std::vector<uint8_t> held(assets, 0);
    for (const auto asset : state.held_order) {
        if (asset >= assets || held[asset] != 0 || state.positions[asset].quantity <= 0) {
            throw std::invalid_argument("El orden de posiciones contiene huecos o duplicados");
        }
        held[asset] = 1;
    }
    for (std::size_t asset = 0; asset < assets; ++asset) {
        const auto& position = state.positions[asset];
        const bool ordered = !std::isnan(position.target);
        const double capacity =
            prices[asset * price_width + volume_column] * parameters.participation;
        if (!nonnegative(position.quantity) ||
            (position.quantity > 0) != (held[asset] != 0) ||
            (state.retired[asset] != 0 && (position.quantity != 0 || ordered)) ||
            (ordered && (!nonnegative(position.target) || position.decision_at < 0 ||
                         position.decision_at > close ||
                         !std::binary_search(tape.close_times.begin(), tape.close_times.end(),
                                             position.decision_at) ||
                         !same_number(position.capacity, capacity) ||
                         std::abs(position.target - position.quantity) < quantity_tolerance)) ||
            (!ordered && (!std::isnan(position.capacity) || position.decision_at != 0))) {
            throw std::invalid_argument("La posición o la orden no corresponde al último cierre");
        }
    }
    std::vector<uint8_t> pending(tape.actions.size(), 0);
    for (const auto& entry : state.receivables) {
        if (entry.action >= tape.actions.size() || pending[entry.action] != 0 ||
            !nonnegative(entry.amount) || entry.pay_at <= close) {
            throw std::invalid_argument("El derecho de cobro no pertenece al estado confirmado");
        }
        const auto& action = tape.actions[entry.action];
        if (action.kind != CorporateKind::dividend || state.applied_actions[entry.action] == 0 ||
            action.pay_at != entry.pay_at) {
            throw std::invalid_argument("El derecho de cobro no corresponde a su dividendo");
        }
        pending[entry.action] = 1;
    }
    if (pending_value(state) != state.account.receivable ||
        !same_number(valuation(state, prices), state.account.nav)) {
        throw std::invalid_argument(
            "El patrimonio no concilia con efectivo, posiciones y derechos");
    }
    const bool missing = std::isnan(state.account.nav);
    if (state.done != (missing || state.account.nav == 0 || state.cursor + 1 == sessions) ||
        (!missing && (state.peak_nav < state.account.nav ||
                      state.max_drawdown < 1 - state.account.nav / state.peak_nav))) {
        throw std::invalid_argument("El final o el drawdown no corresponden al patrimonio");
    }
    if (state.cursor == 0 &&
        (state.account.cash != parameters.capital || state.account.nav != parameters.capital ||
         state.account.costs != 0 || state.account.turnover != 0 || state.account.receivable != 0 ||
         !state.held_order.empty() || !state.receivables.empty() || state.max_drawdown != 0 ||
         state.peak_nav != parameters.capital ||
         std::any_of(state.positions.begin(), state.positions.end(), [](const auto& position) {
             return position.quantity != 0 || !std::isnan(position.target);
         }))) {
        throw std::invalid_argument("El estado inicial contiene operaciones no confirmadas");
    }
}
}

void MarketTape::validate() const {
    const auto count = assets.size();
    const auto sessions = close_times.size();
    if (count == 0 || count > maximum_assets || sessions < 2 || sessions > maximum_sessions ||
        count * sessions > maximum_rows || prices.size() != count * sessions * price_width ||
        scores.size() != count * sessions || open_times.size() != sessions ||
        prediction_times.size() != sessions || actions.size() > maximum_actions ||
        currency.size() != 3 ||
        !std::all_of(currency.begin(), currency.end(), [](char value) {
            return value >= 'A' && value <= 'Z';
        }) ||
        (domain != "synthetic" && domain != "real") ||
        (partition != "train" && partition != "validation") || !valid_digest(source_sha256) ||
        parent_id.empty() || parent_id.size() > maximum_parent_id ||
        (domain == "real" && !historical_audit_verified)) {
        throw std::invalid_argument("La cinta no conserva dimensiones, origen y periodo admitidos");
    }
    std::unordered_set<std::string_view> asset_names;
    for (const auto& asset : assets) {
        if (asset.empty() || asset.size() > maximum_asset_name ||
            !asset_names.insert(asset).second) {
            throw std::invalid_argument("Los activos de la cinta deben ser únicos y acotados");
        }
    }
    const int64_t low = partition == "train" ? 0 : validation_start;
    const int64_t high = partition == "train" ? validation_start : final_test_start;
    for (std::size_t session = 0; session < sessions; ++session) {
        if (open_times[session] < 0 || open_times[session] >= close_times[session] ||
            prediction_times[session] < 0 || prediction_times[session] > close_times[session] ||
            (session != 0 && open_times[session] <= close_times[session - 1]) ||
            (domain == "real" && (close_times[session] < low || close_times[session] >= high))) {
            throw std::invalid_argument(
                "El calendario mezcla decisiones, ejecución o el test cerrado");
        }
    }
    for (std::size_t index = 0; index < prices.size(); ++index) {
        const double value = prices[index];
        if (!std::isnan(value) &&
            (!std::isfinite(value) ||
             (index % price_width == volume_column ? value < 0 : value <= 0))) {
            throw std::invalid_argument(
                "Los precios y volúmenes deben ser válidos o estar ausentes");
        }
    }
    if (std::any_of(scores.begin(), scores.end(), [](double value) { return std::isinf(value); })) {
        throw std::invalid_argument("Las predicciones no pueden contener infinitos");
    }
    std::unordered_set<std::string_view> ids;
    std::vector<std::size_t> per_session(sessions, 0);
    for (const auto& action : actions) {
        const auto moment =
            std::lower_bound(open_times.begin(), open_times.end(), action.effective_at);
        if (action.id.empty() || action.id.size() > maximum_action_id ||
            !ids.insert(action.id).second || action.asset >= count || !action.verified ||
            !nonnegative(action.value) || moment == open_times.end() ||
            *moment != action.effective_at) {
            throw std::invalid_argument(
                "La acción corporativa está duplicada o no está acreditada");
        }
        switch (action.kind) {
        case CorporateKind::split:
            if (action.value == 0) {
                throw std::invalid_argument("El split debe tener una proporción positiva");
            }
            break;
        case CorporateKind::dividend:
            if (!action.pay_at.has_value() || action.pay_at.value() < action.effective_at) {
                throw std::invalid_argument(
                    "El dividendo debe declarar un pago posterior a la fecha ex");
            }
            break;
        case CorporateKind::writeoff:
            if (action.value != 0) {
                throw std::invalid_argument("La baja sin recuperación debe tener valor cero");
            }
            break;
        default:
            throw std::invalid_argument("La clase de acción corporativa no está admitida");
        }
        const auto session = static_cast<std::size_t>(moment - open_times.begin());
        if (++per_session[session] > maximum_assets) {
            throw std::invalid_argument("La sesión excede el presupuesto de acciones corporativas");
        }
    }
}

std::span<const double> MarketTape::frame(std::size_t session) const & {
    if (assets.empty() || assets.size() > maximum_assets || close_times.size() > maximum_sessions ||
        session >= close_times.size() ||
        prices.size() != close_times.size() * assets.size() * price_width) {
        throw std::out_of_range("La sesión no tiene un bloque de precios completo");
    }
    const auto width = assets.size() * price_width;
    return std::span{prices}.subspan(session * width, width);
}

std::span<const double> MarketTape::predictions(std::size_t session) const & {
    if (assets.empty() || assets.size() > maximum_assets || close_times.size() > maximum_sessions ||
        session >= close_times.size() || scores.size() != close_times.size() * assets.size()) {
        throw std::out_of_range("La sesión no tiene un bloque de predicciones completo");
    }
    return std::span{scores}.subspan(session * assets.size(), assets.size());
}

FinancialSession::FinancialSession(std::shared_ptr<const MarketTape> tape, Parameters parameters)
    : tape_(std::move(tape)), parameters_(parameters) {
    if (!tape_) {
        throw std::invalid_argument("La sesión necesita una cinta de mercado");
    }
    tape_->validate();
    validate_parameters(parameters_);
    const auto count = tape_->assets.size();
    state_.source_sha256 = tape_->source_sha256;
    state_.parameters = parameters_;
    state_.positions.assign(count, {0, unknown, unknown, 0});
    state_.account = {parameters_.capital, 0, 0, 0, parameters_.capital};
    state_.retired.assign(count, 0);
    state_.applied_actions.assign(tape_->actions.size(), 0);
    state_.peak_nav = parameters_.capital;
    state_.held_order.reserve(count);
    state_.receivables.reserve(tape_->actions.size());
    currencies_.assign(count, 0);
    lots_.assign(count, 1);
    next_positions_.resize(count);
    trades_.resize(count);
    ranking_.reserve(count);
    actions_by_session_.resize(tape_->close_times.size());
    for (std::size_t index = 0; index < tape_->actions.size(); ++index) {
        const auto moment = std::lower_bound(tape_->open_times.begin(), tape_->open_times.end(),
                                             tape_->actions[index].effective_at);
        const auto session = static_cast<std::size_t>(moment - tape_->open_times.begin());
        actions_by_session_[session].push_back(index);
    }
    apply_actions(state_, *tape_, actions_by_session_.front());
    settle(state_, tape_->close_times.front());
    staged_ = state_;
    staged_.held_order.reserve(count);
    staged_.receivables.reserve(tape_->actions.size());
}

StepOutcome FinancialSession::step(uint8_t action) {
    if (state_.done || action >= exposures.size()) {
        throw std::invalid_argument("El episodio ha terminado o la decisión no está admitida");
    }
    staged_ = state_;
    submit(staged_, *tape_, parameters_, action, ranking_);
    const auto following = state_.cursor + 1;
    const auto prices = tape_->frame(following);
    apply_actions(staged_, *tape_, actions_by_session_[following]);
    settle(staged_, tape_->open_times[following]);
    mt_account_v1 next_account{};
    std::array<char, error_capacity> error{};
    const int status = mt_simulation_step_v1(
        static_cast<uint32_t>(state_.positions.size()), 1, currencies_.data(), lots_.data(),
        staged_.retired.data(), prices.data(), staged_.positions.data(), &staged_.account,
        parameters_.cost_bps / basis_point_denominator, parameters_.participation,
        tape_->close_times[state_.cursor], tape_->open_times[following],
        tape_->close_times[following], next_positions_.data(), &next_account, trades_.data(),
        error.data(), error.size());
    check_native(status, error);
    staged_.positions.swap(next_positions_);
    staged_.account = next_account;
    settle(staged_, tape_->close_times[following]);
    staged_.account.nav = valuation(staged_, prices);
    std::erase_if(staged_.held_order, [this](std::size_t asset) {
        return staged_.positions[asset].quantity == 0;
    });
    ranking_.clear();
    for (std::size_t asset = 0; asset < staged_.positions.size(); ++asset) {
        if (staged_.positions[asset].quantity > 0 && state_.positions[asset].quantity == 0) {
            ranking_.push_back(asset);
        }
    }
    std::sort(ranking_.begin(), ranking_.end(), [this](std::size_t left, std::size_t right) {
        return tape_->assets[left] < tape_->assets[right];
    });
    staged_.held_order.insert(staged_.held_order.end(), ranking_.begin(), ranking_.end());
    StepOutcome result;
    result.reward_valid = !std::isnan(staged_.account.nav);
    result.terminated = result.reward_valid && staged_.account.nav == 0;
    result.truncated = !result.reward_valid ||
                       (following + 1 == tape_->close_times.size() && !result.terminated);
    result.reward = !result.reward_valid ? 0 : result.terminated ? parameters_.ruin_penalty
                                                               : std::log(staged_.account.nav) -
                                                                     std::log(state_.account.nav);
    if (!std::isfinite(result.reward) ||
        !std::isfinite(staged_.account.turnover / parameters_.capital)) {
        throw std::overflow_error("La recompensa o el turnover exceden el rango numérico");
    }
    if (result.reward_valid) {
        staged_.peak_nav = std::max(staged_.peak_nav, staged_.account.nav);
        staged_.max_drawdown = std::max(staged_.max_drawdown,
                                         1 - staged_.account.nav / staged_.peak_nav);
        if (!std::isfinite(staged_.account.nav / parameters_.capital - 1)) {
            throw std::overflow_error("Las métricas normalizadas exceden el rango numérico");
        }
    }
    for (const auto& trade : trades_) {
        if (trade.reason != MT_ORDER_COMPLETE) {
            ++staged_.unfilled_order_observations;
        }
    }
    for (const auto asset : staged_.held_order) {
        if (std::isnan(prices[asset * price_width + close_column])) {
            result.unvalued.push_back(asset);
        }
    }
    result.trades = trades_;
    staged_.cursor = following;
    staged_.done = result.terminated || result.truncated;
    std::swap(state_, staged_);
    return result;
}

std::vector<float> FinancialSession::observation() const {
    const auto prices = tape_->frame(state_.cursor);
    const auto previous = tape_->frame(state_.cursor == 0 ? 0 : state_.cursor - 1);
    const auto scores = tape_->predictions(state_.cursor);
    std::vector<float> result(state_.positions.size() * observation_width + 2);
    std::array<char, error_capacity> error{};
    const int status = mt_simulation_observation_v1(
        static_cast<uint32_t>(state_.positions.size()), prices.data(), previous.data(),
        scores.data(),
        state_.retired.data(), state_.positions.data(), state_.account.nav, state_.account.cash,
        parameters_.score_scale, result.data(), error.data(), error.size());
    check_native(status, error);
    return result;
}

SessionSnapshot FinancialSession::snapshot() const {
    return state_;
}

void FinancialSession::restore(const SessionSnapshot& snapshot) {
    validate_snapshot(snapshot, *tape_, parameters_);
    staged_ = snapshot;
    std::swap(state_, staged_);
}

FinancialMetrics FinancialSession::metrics() const {
    FinancialMetrics result;
    result.costs = state_.account.costs;
    result.turnover = state_.account.turnover / parameters_.capital;
    result.steps = state_.cursor;
    result.completed = state_.done && std::isfinite(state_.account.nav);
    if (!state_.done) {
        result.invalid_reason = "incomplete";
    } else if (!result.completed) {
        result.invalid_reason = "missing_close";
    } else {
        result.net_return = state_.account.nav / parameters_.capital - 1;
        result.max_drawdown = state_.max_drawdown;
        if (state_.account.nav == 0) {
            result.invalid_reason = "ruined";
        }
    }
    return result;
}

bool FinancialSession::done() const noexcept {
    return state_.done;
}

std::size_t FinancialSession::cursor() const noexcept {
    return state_.cursor;
}

uint8_t policy_action(ReferencePolicy policy, std::size_t step) {
    switch (policy) {
    case ReferencePolicy::cash:
        return 1;
    case ReferencePolicy::hold_initial:
        return step == 0 ? static_cast<uint8_t>(exposures.size() - 1) : 0;
    case ReferencePolicy::rebalance_25:
        return 2;
    case ReferencePolicy::rebalance_50:
        return 3;
    case ReferencePolicy::rebalance_75:
        return 4;
    case ReferencePolicy::rebalance_100:
        return static_cast<uint8_t>(exposures.size() - 1);
    }
    throw std::invalid_argument("La política de referencia no está definida");
}

std::string policy_name(ReferencePolicy policy) {
    switch (policy) {
    case ReferencePolicy::cash:
        return "cash";
    case ReferencePolicy::hold_initial:
        return "hold_initial";
    case ReferencePolicy::rebalance_25:
        return "rebalance_25";
    case ReferencePolicy::rebalance_50:
        return "rebalance_50";
    case ReferencePolicy::rebalance_75:
        return "rebalance_75";
    case ReferencePolicy::rebalance_100:
        return "rebalance_100";
    }
    throw std::invalid_argument("La política de referencia no está definida");
}

ReferencePolicy parse_policy(const std::string& name) {
    constexpr std::array policies{ReferencePolicy::cash, ReferencePolicy::hold_initial,
                                 ReferencePolicy::rebalance_25, ReferencePolicy::rebalance_50,
                                 ReferencePolicy::rebalance_75, ReferencePolicy::rebalance_100};
    for (const auto policy : policies) {
        if (policy_name(policy) == name) {
            return policy;
        }
    }
    throw std::invalid_argument("La política de referencia no está definida");
}
}
