#include "mars_titan/reconstructed_tape.hpp"
#include "mars_titan/simulation_files.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>

namespace mars_titan::simulation {
namespace {
using Json = nlohmann::json;
constexpr std::size_t price_width = 5;
constexpr std::size_t open_column = 0;
constexpr std::size_t high_column = 1;
constexpr std::size_t low_column = 2;
constexpr std::size_t volume_column = 4;
constexpr std::size_t digest_length = 64;
constexpr std::size_t maximum_segments = 128;
constexpr std::size_t segment_fields = 6;
constexpr int64_t maximum_payment_lag = 252;
constexpr int64_t final_test_start = 1'704'067'200'000'000;
constexpr int64_t microseconds_per_day = 86'400'000'000;
constexpr int64_t beijing_offset = 8 * 3'600'000'000;
constexpr std::size_t board_prefix = 3;
constexpr double star_minimum_order = 200;
constexpr double board_lot = 100;
// Inversas exactas de las unidades de cotización de unadjusted_prices.py.
constexpr double fraction_inverse = 256;
constexpr double cent_inverse = 100;
constexpr double subdollar_inverse = 10'000;

// RECONSTRUCTED_CONTRACT de simulation/market.py, salvo corporate_actions_complete.
constexpr std::array<std::pair<std::string_view, std::string_view>, 9> contract{{
    {"price_basis", "unadjusted_reconstructed"},
    {"corporate_actions", "provider_events_in_verified_rows"},
    {"exit_returns", "unavailable"},
    {"population", "listed_through_2025_03"},
    {"rows", "verified_only"},
    {"non_trading", "zero_volume_or_missing_row_without_execution"},
    {"valuation", "last_traded_close"},
    {"same_day_split_dividend", "lower_cash_without_execution"},
    {"off_grid_open", "without_execution"},
}};
constexpr std::array<std::string_view, 7> audit_fields{
    "market", "edition_id", "evidence_sha256", "walk_forward", "prediction_fit_ends",
    "assumptions", "corporate_actions_complete"};
constexpr std::array<std::string_view, 4> partitions{"train", "validation", "calibration",
                                                     "evaluation"};

void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::invalid_argument(std::string(message));
    }
}

bool digest(const Json& value) {
    return value.is_string() && value.get_ref<const std::string&>().size() == digest_length &&
           std::ranges::all_of(value.get_ref<const std::string&>(), [](char ch) {
               return (ch >= '0' && ch <= '9') || (ch >= 'a' && ch <= 'f');
           });
}

std::string_view currency_of(std::string_view market) {
    if (market == "US") {
        return "USD";
    }
    if (market == "CN") {
        return "CNY";
    }
    return {};
}

WalkForwardSegment segment_from(const Json& value) {
    require(value.is_object() && value.size() == segment_fields &&
                digest(value.at("receipt_sha256")) && value.at("fold").is_string() &&
                value.at("partition").is_string() &&
                std::ranges::find(partitions, value.at("partition").get<std::string>()) !=
                    partitions.end(),
            "Los tramos walk-forward de la cinta no son válidos");
    WalkForwardSegment result{read_json_int64(value.at("start")), read_json_int64(value.at("end")),
                              read_json_int64(value.at("labels_used_until"))};
    require(result.start >= 0 && result.start < result.end && result.end <= final_test_start,
            "Los tramos walk-forward de la cinta no son válidos");
    return result;
}

int64_t beijing_day(int year, unsigned month, unsigned day) {
    const std::chrono::sys_days date{std::chrono::year{year} / std::chrono::month{month} /
                                     std::chrono::day{day}};
    return date.time_since_epoch().count() * microseconds_per_day - beijing_offset;
}

struct Day {
    int year;
    unsigned month;
    unsigned day;
};

RulePeriod band(Day start, Day end, double value) {
    return {beijing_day(start.year, start.month, start.day),
            beijing_day(end.year, end.month, end.day), value, 0, 0};
}

RulePeriod duty(Day start, Day end, double buy, double sell) {
    return {beijing_day(start.year, start.month, start.day),
            beijing_day(end.year, end.month, end.day), 0, buy, sell};
}

std::string board(std::string_view asset) {
    if (asset.starts_with("CN/")) {
        asset.remove_prefix(3);
    }
    constexpr std::size_t code_length = 6;
    require(asset.size() == code_length + 3 && asset[code_length] == '.' &&
                std::all_of(asset.begin(), asset.begin() + code_length,
                            [](char ch) { return ch >= '0' && ch <= '9'; }),
            "El activo no tiene un código de acción A reconocible");
    const auto exchange = asset.substr(code_length + 1);
    require(exchange == "SS" || exchange == "SH" || exchange == "SZ",
            "El activo no tiene un código de acción A reconocible");
    const auto prefix = asset.substr(0, board_prefix);
    if (exchange == "SZ") {
        if (prefix == "000" || prefix == "001" || prefix == "002" || prefix == "003") {
            return "main";
        }
        if (prefix == "300" || prefix == "301") {
            return "chinext";
        }
    } else {
        if (prefix == "600" || prefix == "601" || prefix == "603" || prefix == "605") {
            return "main";
        }
        if (prefix == "688" || prefix == "689") {
            return "star";
        }
    }
    throw std::invalid_argument("El tablero del activo no tiene reglas acreditadas");
}

// Una apertura ejecutable cae exactamente en la rejilla de su mercado y fecha.
bool on_grid(std::string_view market, int64_t close_time, double value) {
    const auto exact = [value](double inverse) {
        return std::nearbyint(value * inverse) / inverse == value;
    };
    if (market == "CN") {
        return exact(cent_inverse);
    }
    using std::chrono::year_month_day;
    const year_month_day date{std::chrono::floor<std::chrono::days>(
        std::chrono::sys_time<std::chrono::microseconds>{std::chrono::microseconds{close_time}})};
    const year_month_day pilot{std::chrono::year{2000} / 8 / 28};
    const year_month_day complete{std::chrono::year{2001} / 4 / 9};
    if (date < pilot) {
        return exact(fraction_inverse);
    }
    if (date < complete) {
        return exact(fraction_inverse) || exact(cent_inverse);
    }
    return exact(value < 1 ? subdollar_inverse : cent_inverse);
}

std::string action_id(const MarketTape& tape, const CorporateAction& action) {
    return tape.assets.at(action.asset) + "/" + std::to_string(action.effective_at) + "/" +
           (action.kind == CorporateKind::dividend ? "dividend" : "split");
}

void check_sessions(const MarketTape& tape, const ReconstructedAudit& audit) {
    const auto sessions = tape.close_times.size();
    require(audit.prediction_fit_ends.size() == sessions,
            "Las predicciones históricas necesitan un ajuste anterior a cada decisión");
    std::vector<std::size_t> owned(audit.segments.size(), 0);
    for (std::size_t session = 0; session < sessions; ++session) {
        const auto time = tape.close_times[session];
        const auto following = std::ranges::upper_bound(
            audit.segments, time, {}, [](const WalkForwardSegment& item) { return item.start; });
        require(following != audit.segments.begin() && time < std::prev(following)->end,
                "Una sesión de la cinta queda fuera de sus tramos walk-forward");
        const auto owner = static_cast<std::size_t>(std::prev(following) - audit.segments.begin());
        ++owned[owner];
        // Los cortes salen de los recibos y no de las fechas fijas de las particiones.
        require(audit.prediction_fit_ends[session] == audit.segments[owner].labels_used_until,
                "Las predicciones históricas necesitan un ajuste anterior a cada decisión");
    }
    require(std::ranges::none_of(owned, [](std::size_t count) { return count == 0; }),
            "Un tramo walk-forward declarado no tiene sesiones");
}

void check_prices(const MarketTape& tape, std::string_view market) {
    const auto count = tape.assets.size();
    for (std::size_t session = 0; session < tape.close_times.size(); ++session) {
        for (std::size_t asset = 0; asset < count; ++asset) {
            const auto row = tape.frame(session).subspan(asset * price_width, price_width);
            const auto volume = row[volume_column];
            // Sin negociación no hay apertura ejecutable ni precio nuevo.
            if (!(volume > 0)) {
                require(std::isnan(row[open_column]) && std::isnan(row[high_column]) &&
                            std::isnan(row[low_column]),
                        "Una sesión sin negociación conserva precios ejecutables");
            } else {
                require(!std::isnan(row[high_column]) && !std::isnan(row[low_column]),
                        "Una sesión negociada no conserva su máximo y su mínimo");
            }
            require(std::isnan(row[open_column]) ||
                        on_grid(market, tape.close_times[session], row[open_column]),
                    "Una apertura fuera de rejilla no puede ser ejecutable");
        }
    }
}

void check_actions(const MarketTape& tape, const ReconstructedAudit& audit) {
    const auto sessions = static_cast<int64_t>(tape.open_times.size());
    for (const auto& action : tape.actions) {
        require(action.kind == CorporateKind::dividend || action.kind == CorporateKind::split,
                "La cinta reconstruida solo admite dividendos y splits del proveedor");
        require(action.id == action_id(tape, action),
                "La acción corporativa no conserva la identidad de su activo y apertura");
        const auto moment = std::ranges::lower_bound(tape.open_times, action.effective_at);
        require(moment != tape.open_times.end() && *moment == action.effective_at,
                "La acción corporativa no pertenece al calendario");
        const auto position = moment - tape.open_times.begin();
        if (action.kind == CorporateKind::split) {
            require(action.value > 0 && action.value != 1 && !action.pay_at,
                    "El split necesita una proporción distinta de uno");
            continue;
        }
        // El plazo de pago es un supuesto declarado. Si cae fuera de la cinta se paga después.
        const auto paid = position + audit.dividend_payment_lag_sessions;
        const auto expected = paid < sessions ? tape.open_times[static_cast<std::size_t>(paid)]
                                              : tape.close_times.back() + 1;
        require(action.value > 0 && action.pay_at == expected,
                "El dividendo no conserva su importe o el plazo de pago declarado");
        const auto split = std::ranges::find_if(tape.actions, [&action](const auto& other) {
            return other.kind == CorporateKind::split && other.asset == action.asset &&
                   other.effective_at == action.effective_at;
        });
        // Con split y dividendo el mismo día el importe es ambiguo y esa apertura no se ejecuta.
        require(split == tape.actions.end() ||
                    std::isnan(tape.frame(static_cast<std::size_t>(position))[action.asset *
                                                                                  price_width]),
                "Un dividendo ambiguo con split conserva una apertura ejecutable");
    }
}
} // namespace

ReconstructedAudit read_reconstructed_audit(const Json& identity, std::string_view currency) {
    const auto& audit = identity.at("audit");
    require(audit.is_object() && audit.size() == contract.size() + audit_fields.size() &&
                std::ranges::all_of(contract,
                                    [&audit](const auto& entry) {
                                        return audit.contains(std::string(entry.first)) &&
                                               audit.at(std::string(entry.first)) ==
                                                   std::string(entry.second);
                                    }) &&
                std::ranges::all_of(audit_fields,
                                    [&audit](auto name) {
                                        return audit.contains(std::string(name));
                                    }),
            "El escenario real no acredita el contrato histórico reconstruido ni sus cortes "
            "walk-forward");
    ReconstructedAudit result;
    const auto& assumptions = audit.at("assumptions");
    require(audit.at("corporate_actions_complete") == false && audit.at("market").is_string() &&
                !currency.empty() &&
                currency_of(audit.at("market").get<std::string>()) == currency &&
                digest(audit.at("edition_id")) && digest(audit.at("evidence_sha256")) &&
                assumptions.is_object() && assumptions.size() == 1 &&
                assumptions.contains("dividend_payment_lag_sessions"),
            "La cinta reconstruida no declara su tratamiento y sus limitaciones");
    result.market = audit.at("market").get<std::string>();
    result.dividend_payment_lag_sessions =
        read_json_int64(assumptions.at("dividend_payment_lag_sessions"));
    require(result.dividend_payment_lag_sessions >= 0 &&
                result.dividend_payment_lag_sessions <= maximum_payment_lag,
            "La cinta reconstruida no declara su tratamiento y sus limitaciones");
    const auto& segments = audit.at("walk_forward");
    require(segments.is_array() && !segments.empty() && segments.size() <= maximum_segments,
            "Los tramos walk-forward de la cinta no son válidos");
    for (const auto& value : segments) {
        const auto segment = segment_from(value);
        require(result.segments.empty() || result.segments.back().end <= segment.start,
                "Los tramos walk-forward de la cinta no son válidos");
        result.segments.push_back(segment);
    }
    const auto& fits = audit.at("prediction_fit_ends");
    require(fits.is_array() && fits.size() <= maximum_sessions,
            "Las predicciones históricas necesitan un ajuste anterior a cada decisión");
    result.prediction_fit_ends.reserve(fits.size());
    for (const auto& value : fits) {
        result.prediction_fit_ends.push_back(read_json_int64(value));
    }
    result.basis = result.market + "/" + audit.at("edition_id").get<std::string>() + "/lag-" +
                   std::to_string(result.dividend_payment_lag_sessions);
    return result;
}

void admit_reconstructed_tape(MarketTape& tape, const ReconstructedAudit& audit) {
    require(std::ranges::all_of(tape.assets,
                                [&audit](const std::string& asset) {
                                    return asset.starts_with(audit.market + "/");
                                }),
            "La cinta reconstruida necesita activos de su mercado");
    check_sessions(tape, audit);
    check_prices(tape, audit.market);
    check_actions(tape, audit);
    if (audit.market == "CN") {
        // Lotes, bandas diarias y timbre solo tienen sentido con precios negociados.
        require(tape.instruments.size() == tape.assets.size(),
                "Una cinta china reconstruida necesita las reglas de acciones A");
        for (std::size_t asset = 0; asset < tape.assets.size(); ++asset) {
            require(tape.instruments[asset] == china_a_share_rules(tape.assets[asset]),
                    "Una cinta china reconstruida necesita las reglas de acciones A");
        }
    }
    tape.prediction_fit_ends = audit.prediction_fit_ends;
    tape.historical_basis = audit.basis;
    tape.historical_audit_verified = true;
}

InstrumentRules china_a_share_rules(std::string_view asset) {
    // Fuentes y vigencias en docs/engineering/china-market-rules.md, hasta el 1 de enero de 2024.
    constexpr Day verified_end{2024, 1, 1};
    const auto kind = board(asset);
    InstrumentRules result;
    result.rules = "cn_a_share_v1_" + kind;
    result.lot = kind == "star" ? 1 : board_lot;
    result.minimum_order = kind == "star" ? star_minimum_order : 0;
    result.odd_lot_exit = true;
    constexpr double main_band = 0.10;
    constexpr double wide_band = 0.20;
    if (kind == "main") {
        result.price_limits = {band({2006, 7, 1}, verified_end, main_band)};
    } else if (kind == "chinext") {
        result.price_limits = {band({2006, 7, 1}, {2020, 8, 24}, main_band),
                               band({2020, 8, 24}, verified_end, wide_band)};
    } else {
        result.price_limits = {band({2019, 7, 22}, verified_end, wide_band)};
    }
    constexpr double low_duty = 0.001;
    constexpr double high_duty = 0.003;
    constexpr double halved_duty = 0.0005;
    result.taxes = {duty({2005, 1, 24}, {2007, 5, 30}, low_duty, low_duty),
                    duty({2007, 5, 30}, {2008, 4, 24}, high_duty, high_duty),
                    duty({2008, 4, 24}, {2008, 9, 19}, low_duty, low_duty),
                    duty({2008, 9, 19}, {2023, 8, 28}, 0, low_duty),
                    duty({2023, 8, 28}, verified_end, 0, halved_duty)};
    return result;
}

} // namespace mars_titan::simulation
