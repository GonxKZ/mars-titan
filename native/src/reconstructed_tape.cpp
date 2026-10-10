#include "mars_titan/reconstructed_tape.hpp"
#include "mars_titan/simulation_files.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <map>
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
constexpr std::array<std::pair<std::string_view, std::string_view>, 11> contract{{
    {"price_basis", "unadjusted_reconstructed"},
    {"corporate_actions", "provider_events_in_verified_rows"},
    {"exit_returns", "source_exit_price_or_masked_position"},
    {"population", "listed_through_2025_03"},
    {"rows", "verified_only"},
    {"non_trading", "zero_volume_or_missing_row_without_execution"},
    {"valuation", "last_traded_close"},
    {"same_day_split_dividend", "lower_cash_without_execution"},
    {"off_grid_open", "without_execution"},
    {"series_end", "delisting_at_next_open"},
    {"outside_universe_assets", "masked_without_prices_or_predictions"},
}};
constexpr std::array<std::string_view, 10> audit_fields{
    "market",           "edition_id",     "evidence_sha256", "walk_forward",
    "prediction_fit_ends", "assumptions", "corporate_actions_complete", "outside_universe",
    "delistings",       "listing_status"};
// Límites de la tabla del estado de cotización, como listing_status.py.
constexpr std::string_view status_cutoff = "2023-12-31";
// Primera fecha con tramos comprobados, como listing_status.COVERAGE_FROM.
constexpr std::string_view status_coverage = "2010-01-01";
constexpr std::size_t maximum_special_spans = 30;
constexpr std::size_t iso_day_length = 10;
constexpr double special_treatment_band = 0.05;
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
                std::ranges::all_of(asset.substr(0, code_length),
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
bool on_grid(std::string_view market, std::chrono::sys_days day, double value) {
    const auto exact = [value](double inverse) {
        return std::nearbyint(value * inverse) / inverse == value;
    };
    if (market == "CN") {
        return exact(cent_inverse);
    }
    using std::chrono::year_month_day;
    const year_month_day date{day};
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

std::string_view kind_name(CorporateKind kind) {
    switch (kind) {
    case CorporateKind::split:
        return "split";
    case CorporateKind::dividend:
        return "dividend";
    case CorporateKind::writeoff:
        return "writeoff";
    case CorporateKind::delisting:
        return "delisting";
    case CorporateKind::unpriced_delisting:
        return "unpriced_delisting";
    }
    throw std::invalid_argument("La clase de acción corporativa no está admitida");
}

std::string action_id(const MarketTape& tape, const CorporateAction& action) {
    return tape.assets.at(action.asset) + "/" + std::to_string(action.effective_at) + "/" +
           std::string(kind_name(action.kind));
}

bool delisting(const CorporateAction& action) {
    return action.kind == CorporateKind::delisting ||
           action.kind == CorporateKind::unpriced_delisting;
}

// Fecha ISO de calendario. Devuelve la fecha o lanza si no es una fecha real.
Day iso_day(std::string_view value) {
    const auto digits = [value](std::size_t from, std::size_t count) {
        int result = 0;
        for (std::size_t index = from; index < from + count; ++index) {
            require(value[index] >= '0' && value[index] <= '9',
                    "Una fecha del estado de cotización no es ISO");
            result = result * 10 + (value[index] - '0');
        }
        return result;
    };
    require(value.size() == iso_day_length && value[4] == '-' && value[7] == '-',
            "Una fecha del estado de cotización no es ISO");
    const Day result{digits(0, 4), static_cast<unsigned>(digits(5, 2)),
                     static_cast<unsigned>(digits(8, 2))};
    const std::chrono::year_month_day date{std::chrono::year{result.year} /
                                           std::chrono::month{result.month} /
                                           std::chrono::day{result.day}};
    require(date.ok(), "Una fecha del estado de cotización no es una fecha real");
    return result;
}

int64_t beijing_at(std::string_view value) {
    const auto day = iso_day(value);
    return beijing_day(day.year, day.month, day.day);
}

// Fecha de calendario de la sesión: el cierre de EE. UU. y de China cae en el mismo día UTC.
std::string session_day(const MarketTape& tape, std::size_t session) {
    const std::chrono::year_month_day date{std::chrono::floor<std::chrono::days>(
        std::chrono::sys_time<std::chrono::microseconds>{
            std::chrono::microseconds{tape.close_times.at(session)}})};
    std::array<char, iso_day_length + 1> text{};
    std::snprintf(text.data(), text.size(), "%04d-%02u-%02u", static_cast<int>(date.year()),
                  static_cast<unsigned>(date.month()), static_cast<unsigned>(date.day()));
    return {text.data(), iso_day_length};
}

// Tramos [inicio, fin) ordenados, sin solapes y dentro de la cobertura comprobada. Solo el
// último puede seguir abierto en el corte.
std::vector<std::pair<std::string, std::optional<std::string>>> status_spans(const Json& spans) {
    require(spans.is_array() && spans.size() <= maximum_special_spans,
            "Los tramos del estado deben formar una lista");
    std::vector<std::pair<std::string, std::optional<std::string>>> result;
    std::optional<std::string> previous;
    for (std::size_t index = 0; index < spans.size(); ++index) {
        const auto& span = spans[index];
        require(span.is_array() && span.size() == 2 && span[0].is_string() &&
                    (span[1].is_string() || (span[1].is_null() && index + 1 == spans.size())),
                "Un tramo del estado necesita inicio y fin");
        const auto start = span[0].get<std::string>();
        static_cast<void>(iso_day(start));
        std::optional<std::string> end;
        if (span[1].is_string()) {
            end = span[1].get<std::string>();
            static_cast<void>(iso_day(*end));
            require(*end <= status_cutoff, "Un tramo del estado termina después del corte");
        }
        require(start >= status_coverage, "Un tramo del estado empieza antes de la cobertura");
        require(start <= status_cutoff && (!previous || *previous <= start) &&
                    (!end || start < *end),
                "Los tramos del estado deben estar ordenados y sin solapes");
        previous = end;
        result.emplace_back(start, end);
    }
    return result;
}

// Misma comprobación que china_entry en Python, sin consultar calendarios.
ChinaStatus china_status(const Json& value) {
    require(value.is_object() && value.size() == 5 && value.contains("listed_on") &&
                value.contains("limit_free_until") && value.contains("special_treatment") &&
                value.contains("share_reform_pending") && value.contains("limit_free_days") &&
                value.at("listed_on").is_string() &&
                (value.at("limit_free_until").is_null() ||
                 value.at("limit_free_until").is_string()) &&
                value.at("limit_free_days").is_array() &&
                value.at("limit_free_days").size() <= maximum_special_spans,
            "La entrada de una acción A no conserva sus campos");
    ChinaStatus result;
    result.listed_on = value.at("listed_on").get<std::string>();
    static_cast<void>(iso_day(result.listed_on));
    require(result.listed_on <= status_cutoff, "La admisión es posterior al corte");
    if (!value.at("limit_free_until").is_null()) {
        result.limit_free_until = value.at("limit_free_until").get<std::string>();
        static_cast<void>(iso_day(*result.limit_free_until));
        require(result.listed_on < *result.limit_free_until,
                "La exención de límites termina antes de la admisión");
    }
    result.special_treatment = status_spans(value.at("special_treatment"));
    result.share_reform_pending = status_spans(value.at("share_reform_pending"));
    for (const auto& item : value.at("limit_free_days")) {
        require(item.is_string(), "Un día sin límite debe ser una fecha");
        const auto day = item.get<std::string>();
        static_cast<void>(iso_day(day));
        require(day >= status_coverage && day <= status_cutoff &&
                    (result.limit_free_days.empty() || result.limit_free_days.back() < day),
                "Los días sin límite deben estar ordenados dentro de la cobertura");
        result.limit_free_days.push_back(day);
    }
    return result;
}

DelistingRecord delisting_record(const Json& value) {
    require(value.is_object() && value.size() == 2 && value.contains("last_session") &&
                value.contains("exit") && value.at("last_session").is_string(),
            "El registro de la baja no corresponde a su acción ni a su fuente");
    DelistingRecord result{value.at("last_session").get<std::string>(), std::nullopt};
    static_cast<void>(iso_day(result.last_session));
    const auto& exit = value.at("exit");
    if (!exit.is_null()) {
        require(exit.is_object() && exit.size() == 3 && exit.contains("price") &&
                    exit.contains("pay_at") && exit.at("price").is_number() &&
                    exit.contains("source_sha256") && digest(exit.at("source_sha256")),
                "El registro de la baja no corresponde a su acción ni a su fuente");
        result.exit = ExitRecord{exit.at("price").get<double>(),
                                 read_json_int64(exit.at("pay_at")),
                                 exit.at("source_sha256").get<std::string>()};
    }
    return result;
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
        const auto day = std::chrono::floor<std::chrono::days>(
            std::chrono::sys_time<std::chrono::microseconds>{
                std::chrono::microseconds{tape.close_times[session]}});
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
                        on_grid(market, day, row[open_column]),
                    "Una apertura fuera de rejilla no puede ser ejecutable");
        }
    }
}

void check_delistings(const MarketTape& tape, const ReconstructedAudit& audit) {
    // Cada baja de la cinta tiene su registro: la última sesión de la serie es la anterior a la
    // apertura de la baja y la salida coincide con la acción, con precio o sin él.
    std::map<std::string, const CorporateAction*> found;
    for (const auto& action : tape.actions) {
        if (delisting(action)) {
            found.emplace(tape.assets.at(action.asset), &action);
        }
    }
    require(found.size() == audit.delistings.size() &&
                std::ranges::all_of(audit.delistings,
                                    [&found](const auto& entry) {
                                        return found.contains(entry.first);
                                    }),
            "Cada baja de la cinta necesita su registro de salida");
    for (const auto& [asset, record] : audit.delistings) {
        const auto& action = *found.at(asset);
        const auto moment = std::ranges::lower_bound(tape.open_times, action.effective_at);
        require(moment != tape.open_times.begin() && moment != tape.open_times.end() &&
                    *moment == action.effective_at &&
                    record.last_session ==
                        session_day(tape, static_cast<std::size_t>(
                                              moment - tape.open_times.begin() - 1)) &&
                    record.exit.has_value() == (action.kind == CorporateKind::delisting),
                "El registro de la baja no corresponde a su acción ni a su fuente");
        if (record.exit) {
            require(record.exit->price == action.value && action.pay_at == record.exit->pay_at,
                    "El registro de la baja no corresponde a su acción ni a su fuente");
        }
    }
}

void check_universe(const MarketTape& tape, const ReconstructedAudit& audit) {
    // Un activo fuera del universo no tiene precios, predicciones ni acciones en la cinta.
    const auto count = tape.assets.size();
    require(audit.outside_universe.size() < count,
            "La cinta reconstruida necesita activos dentro de su universo");
    for (const auto& name : audit.outside_universe) {
        const auto found = std::ranges::lower_bound(tape.assets, name);
        require(found != tape.assets.end() && *found == name,
                "La cinta reconstruida necesita activos dentro de su universo");
        const auto asset = static_cast<std::size_t>(found - tape.assets.begin());
        require(std::ranges::none_of(tape.actions,
                                     [asset](const auto& action) {
                                         return action.asset == asset;
                                     }),
                "Un activo fuera del universo conserva acciones corporativas");
        for (std::size_t session = 0; session < tape.close_times.size(); ++session) {
            const auto row = tape.frame(session).subspan(asset * price_width, price_width);
            require(std::ranges::all_of(row, [](double value) { return std::isnan(value); }) &&
                        std::isnan(tape.predictions(session)[asset]),
                    "Un activo fuera del universo conserva precios o predicciones");
        }
    }
}

void check_actions(const MarketTape& tape, const ReconstructedAudit& audit) {
    const auto sessions = static_cast<int64_t>(tape.open_times.size());
    for (const auto& action : tape.actions) {
        require(action.kind == CorporateKind::dividend || action.kind == CorporateKind::split ||
                    delisting(action),
                "La cinta reconstruida solo admite dividendos y splits del proveedor y bajas");
        require(action.id == action_id(tape, action),
                "La acción corporativa no conserva la identidad de su activo y apertura");
        const auto moment = std::ranges::lower_bound(tape.open_times, action.effective_at);
        require(moment != tape.open_times.end() && *moment == action.effective_at,
                "La acción corporativa no pertenece al calendario");
        const auto position = moment - tape.open_times.begin();
        if (delisting(action)) {
            continue;
        }
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
    const auto& outside = audit.at("outside_universe");
    require(outside.is_array() && outside.size() <= maximum_assets,
            "La cinta reconstruida no declara su tratamiento y sus limitaciones");
    for (const auto& value : outside) {
        require(value.is_string() && (result.outside_universe.empty() ||
                                      result.outside_universe.back() < value.get<std::string>()),
                "La cinta reconstruida no declara su tratamiento y sus limitaciones");
        result.outside_universe.push_back(value.get<std::string>());
    }
    const auto& delistings = audit.at("delistings");
    require(delistings.is_object() && delistings.size() <= maximum_assets,
            "La cinta reconstruida no declara su tratamiento y sus limitaciones");
    for (const auto& [asset, value] : delistings.items()) {
        result.delistings.emplace(asset, delisting_record(value));
    }
    const auto& status = audit.at("listing_status");
    require(status.is_object() && status.size() == 2 && status.contains("source_sha256") &&
                digest(status.at("source_sha256")) && status.contains("assets") &&
                status.at("assets").is_object() && status.at("assets").size() <= maximum_assets,
            "La cinta reconstruida no declara su tratamiento y sus limitaciones");
    result.listing_status_sha256 = status.at("source_sha256").get<std::string>();
    for (const auto& [asset, value] : status.at("assets").items()) {
        result.listing_status.emplace(asset, china_status(value));
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
    check_delistings(tape, audit);
    check_universe(tape, audit);
    // El estado de cotización cubre exactamente los activos de una cinta china y ninguno en
    // EE. UU., como require_tape_status en Python.
    require(audit.listing_status.size() == (audit.market == "CN" ? tape.assets.size() : 0) &&
                std::ranges::all_of(tape.assets,
                                    [&audit](const std::string& asset) {
                                        return audit.market != "CN" ||
                                               audit.listing_status.contains(asset);
                                    }),
            "El estado de cotización de la cinta no cubre exactamente sus activos");
    require(audit.market != "CN" || tape.close_times.front() >= beijing_at(status_coverage),
            "Una cinta china empieza antes de la cobertura del estado de cotización");
    if (audit.market == "CN") {
        // Lotes, bandas diarias y timbre solo tienen sentido con precios negociados. Las
        // bandas incluyen el estado ST y la exención inicial de la auditoría de la cinta.
        require(tape.instruments.size() == tape.assets.size(),
                "Una cinta china reconstruida necesita las reglas de acciones A");
        for (std::size_t asset = 0; asset < tape.assets.size(); ++asset) {
            const auto& name = tape.assets[asset];
            require(tape.instruments[asset] ==
                        china_a_share_rules(name, audit.listing_status.at(name)),
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
    constexpr Day bands_from{2006, 7, 1};
    constexpr Day chinext_wide_from{2020, 8, 24};
    constexpr Day star_from{2019, 7, 22};
    if (kind == "main") {
        result.price_limits = {band(bands_from, verified_end, main_band)};
    } else if (kind == "chinext") {
        result.price_limits = {band(bands_from, chinext_wide_from, main_band),
                               band(chinext_wide_from, verified_end, wide_band)};
    } else {
        result.price_limits = {band(star_from, verified_end, wide_band)};
    }
    constexpr double low_duty = 0.001;
    constexpr double high_duty = 0.003;
    constexpr double halved_duty = 0.0005;
    constexpr Day duty_from{2005, 1, 24};
    constexpr Day duty_raised{2007, 5, 30};
    constexpr Day duty_lowered{2008, 4, 24};
    constexpr Day seller_only{2008, 9, 19};
    constexpr Day duty_halved{2023, 8, 28};
    result.taxes = {duty(duty_from, duty_raised, low_duty, low_duty),
                    duty(duty_raised, duty_lowered, high_duty, high_duty),
                    duty(duty_lowered, seller_only, low_duty, low_duty),
                    duty(seller_only, duty_halved, 0, low_duty),
                    duty(duty_halved, verified_end, 0, halved_duty)};
    return result;
}

InstrumentRules china_a_share_rules(std::string_view asset, const ChinaStatus& status) {
    constexpr Day verified_end{2024, 1, 1};
    auto result = china_a_share_rules(asset);
    const auto kind = board(asset);
    result.rules = "cn_a_share_v2_" + kind;
    const auto end = beijing_day(verified_end.year, verified_end.month, verified_end.day);
    std::vector<std::pair<int64_t, int64_t>> special;
    std::vector<std::pair<int64_t, int64_t>> free;
    if (kind == "main") {
        for (const auto* spans : {&status.special_treatment, &status.share_reform_pending}) {
            for (const auto& [start, stop] : *spans) {
                special.emplace_back(beijing_at(start), stop ? beijing_at(*stop) : end);
            }
        }
    }
    constexpr int64_t day_length = int64_t{86'400} * 1'000'000;
    for (const auto& day : status.limit_free_days) {
        free.emplace_back(beijing_at(day), beijing_at(day) + day_length);
    }
    if (status.limit_free_until) {
        free.emplace_back(beijing_at(status.listed_on), beijing_at(*status.limit_free_until));
    }
    const auto base = result.price_limits;
    std::vector<int64_t> cuts;
    for (const auto& period : base) {
        cuts.push_back(period.start);
        cuts.push_back(period.end);
    }
    for (const auto& spans : {special, free}) {
        for (const auto& [start, stop] : spans) {
            cuts.push_back(start);
            cuts.push_back(stop);
        }
    }
    std::ranges::sort(cuts);
    cuts.erase(std::unique(cuts.begin(), cuts.end()), cuts.end());
    const auto inside = [](const std::vector<std::pair<int64_t, int64_t>>& spans, int64_t at) {
        return std::ranges::any_of(spans, [at](const auto& span) {
            return span.first <= at && at < span.second;
        });
    };
    std::vector<RulePeriod> periods;
    for (std::size_t index = 0; index + 1 < cuts.size(); ++index) {
        const auto left = cuts[index];
        const auto right = cuts[index + 1];
        const auto period = std::ranges::find_if(base, [left](const RulePeriod& item) {
            return item.start <= left && left < item.end;
        });
        if (period == base.end() || inside(free, left)) {
            continue;
        }
        const double value = inside(special, left) ? special_treatment_band : period->band;
        if (!periods.empty() && periods.back().end == left && periods.back().band == value) {
            periods.back().end = right;
        } else {
            periods.push_back({left, right, value, 0, 0});
        }
    }
    result.price_limits = std::move(periods);
    return result;
}

} // namespace mars_titan::simulation
