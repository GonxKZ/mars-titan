#include "mars_titan/cohort_execution.hpp"
#include "mars_titan/simulation_files.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <numeric>
#include <set>
#include <span>
#include <stdexcept>
#include <string_view>

namespace {
using namespace mars_titan::cohorts;
using Clock = std::chrono::steady_clock;
constexpr std::size_t default_features = 4;
constexpr std::size_t maximum_features = 4096;
constexpr std::size_t maximum_assets = 512;
constexpr std::size_t maximum_steps = 10000;
constexpr std::size_t maximum_recoveries = 100;
constexpr std::size_t maximum_seed = 1000000;
constexpr std::int64_t tick = 10;
constexpr double gain = 0.1;
constexpr double horizon_weight = 0.01;
constexpr std::string_view asset_prefix = "US/CONTROL";
struct Options {
    std::filesystem::path output;
    std::size_t assets = 16;
    std::size_t features = default_features;
    std::size_t cohorts = 128;
    std::size_t batch = 32;
    std::size_t seed = 42;
    std::size_t recovery_repeats = 3;
    bool resume = false;
    bool reverse = false;
};
struct Metrics {
    double predict_seconds = 0;
    double update_seconds = 0;
    std::size_t predict_calls = 0;
};
double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}
std::size_t integer(std::string_view text, std::size_t maximum) {
    std::size_t value = 0;
    const auto result = std::from_chars(text.begin(), text.end(), value);
    if (result.ec != std::errc{} || result.ptr != text.end() || value > maximum) {
        throw std::invalid_argument("El argumento entero no es válido o excede su límite");
    }
    return value;
}
Options options(std::span<char*> args) {
    Options result;
    std::set<std::string_view> seen;
    for (std::size_t i = 1; i < args.size(); ++i) {
        const std::string_view key(args[i]);
        if (!seen.insert(key).second) {
            throw std::invalid_argument("Hay un argumento repetido");
        }
        if (key == "--resume") {
            result.resume = true;
            continue;
        }
        if (key == "--reverse") {
            result.reverse = true;
            continue;
        }
        if (++i == args.size()) {
            throw std::invalid_argument("Falta el valor de un argumento");
        }
        const std::string_view value(args[i]);
        if (key == "--output") {
            result.output = value;
        } else if (key == "--assets") {
            result.assets = integer(value, maximum_assets);
        } else if (key == "--features") {
            result.features = integer(value, maximum_features);
        } else if (key == "--cohorts") {
            result.cohorts = integer(value, maximum_steps);
        } else if (key == "--batch") {
            result.batch = integer(value, maximum_assets);
        } else if (key == "--seed") {
            result.seed = integer(value, maximum_seed);
        } else if (key == "--recovery-repeats") {
            result.recovery_repeats = integer(value, maximum_recoveries);
        } else {
            throw std::invalid_argument("Argumento desconocido: " + std::string(key));
        }
    }
    if (result.output.empty() || result.assets == 0 || result.features == 0 ||
        result.cohorts == 0 || result.batch == 0 || result.recovery_repeats == 0) {
        throw std::invalid_argument("Se requiere --output y presupuestos positivos");
    }
    return result;
}
double signal(std::size_t seed, std::size_t step, std::size_t asset, std::size_t feature) {
    constexpr std::size_t step_weight = 23;
    constexpr std::size_t asset_weight = 7;
    constexpr std::size_t modulus = 97;
    return static_cast<double>((seed + step_weight * step + asset_weight * asset + feature) %
                               modulus) /
           static_cast<double>(modulus);
}
Definition definition(const Options& args) {
    using mars_titan::simulation::content_sha256;
    Definition result;
    result.identity = {
        content_sha256(Json{{"kind", "synthetic_numeric_stream_v1"},
                            {"assets", args.assets},
                            {"features", args.features},
                            {"seed", args.seed}}
                           .dump()),
        content_sha256("numeric-observations-without-targets-v1"),
        content_sha256(Json{{"kind", "numeric_float64_v1"}, {"width", args.features}}.dump()),
        content_sha256("mean-features-plus-horizon-and-delayed-bias-v1")};
    result.tasks = {{"signal", 1}, {"signal", 3}};
    result.initial_state = Json{{"bias", 0.0}, {"feedback_count", 0}};
    result.limits.feature_width = args.features;
    result.limits.max_assets = maximum_assets;
    return result;
}
Callbacks callbacks(const std::shared_ptr<Metrics>& metrics) {
    return {
        [metrics](std::span<const Observation> observations, const Task& task, const Json& state) {
            const auto started = Clock::now();
            std::vector<double> values;
            values.reserve(observations.size());
            for (const auto& row : observations) {
                const double mean = std::accumulate(row.features.begin(), row.features.end(), 0.0) /
                                    static_cast<double>(row.features.size());
                values.push_back(mean + state.at("bias").get<double>() +
                                 static_cast<double>(task.horizon) * horizon_weight);
            }
            metrics->predict_seconds += elapsed(started);
            ++metrics->predict_calls;
            return values;
        },
        [metrics](const Json& state, std::span<const ResolvedFeedback> outcomes) {
            const auto started = Clock::now();
            auto next = state;
            double error = 0;
            for (const auto& outcome : outcomes) {
                error += outcome.label.value - outcome.prediction.value;
            }
            if (!outcomes.empty()) {
                next["bias"] = state.at("bias").get<double>() +
                               gain * error / static_cast<double>(outcomes.size());
            }
            next["feedback_count"] =
                state.at("feedback_count").get<std::size_t>() + outcomes.size();
            metrics->update_seconds += elapsed(started);
            return next;
        }};
}
Cohort observation(const Options& args, std::size_t cursor) {
    Cohort result{cursor, static_cast<std::int64_t>(cursor + 1) * tick, {}};
    result.observations.reserve(args.assets);
    for (std::size_t asset = 0; asset < args.assets; ++asset) {
        Observation row{std::string(asset_prefix) + std::to_string(asset), result.cutoff, {}};
        for (std::size_t feature = 0; feature < args.features; ++feature) {
            row.features.push_back(signal(args.seed, cursor, asset, feature));
        }
        result.observations.push_back(std::move(row));
    }
    if (args.reverse) {
        std::ranges::reverse(result.observations);
    }
    return result;
}
std::vector<Feedback> labels(const Options& args, const Executor& run, std::int64_t cutoff) {
    std::vector<Feedback> result;
    for (const auto& prediction : run.pending()) {
        const auto available =
            prediction.decision_at + static_cast<std::int64_t>(prediction.task.horizon) * tick;
        if (available > cutoff) {
            continue;
        }
        if (!prediction.asset.starts_with(asset_prefix)) {
            throw std::invalid_argument("El activo no pertenece al control sintético");
        }
        const auto asset = integer(std::string_view(prediction.asset).substr(asset_prefix.size()),
                                   args.assets - 1);
        double mean = 0;
        for (std::size_t feature = 0; feature < args.features; ++feature) {
            mean += signal(args.seed, prediction.generation - 1, asset, feature);
        }
        mean /= static_cast<double>(args.features);
        result.push_back(
            {prediction.id, 0, available,
             mean + static_cast<double>(prediction.task.horizon) * horizon_weight + gain});
    }
    if (args.reverse) {
        std::ranges::reverse(result);
    }
    return result;
}
Json percentile(std::vector<double> values, double fraction) {
    if (values.empty()) {
        return nullptr;
    }
    std::ranges::sort(values);
    const double position = fraction * static_cast<double>(values.size() - 1);
    const auto lower = static_cast<std::size_t>(std::floor(position));
    const auto upper = std::min(lower + 1, values.size() - 1);
    return values[lower] +
           (values[upper] - values[lower]) * (position - static_cast<double>(lower));
}
Json run(const Options& args) {
    const auto started = Clock::now();
    const auto metrics = std::make_shared<Metrics>();
    const auto contract = definition(args);
    const auto operators = callbacks(metrics);
    Json snapshot;
    std::vector<double> steps;
    std::size_t first = 0;
    std::vector<std::string> exposures;
    {
        Executor executor(args.output, contract, operators, args.resume);
        first = executor.cursor();
        if (first > args.cohorts) {
            throw std::invalid_argument("El corte solicitado precede al cursor confirmado");
        }
        steps.reserve(args.cohorts - first);
        while (executor.cursor() < args.cohorts) {
            const auto cohort = observation(args, executor.cursor());
            const auto feedback = labels(args, executor, cohort.cutoff);
            const auto begin = Clock::now();
            const auto result = executor.step(cohort, feedback, args.batch);
            steps.push_back(elapsed(begin));
            if (exposures.empty() && !result.applied.empty()) {
                exposures = {replay_id(result.applied.front(), 0),
                             replay_id(result.applied.front(), 1)};
            }
        }
        snapshot = executor.snapshot();
    }
    std::vector<double> recovery;
    for (std::size_t i = 0; i < args.recovery_repeats; ++i) {
        const auto begin = Clock::now();
        Executor reopened(args.output, contract, operators, true);
        recovery.push_back(elapsed(begin));
        if (reopened.snapshot() != snapshot) {
            throw std::runtime_error("La reapertura no conserva el estado confirmado");
        }
    }
    constexpr double median = 0.5;
    constexpr double tail = 0.95;
    constexpr double extreme = 0.99;
    const auto rows = (args.cohorts - first) * args.assets;
    const auto total_steps = std::accumulate(steps.begin(), steps.end(), 0.0);
    return Json{
        {"schema_version", 1},
        {"kind", "synthetic_cohort_control"},
        {"historical_evidence", false},
        {"candidate_trained", false},
        {"gpu_used", false},
        {"energy_joules", nullptr},
        {"total_cost", nullptr},
        {"committed_cohorts", snapshot.at("cursor")},
        {"pending", snapshot.at("pending").size()},
        {"snapshot_sha256", mars_titan::simulation::content_sha256(snapshot.dump())},
        {"history_sha256", snapshot.at("record_sha256")},
        {"record_bytes", snapshot.at("log_bytes")},
        {"checkpoint_retention", 2},
        {"replay_exposure_ids", exposures},
        {"assets", args.assets},
        {"features", args.features},
        {"seed", args.seed},
        {"physical_batch", args.batch},
        {"measured_cohorts", steps.size()},
        {"input_bytes", rows * args.features * sizeof(double)},
        {"step_seconds", steps},
        {"step_p50_seconds", percentile(steps, median)},
        {"step_p95_seconds", percentile(steps, tail)},
        {"step_p99_seconds", percentile(steps, extreme)},
        {"rows_per_second", total_steps > 0 ? static_cast<double>(rows) / total_steps : 0.0},
        {"predict_seconds", metrics->predict_seconds},
        {"predict_calls", metrics->predict_calls},
        {"update_seconds", metrics->update_seconds},
        {"recovery_seconds", recovery},
        {"recovery_p50_seconds", percentile(recovery, median)},
        {"wall_seconds", elapsed(started)},
        {"peak_rss_bytes", mars_titan::simulation::process_memory_high_water()},
        {"compiler", MARS_TITAN_NATIVE_COMPILER_ID},
        {"compiler_version", MARS_TITAN_NATIVE_COMPILER_VERSION},
        {"build_type", MARS_TITAN_NATIVE_BUILD_TYPE},
        {"native_source_sha256", MARS_TITAN_NATIVE_SOURCE_SHA256},
        {"native_build_sha256", MARS_TITAN_NATIVE_BUILD_SHA256}};
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 1) {
            throw std::invalid_argument("Faltan los argumentos del ejecutable");
        }
        const auto arguments = std::span(argv, static_cast<std::size_t>(argc));
        if (arguments.size() == 2 && std::string_view(arguments[1]) == "--help") {
            std::cout << "Uso: mars-titan-cohorts --output DIRECTORIO [--resume] [--reverse]\n"
                         "  --cohorts N  --assets N  --features N  --batch N\n"
                         "  --seed N  --recovery-repeats N\n"
                         "Control numérico sintético con publicación y recuperación completas.\n";
            return 0;
        }
        const auto args = options(arguments);
        const auto report = run(args);
        mars_titan::simulation::atomic_json_file(args.output / "report.json", report,
                                                 default_file_bytes);
        std::cout << report.dump() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "No se completó el control de cohortes: " << error.what() << '\n';
        return 2;
    }
}
