#include "mars_titan/financial_batch.hpp"

#include <algorithm>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string_view>

namespace {
using namespace mars_titan::simulation;
using Clock = std::chrono::steady_clock;
constexpr std::size_t fixture_sessions = 128;
constexpr std::size_t fixture_assets = 64;
constexpr std::size_t fixture_environments = 256;
constexpr double price_scale = 100;
constexpr double volume = 1'000'000;
constexpr double signal_scale = 0.01;
constexpr double step_scale = 0.05;
constexpr std::size_t price_columns = 5;
constexpr std::size_t financial_columns = 6;
constexpr std::size_t digest_length = 64;
constexpr std::size_t observation_tail = 2;
constexpr std::size_t action_count = 6;
constexpr double milliseconds = 1000;
constexpr double quantile_50 = 0.5;
constexpr double quantile_95 = 0.95;
constexpr double quantile_99 = 0.99;
constexpr uint64_t hash_offset = 14695981039346656037ULL;
constexpr uint64_t hash_prime = 1099511628211ULL;
constexpr int report_precision = 12;

struct Options {
    std::size_t environments = fixture_environments;
    std::size_t assets = fixture_assets;
    std::size_t sessions = fixture_sessions;
    std::size_t workers = 1;
    bool reference = false;
};

std::size_t number(std::string_view input) {
    std::size_t value = 0;
    const auto parsed = std::from_chars(input.begin(), input.end(), value);
    if (parsed.ec != std::errc{} || parsed.ptr != input.end()) {
        throw std::invalid_argument("Se necesita un entero positivo");
    }
    return value;
}

Options parse(std::span<char*> arguments) {
    Options result;
    for (std::size_t index = 1; index < arguments.size(); ++index) {
        const std::string_view name(arguments[index]);
        if (name == "--reference") {
            result.reference = true;
        } else if (++index == arguments.size()) {
            throw std::invalid_argument("Falta el valor de un argumento");
        } else if (name == "--environments") {
            result.environments = number(arguments[index]);
        } else if (name == "--assets") {
            result.assets = number(arguments[index]);
        } else if (name == "--sessions") {
            result.sessions = number(arguments[index]);
        } else if (name == "--workers") {
            result.workers = number(arguments[index]);
        } else {
            throw std::invalid_argument("Argumento desconocido");
        }
    }
    if (result.environments == 0 || result.environments > maximum_environments ||
        result.assets == 0 || result.assets > maximum_assets ||
        result.sessions < 4 || result.sessions > maximum_sessions ||
        result.assets * result.sessions > maximum_rows ||
        (result.workers != 1 && result.workers != 2 && result.workers != 4 &&
         result.workers != maximum_batch_workers) ||
        (result.reference && result.workers != 1)) {
        throw std::invalid_argument("El diagnóstico excede sus límites");
    }
    return result;
}

std::shared_ptr<const MarketTape> fixture(const Options& options) {
    auto result = std::make_shared<MarketTape>();
    result->currency = "USD";
    result->domain = "synthetic";
    result->partition = "train";
    result->parent_id = "diagnostic-analytic";
    result->source_sha256.assign(digest_length, 'a');
    for (std::size_t asset = 0; asset < options.assets; ++asset) {
        result->assets.push_back("FIC" + std::to_string(asset));
    }
    result->prices.reserve(options.sessions * options.assets * price_columns);
    result->scores.reserve(options.sessions * options.assets);
    for (std::size_t time = 0; time < options.sessions; ++time) {
        result->open_times.push_back(static_cast<int64_t>(2 * time));
        result->close_times.push_back(static_cast<int64_t>(2 * time + 1));
        result->prediction_times.push_back(static_cast<int64_t>(2 * time + 1));
        for (std::size_t asset = 0; asset < options.assets; ++asset) {
            const auto phase = static_cast<double>(time + asset) * step_scale;
            const auto opening = price_scale + static_cast<double>(asset) + std::sin(phase);
            const auto closing = opening + std::cos(phase) * signal_scale;
            result->prices.insert(result->prices.end(), {opening, std::max(opening, closing),
                std::min(opening, closing), closing, volume});
            result->scores.push_back(signal_scale * std::sin(phase));
        }
    }
    return result;
}

double seconds(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

uint64_t rss_bytes() {
    std::ifstream input("/proc/self/status");
    for (std::string key; input >> key;) {
        if (key == "VmHWM:") {
            uint64_t value = 0;
            input >> value;
            constexpr uint64_t bytes_per_kibibyte = 1024;
            return value * bytes_per_kibibyte;
        }
        input.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
    return 0;
}

void mix(uint64_t& hash, double value) {
    hash = (hash ^ std::bit_cast<uint64_t>(value)) * hash_prime;
}

void run(const Options& options) {
    const auto total_start = Clock::now();
    const auto data = fixture(options);
    const auto width = options.assets * financial_columns + observation_tail;
    std::vector<FinancialSession> references;
    std::unique_ptr<FinancialBatch> batch;
    if (options.reference) {
        references.reserve(options.environments);
        for (std::size_t lane = 0; lane < options.environments; ++lane) {
            references.emplace_back(data);
        }
    } else {
        batch = std::make_unique<FinancialBatch>(
            std::vector<BatchInput>(options.environments, {data, {}, {}}), options.workers);
    }
    const auto setup_seconds = seconds(total_start);
    std::vector<uint8_t> actions(options.environments);
    std::vector<double> rewards(options.environments);
    std::vector<float> observations(options.environments * width);
    const auto one_step = [&] {
        if (batch) {
            rewards = batch->step(actions).rewards;
            std::copy(batch->observations().begin(), batch->observations().end(), observations.begin());
        } else {
            for (std::size_t lane = 0; lane < references.size(); ++lane) {
                rewards[lane] = references[lane].step(actions[lane]).reward;
                const auto observation = references[lane].observation();
                std::copy(observation.begin(), observation.end(), observations.begin() +
                          static_cast<std::ptrdiff_t>(lane * width));
            }
        }
    };
    // Un paso calienta código y pool. Después se vuelve al mismo estado inicial.
    one_step();
    if (batch) {
        std::vector<std::size_t> indices(options.environments);
        std::iota(indices.begin(), indices.end(), 0);
        batch->reset(indices);
    } else {
        for (auto& reference : references) {
            reference = FinancialSession(data);
        }
    }
    uint64_t hash = hash_offset;
    std::vector<double> latencies;
    latencies.reserve(options.sessions - 1);
    const auto rollout_start = Clock::now();
    for (std::size_t time = 0; time + 1 < options.sessions; ++time) {
        for (std::size_t lane = 0; lane < options.environments; ++lane) {
            actions[lane] = static_cast<uint8_t>((time + lane) % action_count);
        }
        const auto step_start = Clock::now();
        one_step();
        latencies.push_back(seconds(step_start) * milliseconds);
        for (const auto reward : rewards) {
            mix(hash, reward);
        }
        for (const auto value : observations) {
            mix(hash, static_cast<double>(value));
        }
    }
    const auto rollout_seconds = seconds(rollout_start);
    const auto recovery_start = Clock::now();
    if (batch) {
        batch->restore(batch->snapshot());
    } else {
        for (auto& reference : references) {
            reference.restore(reference.snapshot());
        }
    }
    const auto recovery_seconds = seconds(recovery_start);
    std::sort(latencies.begin(), latencies.end());
    const auto quantile = [&](double q) {
        return latencies[static_cast<std::size_t>(std::ceil(q * static_cast<double>(latencies.size()))) - 1];
    };
    const auto transitions = options.environments * (options.sessions - 1);
    std::cout << std::setprecision(report_precision)
              << "{\"domain\":\"technical\",\"reference\":" << (options.reference ? "true" : "false")
              << ",\"environments\":" << options.environments << ",\"assets\":" << options.assets
              << ",\"sessions\":" << options.sessions << ",\"workers\":" << options.workers
              << ",\"transitions\":" << transitions << ",\"checksum\":\"" << hash << '"'
              << ",\"setup_seconds\":" << setup_seconds << ",\"rollout_seconds\":" << rollout_seconds
              << ",\"recovery_seconds\":" << recovery_seconds << ",\"total_seconds\":" << seconds(total_start)
              << ",\"transitions_per_second\":" << static_cast<double>(transitions) / rollout_seconds
              << ",\"step_p50_ms\":" << quantile(quantile_50) << ",\"step_p95_ms\":" << quantile(quantile_95)
              << ",\"step_p99_ms\":" << quantile(quantile_99) << ",\"peak_rss_bytes\":" << rss_bytes()
              << ",\"reserved_payload_bytes\":" << (batch ? batch->reserved_payload_bytes() : 0)
              << ",\"observation_output_bytes\":" << transitions * width * sizeof(float)
              << ",\"source_sha256\":\"" << MARS_TITAN_NATIVE_SOURCE_SHA256 << '"'
              << ",\"build_sha256\":\"" << MARS_TITAN_NATIVE_BUILD_SHA256 << "\"}\n";
}
}

int main(int argc, char** argv) {
    try {
        run(parse(std::span<char*>(argv, static_cast<std::size_t>(argc))));
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
