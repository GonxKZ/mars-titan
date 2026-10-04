#include "mars_titan/episodic_memory.hpp"

#include <ATen/Parallel.h>

#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string_view>
#include <utility>
#include <vector>

namespace {
using namespace mars_titan::learning;
using Clock = std::chrono::steady_clock;
constexpr std::size_t default_iterations = 4096;
constexpr std::size_t maximum_iterations = 65536;
constexpr uint64_t fixture_seed = 42;
constexpr double fixture_reward = 0.01;
constexpr int64_t query_cutoff = 1'000'000;
constexpr uint64_t hash_offset = 14695981039346656037ULL;
constexpr uint64_t hash_prime = 1099511628211ULL;

struct Options {
    std::size_t capacity = episodic_memory_capacity;
    std::size_t iterations = default_iterations;
    std::string_view mode = "step";
};

std::size_t number(std::string_view text) {
    std::size_t result = 0;
    const auto parsed = std::from_chars(text.begin(), text.end(), result);
    if (parsed.ec != std::errc{} || parsed.ptr != text.end()) {
        throw std::invalid_argument("El argumento necesita un entero positivo");
    }
    return result;
}

Options parse(std::span<char*> arguments) {
    Options options;
    for (std::size_t index = 1; index < arguments.size(); ++index) {
        const std::string_view name(arguments[index]);
        if (++index == arguments.size()) {
            throw std::invalid_argument("Falta el valor del argumento");
        }
        if (name == "--capacity") {
            options.capacity = number(arguments[index]);
        } else if (name == "--iterations") {
            options.iterations = number(arguments[index]);
        } else if (name == "--mode") {
            options.mode = arguments[index];
        } else {
            throw std::invalid_argument("Argumento desconocido");
        }
    }
    if (options.capacity == 0 || options.capacity > episodic_memory_capacity ||
        options.iterations == 0 || options.iterations > maximum_iterations ||
        (options.mode != "query" && options.mode != "prepared" && options.mode != "step")) {
        throw std::invalid_argument("El diagnóstico excede sus límites o utiliza un modo desconocido");
    }
    return options;
}

MemoryRecord record(uint64_t id) {
    MemoryRecord result;
    result.id = id;
    result.decision_at = static_cast<int64_t>(id * 2);
    result.available_at = result.decision_at;
    result.maturity_at = result.decision_at + 1;
    result.reward_valid = true;
    result.reward = fixture_reward;
    for (std::size_t column = 0; column < episodic_memory_width; ++column) {
        result.key.at(column) = static_cast<float>(std::sin(static_cast<double>(id * (column + 1))));
        result.value.at(column) = static_cast<float>(std::cos(static_cast<double>(id + column)));
    }
    return result;
}

void mix(uint64_t& hash, const MemoryQuery& query) {
    hash = (hash ^ query.count) * hash_prime;
    for (std::size_t index = 0; index < query.count; ++index) {
        const auto& neighbor = query.neighbors.at(index);
        hash = (hash ^ neighbor.record.id) * hash_prime;
        hash = (hash ^ std::bit_cast<uint64_t>(neighbor.similarity)) * hash_prime;
    }
}

void run(const Options& options) {
    EpisodicMemory memory({"diagnostic", "train", "fold-0", "fixed-v1", 0}, fixture_seed,
                           options.capacity);
    for (uint64_t id = 1; id <= options.capacity; ++id) {
        const auto value = record(id);
        auto candidate = memory.prepare_write(value, value.maturity_at);
        if (!memory.commit(std::move(candidate))) {
            throw std::runtime_error("No se confirmó el estado inicial");
        }
    }
    std::vector<MemoryRecord> inputs;
    inputs.reserve(options.iterations);
    for (uint64_t index = 0; index < options.iterations; ++index) {
        inputs.push_back(record(options.capacity + index + 1));
    }
    static_cast<void>(memory.query(inputs.front().key, query_cutoff));
    uint64_t digest = hash_offset;
    const auto start = Clock::now();
    for (const auto& value : inputs) {
        if (options.mode == "query") {
            mix(digest, memory.query(value.key, query_cutoff));
        } else {
            auto candidate = memory.prepare_write(value, value.maturity_at);
            mix(digest, memory.query_prepared(value.key, query_cutoff, candidate));
            if (options.mode == "step" && !memory.commit(std::move(candidate))) {
                throw std::runtime_error("No se confirmó la escritura medida");
            }
        }
    }
    const double elapsed = std::chrono::duration<double>(Clock::now() - start).count();
    constexpr int decimal_digits = 17;
    std::cout << std::setprecision(decimal_digits)
              << "{\"capacity\":" << options.capacity << ",\"width\":" << episodic_memory_width
              << ",\"iterations\":" << options.iterations << ",\"mode\":\"" << options.mode
              << "\",\"seconds\":" << elapsed
              << ",\"operations_per_second\":" << static_cast<double>(options.iterations) / elapsed
              << ",\"digest\":\"" << digest << "\",\"seen\":" << memory.seen() << "}\n";
}
} // namespace

int main(int argc, char** argv) {
    try {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        run(parse({argv, static_cast<std::size_t>(argc)}));
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
