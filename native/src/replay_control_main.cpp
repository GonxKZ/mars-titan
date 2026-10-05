#include "mars_titan/replay_schedule.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
using namespace mars_titan::controls;
using Clock = std::chrono::steady_clock;
constexpr double learning_rate = 0.03;
std::size_t parse(std::string_view value, std::size_t limit) {
    std::size_t parsed = 0;
    const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), parsed);
    if (error != std::errc{} || end != value.data() + value.size() || parsed == 0 || parsed > limit)
        throw std::invalid_argument(
            "Los argumentos deben ser enteros positivos dentro del presupuesto");
    return parsed;
}
double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}
struct Result {
    double build_ms = 0;
    double replay_ms = 0;
    double weight = 0;
    std::size_t updates = 0;
    std::size_t index_bytes = 0;
    std::size_t checkpoint_bytes = 0;
};
Result run(const ReplayScheduleConfig& settings, const std::vector<MatureEpisode>& rows) {
    const auto start = Clock::now();
    ReplaySchedule schedule(settings, rows);
    const auto built = Clock::now();
    double weight = 0;
    while (schedule.cursor() < schedule.size()) {
        const auto batch = schedule.prepare();
        double target = 0;
        for (const auto index : batch.indices)
            target += index < rows.size() / 2 ? -1.0 : 1.0;
        target /= static_cast<double>(batch.indices.size());
        weight += learning_rate * (target - weight);
        schedule.commit(batch.token);
    }
    const auto end = Clock::now();
    const auto checkpoint = serialize_schedule(schedule.snapshot());
    return {milliseconds(start, built), milliseconds(built, end), weight,
            schedule.updates(),         schedule.index_bytes(),   checkpoint.size()};
}
} // namespace
int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string_view(argv[1]) == "--help") {
            std::cout << "Uso: mars-titan-replay-control [episodios exposiciones lote semilla "
                         "repeticiones]\n"
                         "Control escalar de dos objetivos temporales, sin datos financieros.\n";
            return 0;
        }
        if (argc > 6)
            throw std::invalid_argument("El control admite como máximo cinco argumentos");
        const auto count = argc > 1 ? parse(argv[1], 1U << 20) : 1024;
        const auto exposures = argc > 2 ? parse(argv[2], 1024) : 4;
        const auto batch = argc > 3 ? parse(argv[3], 1U << 20) : 32;
        const auto seed = argc > 4 ? parse(argv[4], 1U << 30) : 71;
        const auto repetitions = argc > 5 ? parse(argv[5], 101) : 7;
        if (count > (1U << 24) / exposures)
            throw std::invalid_argument(
                "El producto de episodios y exposiciones excede el presupuesto");
        std::vector<MatureEpisode> rows;
        rows.reserve(count);
        for (std::size_t index = 0; index < count; ++index)
            rows.push_back({index + 1, index, index + 1, static_cast<uint32_t>(exposures)});
        std::cout << std::setprecision(10) << "{\"schema\":1,\"episodes\":" << count
                  << ",\"exposures_per_episode\":" << exposures << ",\"batch_size\":" << batch
                  << ",\"seed\":" << seed << ",\"learning_rate\":" << learning_rate
                  << ",\"warmups\":1,\"repetitions\":" << repetitions << ",\"results\":[";
        bool first = true;
        for (const auto order : {ReplayOrder::uniform, ReplayOrder::recent, ReplayOrder::spaced}) {
            const ReplayScheduleConfig settings{.run_id = "two-target-control-v1",
                                                .order = order,
                                                .cutoff = count,
                                                .order_seed = seed,
                                                .batch_size = batch,
                                                .minimum_distance = count,
                                                .max_exposures = 1U << 24};
            static_cast<void>(run(settings, rows));
            std::vector<double> elapsed;
            double build = 0;
            double replay = 0;
            Result result;
            for (std::size_t repetition = 0; repetition < repetitions; ++repetition) {
                result = run(settings, rows);
                build += result.build_ms;
                replay += result.replay_ms;
                elapsed.push_back(result.build_ms + result.replay_ms);
            }
            std::sort(elapsed.begin(), elapsed.end());
            if (!first)
                std::cout << ',';
            first = false;
            const auto total = static_cast<double>(count * exposures);
            std::cout << "{\"order\":\"" << replay_order_name(order)
                      << "\",\"updates\":" << result.updates
                      << ",\"index_bytes\":" << result.index_bytes
                      << ",\"checkpoint_bytes\":" << result.checkpoint_bytes
                      << ",\"build_mean_ms\":" << build / static_cast<double>(repetitions)
                      << ",\"replay_mean_ms\":" << replay / static_cast<double>(repetitions)
                      << ",\"total_min_ms\":" << elapsed.front()
                      << ",\"total_median_ms\":" << elapsed[elapsed.size() / 2]
                      << ",\"total_max_ms\":" << elapsed.back() << ",\"exposures_per_second\":"
                      << total * 1000.0 / elapsed[elapsed.size() / 2]
                      << ",\"final_weight\":" << result.weight
                      << ",\"old_target_mae\":" << std::abs(-1.0 - result.weight)
                      << ",\"new_target_mae\":" << std::abs(1.0 - result.weight) << '}';
        }
        std::cout << "]}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
