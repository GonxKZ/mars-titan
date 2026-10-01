#include "mars_titan/simulation.h"

#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <thread>

namespace {
constexpr std::size_t error_capacity = 256;
constexpr int repetitions = 128;
constexpr double expected_cash = 598.6;
constexpr double tolerance = 1e-10;
constexpr std::array<double, 2> expected_positions{100, 20};
} // namespace

int main() {
    std::atomic<bool> valid{true};
    {
        std::array<std::jthread, 4> workers;
        for (auto& worker : workers) {
            worker = std::jthread([&valid]() {
                const std::array<uint32_t, 2> currencies{0, 0};
                const std::array<double, 2> lots{1, 1};
                const std::array<uint8_t, 2> retired{0, 0};
                const std::array<double, 10> prices{10, 10, 10, 10, 10000, 20, 20, 20, 20, 10000};
                const std::array<mt_position_v1, 2> positions{
                    {{0, 100, 1000, 1}, {0, 20, 1000, 1}}};
                const std::array<mt_account_v1, 1> accounts{{{2000, 0, 0, 0, 2000}}};
                std::array<mt_position_v1, 2> next_positions{};
                std::array<mt_account_v1, 1> next_accounts{};
                std::array<mt_trade_v1, 2> trades{};
                std::array<char, error_capacity> error{};
                for (int iteration = 0; iteration < repetitions; ++iteration) {
                    const int result = mt_simulation_step_v1(
                        2, 1, currencies.data(), lots.data(), retired.data(), prices.data(),
                        positions.data(), accounts.data(), 0.001, 0.1, 1, 2, 3,
                        next_positions.data(), next_accounts.data(), trades.data(), error.data(),
                        error.size());
                    if (result != MT_SIM_OK ||
                        std::abs(next_accounts[0].cash - expected_cash) > tolerance ||
                        next_positions[0].quantity != expected_positions[0] ||
                        next_positions[1].quantity != expected_positions[1]) {
                        valid.store(false, std::memory_order_relaxed);
                    }
                }
            });
        }
    }
    if (!valid.load(std::memory_order_relaxed)) {
        std::cerr << "Las llamadas independientes no conservan sus cuentas\n";
        return 1;
    }
    std::cout << "Cuatro llamadas concurrentes independientes comprobadas\n";
    return 0;
}
