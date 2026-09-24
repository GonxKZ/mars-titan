#include "mars_titan/simulation.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <span>

namespace {
constexpr std::size_t asset_capacity = 4;
constexpr std::size_t account_count = 2;
constexpr std::size_t price_width = 5;
constexpr std::size_t price_columns = 4;
constexpr std::size_t error_capacity = 256;
constexpr uint32_t rejected_asset_count = std::numeric_limits<uint32_t>::max();
constexpr unsigned missing_open_mask = 16;
constexpr unsigned missing_close_mask = 32;
constexpr double volume_scale = 10;
constexpr double known_capacity = 100;
constexpr double rate = 0.001;
constexpr double participation = 0.1;
constexpr std::size_t volume_input = 5;
constexpr std::size_t quantity_input = 6;
constexpr std::size_t target_input = 7;
constexpr std::size_t capacity_input = 8;
constexpr std::size_t time_input = 9;
constexpr double unknown = std::numeric_limits<double>::quiet_NaN();
constexpr mt_account_v1 initial_account{1000, 0, 0, 0, 1000};
} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    if (size < asset_capacity) {
        return 0;
    }
    const std::span bytes{data, size};
    const auto value = [&bytes](std::size_t index) { return bytes[index % bytes.size()]; };
    const uint32_t assets = 1 + static_cast<uint32_t>(value(0) % asset_capacity);
    std::array<uint32_t, asset_capacity> currencies{};
    std::array<double, asset_capacity> lots{};
    std::array<uint8_t, asset_capacity> retired{};
    std::array<double, asset_capacity * price_width> prices{};
    std::array<mt_position_v1, asset_capacity> positions{};
    std::array<mt_account_v1, account_count> accounts{};
    std::array<mt_position_v1, asset_capacity> next_positions{};
    std::array<mt_account_v1, account_count> next_accounts{};
    std::array<mt_trade_v1, asset_capacity> trades{};
    std::array<char, error_capacity> error{};
    for (std::size_t index = 0; index < assets; ++index) {
        currencies.at(index) = value(index + 1) % account_count;
        lots.at(index) = value(index + 2) == 0 ? 0 : 1;
        const double price = static_cast<double>(value(index + 3)) + 1;
        for (std::size_t column = 0; column < price_columns; ++column) {
            prices.at(index * price_width + column) = price;
        }
        prices.at(index * price_width + price_columns) = value(index + volume_input) * volume_scale;
        if ((value(index + price_columns) & missing_open_mask) != 0) {
            prices.at(index * price_width) = unknown;
        }
        if ((value(index + price_columns) & missing_close_mask) != 0) {
            prices.at(index * price_width + 3) = unknown;
        }
        positions.at(index) = {static_cast<double>(value(index + quantity_input)),
                               static_cast<double>(value(index + target_input)),
                               (value(index + capacity_input) & 1) != 0 ? unknown : known_capacity,
                               (value(index + time_input) & 1) != 0 ? 1 : 4};
    }
    accounts.fill(initial_account);
    const int result = mt_simulation_step_v1(
        (value(1) == 0 ? rejected_asset_count : assets), account_count, currencies.data(),
        lots.data(), retired.data(), prices.data(), positions.data(), accounts.data(), rate,
        participation, 1, 2, 3, next_positions.data(), next_accounts.data(), trades.data(),
        error.data(), error.size());
    if (result == MT_SIM_OK) {
        for (const auto& account : next_accounts) {
            if (!std::isfinite(account.cash) || account.cash < 0 || !std::isfinite(account.costs) ||
                (!std::isnan(account.nav) && (!std::isfinite(account.nav) || account.nav < 0))) {
                std::abort();
            }
        }
        for (std::size_t index = 0; index < assets; ++index) {
            if (!std::isfinite(next_positions.at(index).quantity) ||
                next_positions.at(index).quantity < 0) {
                std::abort();
            }
        }
    }
    return 0;
}
