#include "mars_titan/financial_session.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <limits>
#include <memory>
#include <span>

namespace {
constexpr std::size_t maximum_fuzz_assets = 4;
constexpr std::size_t maximum_fuzz_steps = 8;
constexpr std::size_t digest_length = 64;
constexpr std::size_t action_count = 6;
constexpr double volume_multiplier = 10;
constexpr double initial_capital = 1000;
constexpr double unknown = std::numeric_limits<double>::quiet_NaN();

bool same_number(double left, double right) {
    return left == right || (std::isnan(left) && std::isnan(right));
}

void require_unchanged(const mars_titan::simulation::SessionSnapshot& before,
                       const mars_titan::simulation::SessionSnapshot& after) {
    if (before.source_sha256 != after.source_sha256 || before.parameters != after.parameters ||
        before.cursor != after.cursor || before.done != after.done ||
        before.account.cash != after.account.cash || before.account.costs != after.account.costs ||
        before.account.turnover != after.account.turnover ||
        before.account.receivable != after.account.receivable ||
        !same_number(before.account.nav, after.account.nav) || before.retired != after.retired ||
        before.applied_actions != after.applied_actions || before.held_order != after.held_order ||
        before.positions.size() != after.positions.size() ||
        before.receivables.size() != after.receivables.size() ||
        before.peak_nav != after.peak_nav ||
        before.max_drawdown != after.max_drawdown ||
        before.unfilled_order_observations != after.unfilled_order_observations) {
        std::abort();
    }
    for (std::size_t asset = 0; asset < before.positions.size(); ++asset) {
        const auto& left = before.positions[asset];
        const auto& right = after.positions[asset];
        if (left.quantity != right.quantity || !same_number(left.target, right.target) ||
            !same_number(left.capacity, right.capacity) || left.decision_at != right.decision_at) {
            std::abort();
        }
    }
    for (std::size_t index = 0; index < before.receivables.size(); ++index) {
        const auto& left = before.receivables[index];
        const auto& right = after.receivables[index];
        if (left.action != right.action || left.pay_at != right.pay_at ||
            left.amount != right.amount) {
            std::abort();
        }
    }
}
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    if (size < maximum_fuzz_assets) {
        return 0;
    }
    const std::span bytes{data, size};
    const auto value = [&bytes](std::size_t index) { return bytes[index % bytes.size()]; };
    const auto assets = 1 + value(0) % maximum_fuzz_assets;
    const auto sessions = 2 + value(1) % maximum_fuzz_steps;
    auto tape = std::make_shared<mars_titan::simulation::MarketTape>();
    tape->currency = "USD";
    tape->domain = "synthetic";
    tape->partition = "train";
    tape->parent_id = "referencia-fuzz";
    tape->source_sha256.assign(digest_length, 'a');
    for (std::size_t asset = 0; asset < assets; ++asset) {
        tape->assets.emplace_back(1, static_cast<char>('A' + asset));
    }
    for (std::size_t session = 0; session < sessions; ++session) {
        const auto at = static_cast<int64_t>(session * 2);
        tape->open_times.push_back(at);
        tape->close_times.push_back(at + 1);
        tape->prediction_times.push_back(at + 1);
        for (std::size_t asset = 0; asset < assets; ++asset) {
            const auto offset = session * assets + asset;
            const double price = static_cast<double>(value(offset)) + 1;
            const double close = value(offset + 1) == 0 ? unknown : price;
            const double opening = value(offset + 2) == 0 ? unknown : price;
            const double volume = static_cast<double>(value(offset + 3)) * volume_multiplier;
            tape->prices.insert(tape->prices.end(), {opening, price, price, close, volume});
            tape->scores.push_back(static_cast<double>(value(offset)) - volume_multiplier);
        }
    }
    mars_titan::simulation::Parameters parameters;
    parameters.capital = initial_capital;
    parameters.cost_bps = value(2);
    parameters.participation = 1;
    try {
        mars_titan::simulation::FinancialSession session(tape, parameters);
        for (std::size_t step = 0; step + 1 < sessions && !session.done(); ++step) {
            const auto before = session.snapshot();
            try {
                static_cast<void>(session.step(static_cast<uint8_t>(value(step) % action_count)));
            } catch (const std::exception&) {
                require_unchanged(before, session.snapshot());
                return 0;
            }
            const auto saved = session.snapshot();
            mars_titan::simulation::FinancialSession restored(tape, parameters);
            restored.restore(saved);
            require_unchanged(saved, restored.snapshot());
            if (session.observation() != restored.observation()) {
                std::abort();
            }
            auto damaged = saved;
            switch (value(step + 1) % maximum_fuzz_assets) {
            case 0:
                damaged.cursor = sessions;
                break;
            case 1:
                damaged.account.nav = std::numeric_limits<double>::infinity();
                break;
            case 2:
                damaged.held_order.push_back(assets);
                break;
            default:
                damaged.positions.front().quantity = -1;
                break;
            }
            bool rejected = false;
            try {
                restored.restore(damaged);
            } catch (const std::exception&) {
                rejected = true;
            }
            if (!rejected) {
                std::abort();
            }
            require_unchanged(saved, restored.snapshot());
        }
    } catch (const std::exception&) {
        std::abort();
    }
    return 0;
}
