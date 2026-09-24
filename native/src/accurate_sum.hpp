#ifndef MARS_TITAN_ACCURATE_SUM_HPP
#define MARS_TITAN_ACCURATE_SUM_HPP

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <span>
#include <utility>

namespace mars_titan::simulation {
inline constexpr std::size_t partial_capacity = 128;
inline constexpr int binary64_precision = 53;
inline constexpr double cash_roundoff_ulps = 8;

/* Parciales no solapados mediante TwoSum, con presupuesto fijo para exponentes binary64. */
class AccurateSum {
  public:
    [[nodiscard]] bool add(double value) noexcept {
        if (!std::isfinite(value)) {
            return false;
        }
        std::size_t retained = 0;
        const std::span partials{partials_};
        for (std::size_t index = 0; index < count_; ++index) {
            double other = partials[index];
            if (std::abs(value) < std::abs(other)) {
                std::swap(value, other);
            }
            const double high = value + other;
            const double low = other - (high - value);
            if (!std::isfinite(high)) {
                return false;
            }
            if (low != 0) {
                partials[retained++] = low;
            }
            value = high;
        }
        count_ = retained;
        if (value != 0) {
            if (count_ == partials_.size()) {
                return false;
            }
            partials[count_++] = value;
        }
        return true;
    }

    [[nodiscard]] double value() const noexcept {
        if (count_ == 0) {
            return 0;
        }
        auto remaining = count_ - 1;
        const std::span partials{partials_};
        double high = partials[remaining];
        double low = 0;
        while (remaining != 0) {
            const double previous = high;
            const double following = partials[--remaining];
            high = previous + following;
            low = following - (high - previous);
            if (low != 0) {
                break;
            }
        }
        // Redondear al par cuando los parciales restantes están en el mismo lado del empate.
        if (remaining != 0 && ((low < 0 && partials[remaining - 1] < 0) ||
                               (low > 0 && partials[remaining - 1] > 0))) {
            const double adjustment = low * 2;
            const double rounded = high + adjustment;
            if (rounded - high == adjustment) {
                high = rounded;
            }
        }
        return high;
    }

  private:
    std::array<double, partial_capacity> partials_{};
    std::size_t count_ = 0;
};

struct CashMovements {
    AccurateSum balance;
    AccurateSum magnitude;

    [[nodiscard]] bool add(double value) noexcept {
        return balance.add(value) && magnitude.add(std::abs(value));
    }

    [[nodiscard]] double reconciled() const noexcept {
        const double result = balance.value();
        const double size = magnitude.value();
        int exponent = 0;
        std::frexp(size, &exponent);
        const double ulp = size == 0 ? std::numeric_limits<double>::denorm_min()
                                     : std::max(std::numeric_limits<double>::denorm_min(),
                                                std::ldexp(1.0, exponent - binary64_precision));
        return std::abs(result) <= cash_roundoff_ulps * ulp ? 0 : result;
    }
};
} // namespace mars_titan::simulation

#endif
