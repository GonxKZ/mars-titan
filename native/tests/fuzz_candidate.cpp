#include "mars_titan/candidate.hpp"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>

// El generador limita las formas y reserva los bytes para casos de frontera.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    if (size < 3) {
        return 0;
    }
    using namespace mars_titan::candidate;
    at::set_num_threads(1);
    const at::NoGradGuard guard;
    const std::span<const uint8_t> bytes(data, size);
    Config config;
    config.dimensions = {1, 1, 1, 1, 1};
    config.normalization_id = "fuzz-synthetic-v1";
    config.max_batch = 2;
    config.max_episodes = 2;
    const Candidate model(config);
    const int64_t rows = bytes[0] % 3;
    const auto value = static_cast<float>(bytes[1]) / 255.F;
    Inputs inputs{at::full({rows, 64, 1}, value),
                  at::ones({rows, 1}),
                  at::ones({rows, 1}),
                  at::ones({rows, 1}),
                  at::ones({rows, 1}),
                  at::ones({rows, 5}, at::kBool)};
    const bool incomplete = (bytes[2] & 1U) != 0;
    const bool nonfinite = (bytes[2] & 2U) != 0;
    if (incomplete) {
        inputs.presence.fill_(false);
    }
    if (nonfinite) {
        inputs.macro.fill_(std::numeric_limits<float>::infinity());
    }
    const bool invalid = rows == 0 || incomplete || nonfinite;
    try {
        const auto prediction = model.forward(inputs, model.empty_memory());
        if (invalid || !at::isfinite(prediction.quantiles).all().item<bool>() ||
            !(prediction.quantiles.slice(1, 1) >= prediction.quantiles.slice(1, 0, 4))
                 .all()
                 .item<bool>()) {
            __builtin_trap();
        }
    } catch (const std::invalid_argument&) {
        if (!invalid) {
            __builtin_trap();
        }
    }
    return 0;
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
