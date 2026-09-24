#include "mars_titan/simulation_files.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <span>
#include <stdexcept>
#include <string>

namespace {
constexpr std::size_t maximum_input_bytes = 4096;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t* data, std::size_t size) {
    if (size > maximum_input_bytes) {
        return 0;
    }
    std::string input;
    input.reserve(size);
    for (const auto byte : std::span(data, size)) {
        input.push_back(static_cast<char>(byte));
    }
    using namespace mars_titan::simulation;
    nlohmann::json value;
    try {
        value = parse_bounded_json(input);
    } catch (const nlohmann::json::exception&) {
        return 0;
    } catch (const std::invalid_argument&) {
        return 0;
    }
    // La admisión rechaza entradas incompletas. Una entrada admitida debe conservar sus campos.
    nlohmann::json canonical;
    try {
        canonical = snapshot_json(read_snapshot(value));
    } catch (const nlohmann::json::exception&) {
        return 0;
    } catch (const std::invalid_argument&) {
        return 0;
    }
    const auto restored = read_snapshot(parse_bounded_json(canonical.dump()));
    if (snapshot_json(restored) != canonical) {
        std::abort();
    }
    return 0;
}
