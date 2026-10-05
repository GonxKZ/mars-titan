#ifndef MARS_TITAN_MEMORY_STRESS_INTERNAL_HPP
#define MARS_TITAN_MEMORY_STRESS_INTERNAL_HPP

#include "mars_titan/memory_stress.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <locale>
#include <random>
#include <sstream>
#include <stdexcept>

namespace mars_titan::stress::detail {
using Json = nlohmann::json;
using Engine = std::mt19937_64;
inline constexpr std::size_t maximum_steps = 1'000'000;
inline constexpr std::size_t maximum_delay = 4096;
inline constexpr std::size_t maximum_rng_bytes = 8192;
inline constexpr int maximum_json_depth = 16;
inline constexpr uint64_t hash_offset = 14695981039346656037ULL;
inline constexpr uint64_t hash_prime = 1099511628211ULL;

inline void require(bool condition, std::string_view message) {
    if (!condition) {
        throw std::invalid_argument(std::string(message));
    }
}
inline Json parse(std::string_view bytes) {
    require(!bytes.empty() && bytes.size() <= maximum_stress_archive_bytes,
            "El archivo supera el límite o está vacío");
    return Json::parse(bytes, [](int depth, Json::parse_event_t, Json&) {
        require(depth <= maximum_json_depth, "El archivo supera el límite de anidamiento");
        return true;
    });
}
inline uint64_t integer(const Json& value, uint64_t maximum = maximum_steps) {
    require(value.is_number_unsigned() || (value.is_number_integer() && value.get<int64_t>() >= 0),
            "Se requiere un entero no negativo");
    const auto result = value.get<uint64_t>();
    require(result <= maximum, "El entero excede su límite");
    return result;
}
inline double number(const Json& value) {
    require(value.is_number(), "Se requiere un número finito");
    const auto result = value.get<double>();
    require(std::isfinite(result), "Se requiere un número finito");
    return result;
}
inline Engine engine(uint64_t seed, uint32_t stream) {
    constexpr unsigned word_bits = 32;
    std::seed_seq sequence{static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> word_bits),
                           stream};
    return Engine(sequence);
}
inline std::string encode_rng(const Engine& rng) {
    std::ostringstream stream;
    stream.imbue(std::locale::classic());
    stream << rng;
    require(stream.good(), "No se pudo guardar el RNG");
    return stream.str();
}
inline Engine decode_rng(const Json& value) {
    const auto text = value.get<std::string>();
    require(!text.empty() && text.size() <= maximum_rng_bytes, "Estado RNG fuera de límite");
    Engine result;
    std::istringstream stream(text);
    stream.imbue(std::locale::classic());
    stream >> result;
    require(!stream.fail() && encode_rng(result) == text, "El estado RNG no es canónico");
    return result;
}
inline void valid_record(const learning::MemoryRecord& record) {
    require(record.id > 0 && record.id <= maximum_steps && record.decision_at >= 0 &&
                record.available_at >= 0 && record.available_at <= record.decision_at &&
                record.maturity_at > record.decision_at && record.label_valid &&
                std::isfinite(record.label) && std::isfinite(record.reward),
            "El registro necesita identidad, disponibilidad y etiqueta madura válidas");
    double norm = 0;
    for (std::size_t index = 0; index < learning::episodic_memory_width; ++index) {
        const auto key = static_cast<double>(record.key.at(index));
        require(std::isfinite(key) && std::isfinite(record.value.at(index)),
                "El registro contiene NaN o infinito");
        norm += key * key;
    }
    require(norm > 0, "La clave no puede ser nula");
}
inline learning::MemoryVector normalized(const learning::MemoryVector& key) {
    double norm = 0;
    for (const float component : key) {
        require(std::isfinite(component), "La clave contiene NaN o infinito");
        const auto value = static_cast<double>(component);
        norm += value * value;
    }
    require(norm > 0, "La clave no puede ser nula");
    norm = std::sqrt(norm);
    learning::MemoryVector result{};
    std::transform(key.begin(), key.end(), result.begin(), [norm](float value) {
        return static_cast<float>(static_cast<double>(value) / norm);
    });
    return result;
}
inline Json record_json(const learning::MemoryRecord& record) {
    return {{"id", record.id},
            {"decision_at", record.decision_at},
            {"available_at", record.available_at},
            {"maturity_at", record.maturity_at},
            {"key", record.key},
            {"value", record.value},
            {"label", record.label},
            {"label_valid", record.label_valid},
            {"reward", record.reward},
            {"reward_valid", record.reward_valid}};
}
inline learning::MemoryRecord read_record(const Json& value) {
    learning::MemoryRecord result;
    result.id = integer(value.at("id"));
    constexpr uint64_t maximum_time = 2 * maximum_steps + maximum_delay;
    result.decision_at = static_cast<int64_t>(integer(value.at("decision_at"), maximum_time));
    result.available_at = static_cast<int64_t>(integer(value.at("available_at"), maximum_time));
    result.maturity_at = static_cast<int64_t>(integer(value.at("maturity_at"), maximum_time));
    const auto vector = [](const Json& array) {
        require(array.is_array() && array.size() == learning::episodic_memory_width,
                "El vector tiene una dimensión incompatible");
        learning::MemoryVector output{};
        for (std::size_t index = 0; index < output.size(); ++index) {
            const double component = number(array.at(index));
            require(std::abs(component) <= static_cast<double>(std::numeric_limits<float>::max()),
                    "El componente excede FP32");
            output.at(index) = static_cast<float>(component);
        }
        return output;
    };
    result.key = vector(value.at("key"));
    result.value = vector(value.at("value"));
    result.label = number(value.at("label"));
    result.reward = number(value.at("reward"));
    result.label_valid = value.at("label_valid").get<bool>();
    result.reward_valid = value.at("reward_valid").get<bool>();
    valid_record(result);
    return result;
}
} // namespace mars_titan::stress::detail
#endif
