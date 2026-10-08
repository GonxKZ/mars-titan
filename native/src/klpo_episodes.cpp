#include "mars_titan/klpo_episodes.hpp"
#include "accurate_sum.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <unordered_set>
#include <utility>

namespace mars_titan::learning {
namespace {
constexpr std::size_t maximum_width = 32768;
constexpr std::size_t maximum_identifier = 128;
constexpr std::size_t digest_size = 64;
constexpr double mass_tolerance = 1e-6;
constexpr int64_t final_test_start = 1'704'067'200'000'000;
constexpr std::string_view magic = "MTKLPO01";
constexpr unsigned bits_per_byte = 8;
constexpr unsigned integer_bits = std::numeric_limits<uint64_t>::digits;
constexpr unsigned float_bits = std::numeric_limits<uint32_t>::digits;
constexpr uint32_t byte_mask = 0xffU;
static_assert(sizeof(float) == sizeof(uint32_t) && sizeof(double) == sizeof(uint64_t));
static_assert(std::numeric_limits<float>::is_iec559 && std::numeric_limits<double>::is_iec559);

void require(bool condition, const char* message) {
    if (!condition) { throw std::invalid_argument(message); }
}

bool identifier(std::string_view value) {
    return !value.empty() && value.size() <= maximum_identifier &&
        std::ranges::all_of(value, [](char c) {
            return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == ':' || c == '/';
        });
}

bool digest(std::string_view value) {
    return value.size() == digest_size && std::ranges::all_of(value, [](char c) {
        return (c >= 'a' && c <= 'f') || (c >= '0' && c <= '9');
    });
}

void account(std::size_t count, std::size_t width, std::size_t limit, std::size_t& used) {
    require(used <= limit && (width == 0 || count <= (limit - used) / width),
            "El registro terminal supera su presupuesto");
    used += count * width;
}

void validate_header(const KlpoEpisodeBatch& batch) {
    require(identifier(batch.fold) && digest(batch.reference_sha256) &&
                std::isfinite(batch.beta) && batch.beta > 0 &&
                std::isfinite(batch.gamma) && batch.gamma > 0 && batch.gamma <= 1 &&
                batch.observation_width > 0 && batch.observation_width <= maximum_width &&
                batch.max_bytes > 0 && batch.max_bytes <= maximum_klpo_record_bytes &&
                !batch.episodes.empty() && batch.episodes.size() <= maximum_klpo_episodes,
            "La identidad, geometría o configuración terminal no es válida");
}

void validate_spec(const KlpoEpisodeSpec& spec) {
    require(identifier(spec.id) && digest(spec.source_sha256) &&
                (spec.context_sha256.empty() || digest(spec.context_sha256)) &&
                spec.partition == "train" && spec.close_times.size() >= 2 &&
                spec.close_times.size() <= maximum_klpo_steps + 1 &&
                spec.forced_prefix < spec.close_times.size() && spec.close_times.front() > 0 &&
                spec.close_times.back() < final_test_start &&
                std::adjacent_find(spec.close_times.begin(), spec.close_times.end(),
                                   std::greater_equal<>()) == spec.close_times.end(),
            "El episodio no conserva su identidad, partición o calendario");
}

void validate_step(const KlpoEpisodeBatch& batch, const KlpoEpisodeRecord& episode,
                   const KlpoEpisodeStep& step, std::size_t cursor) {
    const auto& spec = episode.spec;
    require(step.cursor == cursor && step.decision_at == spec.close_times[cursor] &&
                step.outcome_at == spec.close_times[cursor + 1] &&
                step.observation.size() == batch.observation_width &&
                std::ranges::all_of(step.observation, [](float v) { return std::isfinite(v); }) &&
                step.action < step.behavior.size() && std::isfinite(step.reward) &&
                step.sampled == (cursor >= spec.forced_prefix),
            "La transición terminal no conserva ordinal, fecha, forma o acción");
    double mass = 0;
    for (const auto weight : step.behavior) {
        require(std::isfinite(weight) && (step.sampled ? weight > 0 : weight == 0),
                "La distribución histórica no conserva soporte o ausencia explícita");
        mass += static_cast<double>(weight);
    }
    require(step.sampled ? std::abs(mass - 1) <= mass_tolerance : step.action == 1,
            "La acción forzada o la masa histórica no corresponde al contrato");
    const bool last_time = cursor + 2 == spec.close_times.size();
    if (!step.valuation_valid) {
        require(step.truncated && !step.terminated && step.reward == 0,
                "La valoración ausente no conserva la señal del motor");
    } else {
        require(!(step.terminated && step.truncated) &&
                    (step.terminated ? step.reward < 0 : step.truncated == last_time),
                "El cierre no corresponde al horizonte o a la ruina");
    }
    if (step.terminated || step.truncated) {
        require(cursor + 1 == episode.steps.size(), "El episodio contiene pasos después de su cierre");
    }
}

void put_uint(std::string& out, uint64_t value) {
    for (unsigned shift = 0; shift < integer_bits; shift += bits_per_byte) {
        out.push_back(static_cast<char>((value >> shift) & byte_mask));
    }
}

void put_string(std::string& out, std::string_view text) {
    put_uint(out, text.size());
    out.append(text);
}

void put_float(std::string& out, float value) {
    const auto bits = std::bit_cast<uint32_t>(value);
    for (unsigned shift = 0; shift < float_bits; shift += bits_per_byte) {
        out.push_back(static_cast<char>((bits >> shift) & byte_mask));
    }
}

class Reader {
public:
    explicit Reader(std::string_view bytes) : bytes_(bytes) {}
    uint64_t integer() {
        require(bytes_.size() - cursor_ >= sizeof(uint64_t), "El registro está incompleto");
        uint64_t result = 0;
        for (unsigned shift = 0; shift < integer_bits; shift += bits_per_byte) {
            result |= static_cast<uint64_t>(static_cast<unsigned char>(bytes_[cursor_++])) << shift;
        }
        return result;
    }
    std::size_t size(std::size_t maximum) {
        const auto value = integer();
        require(value <= maximum, "La dimensión declarada excede el límite");
        return static_cast<std::size_t>(value);
    }
    int64_t time() {
        const auto value = integer();
        require(value <= static_cast<uint64_t>(std::numeric_limits<int64_t>::max()),
                "La fecha no cabe en su tipo");
        return static_cast<int64_t>(value);
    }
    std::string text() {
        const auto length = size(maximum_identifier);
        require(length <= bytes_.size() - cursor_, "El identificador está incompleto");
        std::string result(bytes_.substr(cursor_, length));
        cursor_ += length;
        return result;
    }
    float number() {
        require(bytes_.size() - cursor_ >= sizeof(float), "La observación está incompleta");
        uint32_t result = 0;
        for (unsigned shift = 0; shift < float_bits; shift += bits_per_byte) {
            result |= static_cast<uint32_t>(static_cast<unsigned char>(bytes_[cursor_++])) << shift;
        }
        return std::bit_cast<float>(result);
    }
    bool flag() {
        const auto value = integer();
        require(value <= 1, "La máscara no es booleana");
        return value != 0;
    }
    [[nodiscard]] bool done() const { return cursor_ == bytes_.size(); }
private:
    std::string_view bytes_;
    std::size_t cursor_ = 0;
};
}

KlpoEpisodeStatus klpo_episode_status(const KlpoEpisodeRecord& episode) {
    if (episode.steps.empty()) { return KlpoEpisodeStatus::open; }
    const auto& last = episode.steps.back();
    if (!last.valuation_valid) { return KlpoEpisodeStatus::missing_valuation; }
    if (last.terminated) { return KlpoEpisodeStatus::ruin; }
    return last.truncated ? KlpoEpisodeStatus::horizon : KlpoEpisodeStatus::open;
}

void validate_klpo_batch(const KlpoEpisodeBatch& batch, bool require_complete) {
    validate_header(batch);
    std::size_t used = 0;
    account(1, sizeof(KlpoEpisodeBatch) + batch.fold.size() + batch.reference_sha256.size(),
            batch.max_bytes, used);
    std::unordered_set<std::string_view> identities;
    for (const auto& episode : batch.episodes) {
        validate_spec(episode.spec);
        require(identities.insert(episode.spec.id).second, "El registro duplica un episodio");
        const auto planned = episode.spec.close_times.size() - 1;
        account(1, sizeof(KlpoEpisodeRecord) + episode.spec.id.size() + digest_size * 2 +
                    episode.spec.partition.size(), batch.max_bytes, used);
        account(episode.spec.close_times.size(), sizeof(int64_t), batch.max_bytes, used);
        account(planned, sizeof(KlpoEpisodeStep) + batch.observation_width * sizeof(float),
                batch.max_bytes, used);
        require(episode.steps.size() <= planned, "El episodio excede el horizonte declarado");
        for (std::size_t cursor = 0; cursor < episode.steps.size(); ++cursor) {
            validate_step(batch, episode, episode.steps[cursor], cursor);
        }
        if (require_complete) {
            const auto status = klpo_episode_status(episode);
            require(status == KlpoEpisodeStatus::horizon || status == KlpoEpisodeStatus::ruin,
                    "La oleada está incompleta o tiene una valoración inválida");
        }
    }
}

std::vector<double> klpo_terminal_returns(const KlpoEpisodeBatch& batch) {
    validate_klpo_batch(batch, true);
    std::vector<double> result;
    result.reserve(batch.episodes.size());
    for (const auto& episode : batch.episodes) {
        simulation::AccurateSum sum;
        double factor = 1;
        for (const auto& step : episode.steps) {
            require(sum.add(factor * step.reward), "El retorno excede su rango numérico");
            factor *= batch.gamma;
        }
        result.push_back(sum.value());
    }
    return result;
}

std::string serialize_klpo_batch(const KlpoEpisodeBatch& batch) {
    validate_klpo_batch(batch, false);
    std::string out(magic);
    put_uint(out, batch.max_bytes);
    put_uint(out, batch.observation_width);
    put_uint(out, std::bit_cast<uint64_t>(batch.beta));
    put_uint(out, std::bit_cast<uint64_t>(batch.gamma));
    put_string(out, batch.fold);
    put_string(out, batch.reference_sha256);
    put_uint(out, batch.episodes.size());
    for (const auto& episode : batch.episodes) {
        const auto& spec = episode.spec;
        put_string(out, spec.id);
        put_string(out, spec.source_sha256);
        put_string(out, spec.context_sha256);
        put_string(out, spec.partition);
        put_uint(out, spec.forced_prefix);
        put_uint(out, spec.close_times.size());
        for (const auto moment : spec.close_times) { put_uint(out, static_cast<uint64_t>(moment)); }
        put_uint(out, episode.steps.size());
    }
    for (const auto& episode : batch.episodes) {
        for (const auto& step : episode.steps) {
            put_uint(out, step.cursor);
            put_uint(out, static_cast<uint64_t>(step.decision_at));
            put_uint(out, static_cast<uint64_t>(step.outcome_at));
            put_uint(out, step.action);
            put_uint(out, step.sampled);
            put_uint(out, step.valuation_valid);
            put_uint(out, step.terminated);
            put_uint(out, step.truncated);
            put_uint(out, std::bit_cast<uint64_t>(step.reward));
            for (const auto value : step.behavior) { put_float(out, value); }
            for (const auto value : step.observation) { put_float(out, value); }
        }
    }
    require(out.size() <= batch.max_bytes, "El registro serializado supera el presupuesto");
    return out;
}

KlpoEpisodeBatch deserialize_klpo_batch(std::string_view bytes, std::size_t max_bytes) {
    require(max_bytes > 0 && max_bytes <= maximum_klpo_record_bytes && bytes.size() <= max_bytes &&
                bytes.starts_with(magic), "El registro no conserva versión o presupuesto");
    Reader reader(bytes.substr(magic.size()));
    KlpoEpisodeBatch batch;
    batch.max_bytes = reader.size(max_bytes);
    batch.observation_width = reader.size(maximum_width);
    batch.beta = std::bit_cast<double>(reader.integer());
    batch.gamma = std::bit_cast<double>(reader.integer());
    batch.fold = reader.text();
    batch.reference_sha256 = reader.text();
    const auto count = reader.size(maximum_klpo_episodes);
    std::vector<std::size_t> lengths;
    lengths.reserve(count);
    batch.episodes.resize(count);
    for (auto& episode : batch.episodes) {
        auto& spec = episode.spec;
        spec.id = reader.text();
        spec.source_sha256 = reader.text();
        spec.context_sha256 = reader.text();
        spec.partition = reader.text();
        spec.forced_prefix = reader.size(maximum_klpo_steps);
        const auto times = reader.size(maximum_klpo_steps + 1);
        spec.close_times.reserve(times);
        for (std::size_t index = 0; index < times; ++index) { spec.close_times.push_back(reader.time()); }
        lengths.push_back(reader.size(maximum_klpo_steps));
    }
    // La cabecera completa limita las reservas antes de leer observaciones.
    validate_klpo_batch(batch, false);
    for (std::size_t index = 0; index < count; ++index) {
        auto& episode = batch.episodes[index];
        require(lengths[index] < episode.spec.close_times.size(), "El número de pasos no corresponde a la cinta");
        episode.steps.reserve(episode.spec.close_times.size() - 1);
        for (std::size_t cursor = 0; cursor < lengths[index]; ++cursor) {
            KlpoEpisodeStep step;
            step.cursor = reader.size(maximum_klpo_steps);
            step.decision_at = reader.time();
            step.outcome_at = reader.time();
            step.action = static_cast<uint8_t>(reader.size(klpo_record_action_count - 1));
            step.sampled = reader.flag();
            step.valuation_valid = reader.flag();
            step.terminated = reader.flag();
            step.truncated = reader.flag();
            step.reward = std::bit_cast<double>(reader.integer());
            for (auto& value : step.behavior) { value = reader.number(); }
            step.observation.resize(batch.observation_width);
            for (auto& value : step.observation) { value = reader.number(); }
            episode.steps.push_back(std::move(step));
        }
    }
    require(reader.done(), "El registro contiene bytes ajenos");
    validate_klpo_batch(batch, false);
    return batch;
}
}
