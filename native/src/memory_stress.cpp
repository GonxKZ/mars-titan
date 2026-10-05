#include "mars_titan/memory_stress.hpp"
#include "memory_stress_internal.hpp"

#include <array>
#include <bit>
#include <chrono>
#include <deque>
#include <numbers>

namespace mars_titan::stress {
using namespace detail;
using namespace learning;
namespace {
using Clock = std::chrono::steady_clock;
constexpr std::size_t policy_count = 3;
constexpr std::size_t phase_count = 4;
constexpr std::array<Retention, policy_count> policies{Retention::uniform, Retention::recent,
                                                       Retention::selective};
constexpr std::array<std::string_view, policy_count> policy_names{"uniform", "recent", "selective"};
constexpr double smoothing = 0.05;
constexpr double maximum_scaled_error = 4;
constexpr uint64_t maximum_seed = std::numeric_limits<uint64_t>::max();
constexpr std::size_t outlier_period = 31;
constexpr double outlier_amplitude = 8;
constexpr double maximum_noise = 100;
constexpr double uniform_variance_denominator = 3;
constexpr double median_probability = 0.50;
constexpr double tail_probability = 0.95;
constexpr double extreme_probability = 0.99;
constexpr std::size_t histogram_buckets = std::numeric_limits<uint64_t>::digits;

Json build_identity() {
    return {{"source_sha256", MARS_TITAN_STRESS_SOURCE_SHA256},
            {"libtorch", MARS_TITAN_STRESS_TORCH_VERSION},
            {"compiler", __VERSION__},
            {"standard", __cplusplus},
            {"stdlib", __GLIBCXX__},
            {"threads", 1}};
}

void validate(const Config& config) {
    require(config.capacity > 0 && config.capacity <= episodic_memory_capacity &&
                config.steps <= maximum_steps && config.steps > config.capacity + config.delay &&
                config.regime_length > 0 && config.regime_length <= maximum_steps &&
                config.delay > 0 && config.delay <= maximum_delay && std::isfinite(config.noise) &&
                config.noise >= 0 && config.noise <= maximum_noise &&
                config.invalid_every <= maximum_steps,
            "Configuración fuera de límite o recorrido insuficiente para saturar la memoria");
    static_cast<void>(scenario_name(config.scenario));
}
Json config_json(const Config& config) {
    return {{"steps", config.steps},
            {"capacity", config.capacity},
            {"regime_length", config.regime_length},
            {"delay", config.delay},
            {"data_seed", config.data_seed},
            {"retention_seed", config.retention_seed},
            {"noise", config.noise},
            {"invalid_every", config.invalid_every},
            {"scenario", scenario_name(config.scenario)}};
}
Config decode_config(const Json& value) {
    Config config;
    config.steps = integer(value.at("steps"));
    config.capacity = integer(value.at("capacity"), episodic_memory_capacity);
    config.regime_length = integer(value.at("regime_length"));
    config.delay = integer(value.at("delay"), maximum_delay);
    config.data_seed = integer(value.at("data_seed"), maximum_seed);
    config.retention_seed = integer(value.at("retention_seed"), maximum_seed);
    config.noise = number(value.at("noise"));
    config.invalid_every = integer(value.at("invalid_every"));
    config.scenario = scenario_from_name(value.at("scenario").get<std::string>());
    validate(config);
    return config;
}
void hash(uint64_t& digest, uint64_t value) { digest = (digest ^ value) * hash_prime; }
uint64_t bank_digest(const Json& bank) {
    auto digest = hash_offset;
    for (const auto& record : bank.at("records")) {
        hash(digest, integer(record.at("id")));
        hash(digest, std::bit_cast<uint64_t>(number(record.at("label"))));
    }
    return digest;
}
struct Errors {
    uint64_t count = 0;
    double observed = 0;
    double latent = 0;
    double squared = 0;
    void add(double prediction, const MemoryRecord& record, double truth) {
        ++count;
        observed += std::abs(prediction - record.label);
        latent += std::abs(prediction - truth);
        squared += (prediction - record.label) * (prediction - record.label);
    }
    [[nodiscard]] Json state() const {
        return {{"count", count}, {"observed", observed}, {"latent", latent}, {"squared", squared}};
    }
    static Errors read(const Json& value) {
        Errors result{integer(value.at("count")), number(value.at("observed")),
                      number(value.at("latent")), number(value.at("squared"))};
        require(result.observed >= 0 && result.latent >= 0 && result.squared >= 0,
                "El acumulador de errores no puede ser negativo");
        return result;
    }
    [[nodiscard]] Json report() const {
        const auto denominator = static_cast<double>(std::max(count, uint64_t{1}));
        return {{"evaluated", count},
                {"mae_observed", observed / denominator},
                {"mae_latent_evaluation_only", latent / denominator},
                {"rmse_observed", std::sqrt(squared / denominator)}};
    }
};
struct Pending {
    MemoryRecord record;
    std::array<double, policy_count> predictions{};
    double baseline = 0;
    double truth = 0;
    std::size_t phase = 0;
    [[nodiscard]] Json state() const {
        return {{"record", record_json(record)},
                {"predictions", predictions},
                {"baseline", baseline},
                {"truth_evaluation_only", truth},
                {"phase", phase}};
    }
};

struct Measurements {
    double seconds = 0;
    std::array<double, policy_count> query_seconds{};
    std::array<double, policy_count> write_seconds{};
    std::array<uint64_t, histogram_buckets> histogram{};
    std::size_t peak_pending = 0;
    void observe(Clock::duration duration) {
        const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(duration).count();
        const auto bucket = std::min<std::size_t>(
            histogram_buckets - 1,
            static_cast<std::size_t>(std::bit_width(static_cast<uint64_t>(nanos))));
        ++histogram.at(bucket);
    }
    [[nodiscard]] uint64_t quantile(double probability) const {
        uint64_t count = 0;
        for (const auto bucket : histogram) {
            count += bucket;
        }
        if (count == 0) {
            return 0;
        }
        const auto target =
            static_cast<uint64_t>(std::ceil(probability * static_cast<double>(count)));
        uint64_t accumulated = 0;
        for (std::size_t index = 0; index < histogram.size(); ++index) {
            accumulated += histogram.at(index);
            if (accumulated >= target) {
                return uint64_t{1} << index;
            }
        }
        return 0;
    }
};
} // namespace

Scenario scenario_from_name(std::string_view name) {
    if (name == "recurrence") {
        return Scenario::recurrence;
    }
    if (name == "persistent") {
        return Scenario::persistent;
    }
    if (name == "noise") {
        return Scenario::noise;
    }
    if (name == "outliers") {
        return Scenario::outliers;
    }
    throw std::invalid_argument("Escenario desconocido");
}
std::string_view scenario_name(Scenario scenario) {
    switch (scenario) {
    case Scenario::recurrence:
        return "recurrence";
    case Scenario::persistent:
        return "persistent";
    case Scenario::noise:
        return "noise";
    case Scenario::outliers:
        return "outliers";
    }
    throw std::invalid_argument("Escenario desconocido");
}

struct Experiment::Impl {
    explicit Impl(Config settings)
        : config(settings), features(engine(config.data_seed, 1)),
          targets(engine(config.data_seed, 2)) {
        validate(config);
        for (std::size_t index = 0; index < policy_count; ++index) {
            banks.at(index) = std::make_unique<RetentionBank>(policies.at(index), config.capacity,
                                                              config.retention_seed);
        }
    }

    Pending generate() {
        Pending sample;
        const uint64_t id = cursor + 1;
        sample.record.id = id;
        sample.record.decision_at = static_cast<int64_t>(2 * id);
        sample.record.available_at = sample.record.decision_at;
        sample.record.maturity_at = sample.record.decision_at + static_cast<int64_t>(config.delay);
        sample.record.label_valid = true;
        sample.phase = cursor / config.regime_length % phase_count;
        constexpr std::array<std::size_t, phase_count> contexts{0, 1, 2, 0};
        constexpr std::array<double, 3> means{1, -1, 0.5};
        const auto context = contexts.at(sample.phase);
        sample.truth = config.scenario == Scenario::noise ? 0 : means.at(context);
        const auto visible_context = config.scenario == Scenario::persistent ? 0 : context;
        const double angle = static_cast<double>(visible_context) * 2 * std::numbers::pi / 3;
        std::uniform_real_distribution<double> symmetric(-1, 1);
        constexpr double feature_noise = 0.04;
        for (auto& component : sample.record.key) {
            component = static_cast<float>(feature_noise * symmetric(features));
        }
        sample.record.key[0] += static_cast<float>(std::cos(angle));
        sample.record.key[1] += static_cast<float>(std::sin(angle));
        sample.record.label = sample.truth + config.noise *
                                                 std::sqrt(uniform_variance_denominator) *
                                                 symmetric(targets);
        if (config.scenario == Scenario::outliers && id % outlier_period == 0) {
            sample.record.label += symmetric(targets) < 0 ? -outlier_amplitude : outlier_amplitude;
        }
        sample.record.value[0] = static_cast<float>(sample.record.label);
        if (config.invalid_every != 0 && id % config.invalid_every == 0) {
            switch ((id / config.invalid_every) % 3) {
            case 0:
                sample.record.key.fill(0);
                break;
            case 1:
                sample.record.key[0] = std::numeric_limits<float>::quiet_NaN();
                break;
            default:
                sample.record.key[0] = std::numeric_limits<float>::infinity();
                break;
            }
        }
        sample.baseline = baseline;
        return sample;
    }

    void mature(int64_t cutoff) {
        while (!pending.empty() && pending.front().record.maturity_at <= cutoff) {
            const auto& sample = pending.front();
            const double error = sample.record.label - sample.baseline;
            const double scaled =
                std::clamp(error / (1 + error_scale), -maximum_scaled_error, maximum_scaled_error);
            const double next_persistence = (1 - smoothing) * persistence + smoothing * scaled;
            const double priority = std::abs(scaled * next_persistence);
            for (std::size_t index = 0; index < policy_count; ++index) {
                errors.at(index).add(sample.predictions.at(index), sample.record, sample.truth);
                phase_errors.at(index)
                    .at(sample.phase)
                    .add(sample.predictions.at(index), sample.record, sample.truth);
                const auto before = Clock::now();
                banks.at(index)->write(sample.record, cutoff, priority);
                measurements.write_seconds.at(index) +=
                    std::chrono::duration<double>(Clock::now() - before).count();
            }
            baseline_error.add(sample.baseline, sample.record, sample.truth);
            baseline = (1 - smoothing) * baseline + smoothing * sample.record.label;
            error_scale = (1 - smoothing) * error_scale + smoothing * std::abs(error);
            persistence = next_persistence;
            pending.pop_front();
        }
    }

    void tick() {
        const auto start = Clock::now();
        auto sample = generate();
        hash(data_digest, sample.record.id);
        for (const float component : sample.record.key) {
            hash(data_digest, std::bit_cast<uint32_t>(component));
        }
        hash(data_digest, std::bit_cast<uint64_t>(sample.record.label));
        double norm = 0;
        for (const float component : sample.record.key) {
            norm += static_cast<double>(component) * static_cast<double>(component);
        }
        const bool invalid = !std::isfinite(norm) || norm == 0;
        if (invalid) {
            // La observación inválida no llega al banco ni se convierte en etiqueta negativa.
            ++invalid_observations;
        } else {
            for (std::size_t index = 0; index < policy_count; ++index) {
                const auto& bank = *banks.at(index);
                examined.at(index) += bank.size();
                const auto before = Clock::now();
                const auto query = bank.query(sample.record.key, sample.record.decision_at);
                double sum = 0;
                for (std::size_t neighbor = 0; neighbor < query.count; ++neighbor) {
                    sum += query.neighbors.at(neighbor).record.label;
                }
                sample.predictions.at(index) =
                    query.count == 0 ? 0 : sum / static_cast<double>(query.count);
                measurements.query_seconds.at(index) +=
                    std::chrono::duration<double>(Clock::now() - before).count();
                hash(prediction_digests.at(index), sample.record.id);
                hash(prediction_digests.at(index),
                     std::bit_cast<uint64_t>(sample.predictions.at(index)));
            }
            pending.push_back(sample);
            measurements.peak_pending = std::max(measurements.peak_pending, pending.size());
        }
        // Se emiten todas las rutas antes de entregar cualquier resultado que madura en este corte.
        mature(sample.record.decision_at);
        ++cursor;
        measurements.observe(Clock::now() - start);
    }

    Config config;
    Engine features;
    Engine targets;
    std::array<std::unique_ptr<RetentionBank>, policy_count> banks;
    std::size_t cursor = 0;
    uint64_t invalid_observations = 0;
    uint64_t data_digest = hash_offset;
    std::array<uint64_t, policy_count> prediction_digests{hash_offset, hash_offset, hash_offset};
    std::array<uint64_t, policy_count> examined{};
    std::deque<Pending> pending;
    std::array<Errors, policy_count> errors{};
    std::array<std::array<Errors, phase_count>, policy_count> phase_errors{};
    Errors baseline_error;
    double baseline = 0;
    double error_scale = 0;
    double persistence = 0;
    Measurements measurements;
};

Experiment::Experiment(Config config) : impl_(std::make_unique<Impl>(config)) {}
Experiment::~Experiment() = default;
const Config& Experiment::config() const noexcept { return impl_->config; }
void Experiment::run_until(std::size_t cursor) {
    require(cursor >= impl_->cursor && cursor <= config().steps,
            "El cursor retrocede o excede el recorrido");
    const auto start = Clock::now();
    while (impl_->cursor < cursor) {
        impl_->tick();
    }
    if (cursor == config().steps) {
        impl_->mature(static_cast<int64_t>(2 * cursor + config().delay));
    }
    impl_->measurements.seconds += std::chrono::duration<double>(Clock::now() - start).count();
}

std::string Experiment::checkpoint() const {
    Json banks = Json::array();
    Json errors = Json::array();
    Json phases = Json::array();
    for (std::size_t index = 0; index < policy_count; ++index) {
        banks.push_back(parse(impl_->banks.at(index)->checkpoint()));
        errors.push_back(impl_->errors.at(index).state());
        Json phase = Json::array();
        for (const auto& item : impl_->phase_errors.at(index)) {
            phase.push_back(item.state());
        }
        phases.push_back(std::move(phase));
    }
    Json pending = Json::array();
    for (const auto& item : impl_->pending) {
        pending.push_back(item.state());
    }
    return Json{{"version", 1},
                {"build", build_identity()},
                {"config", config_json(config())},
                {"cursor", impl_->cursor},
                {"features_rng", encode_rng(impl_->features)},
                {"targets_rng", encode_rng(impl_->targets)},
                {"banks", std::move(banks)},
                {"pending", std::move(pending)},
                {"errors", std::move(errors)},
                {"phase_errors", std::move(phases)},
                {"baseline_error", impl_->baseline_error.state()},
                {"baseline", impl_->baseline},
                {"error_scale", impl_->error_scale},
                {"persistence", impl_->persistence},
                {"invalid_observations", impl_->invalid_observations},
                {"data_digest", impl_->data_digest},
                {"prediction_digests", impl_->prediction_digests},
                {"examined", impl_->examined},
                {"measurements",
                 {{"seconds", impl_->measurements.seconds},
                  {"query_seconds", impl_->measurements.query_seconds},
                  {"write_seconds", impl_->measurements.write_seconds},
                  {"histogram", impl_->measurements.histogram},
                  {"peak_pending", impl_->measurements.peak_pending}}}}
        .dump();
}

Config read_config(std::string_view checkpoint) {
    const auto value = parse(checkpoint);
    require(integer(value.at("version")) == 1, "Versión de checkpoint incompatible");
    return decode_config(value.at("config"));
}

void Experiment::restore(std::string_view bytes) {
    const auto value = parse(bytes);
    require(integer(value.at("version")) == 1 && value.at("build") == build_identity() &&
                decode_config(value.at("config")) == config(),
            "Versión o configuración incompatible");
    auto next = std::make_unique<Impl>(config());
    next->cursor = integer(value.at("cursor"), config().steps);
    next->features = decode_rng(value.at("features_rng"));
    next->targets = decode_rng(value.at("targets_rng"));
    next->data_digest = integer(value.at("data_digest"), maximum_seed);
    next->invalid_observations = integer(value.at("invalid_observations"), next->cursor);
    require(next->invalid_observations ==
                (config().invalid_every == 0 ? 0 : next->cursor / config().invalid_every),
            "El número de observaciones inválidas no coincide con el cursor");
    next->baseline_error = Errors::read(value.at("baseline_error"));
    next->baseline = number(value.at("baseline"));
    next->error_scale = number(value.at("error_scale"));
    next->persistence = number(value.at("persistence"));
    require(next->error_scale >= 0 && std::abs(next->persistence) <= maximum_scaled_error,
            "El estado del control causal no es válido");
    const auto& queue = value.at("pending");
    require(queue.is_array() && queue.size() <= config().delay + 1, "La cola excede el horizonte");
    uint64_t previous_id = 0;
    for (const auto& item : queue) {
        Pending sample;
        sample.record = read_record(item.at("record"));
        sample.baseline = number(item.at("baseline"));
        sample.truth = number(item.at("truth_evaluation_only"));
        sample.phase = integer(item.at("phase"), phase_count - 1);
        require(item.at("predictions").is_array() && item.at("predictions").size() == policy_count,
                "Falta una predicción pendiente");
        for (std::size_t index = 0; index < policy_count; ++index) {
            sample.predictions.at(index) = number(item.at("predictions").at(index));
        }
        require(sample.record.id > previous_id && sample.record.id <= next->cursor &&
                    sample.record.decision_at == static_cast<int64_t>(2 * sample.record.id) &&
                    sample.record.maturity_at ==
                        sample.record.decision_at + static_cast<int64_t>(config().delay) &&
                    sample.record.maturity_at > static_cast<int64_t>(2 * next->cursor) &&
                    sample.phase == (sample.record.id - 1) / config().regime_length % phase_count &&
                    (config().invalid_every == 0 || sample.record.id % config().invalid_every != 0),
                "La cola contiene una identidad repetida o una maduración incoherente");
        previous_id = sample.record.id;
        next->pending.push_back(sample);
    }
    require(next->pending.size() <= next->cursor - next->invalid_observations,
            "La cola contiene más muestras que el prefijo válido");
    const auto expected = next->cursor - next->invalid_observations - next->pending.size();
    require(next->baseline_error.count == expected &&
                (next->cursor != config().steps || next->pending.empty()),
            "El cursor y las evaluaciones no conservan la cola");
    for (std::string_view field :
         {"banks", "errors", "phase_errors", "prediction_digests", "examined"}) {
        require(value.at(field).is_array() && value.at(field).size() == policy_count,
                "El checkpoint no conserva las tres políticas");
    }
    for (std::size_t index = 0; index < policy_count; ++index) {
        const auto& bank_state = value.at("banks").at(index);
        const auto last_id = integer(bank_state.at("last_id"));
        const auto confirmed_limit =
            2 * next->cursor + (next->cursor == config().steps ? config().delay : 0);
        require(last_id <= next->cursor &&
                    integer(bank_state.at("confirmed_at"), 2 * maximum_steps + maximum_delay) <=
                        confirmed_limit,
                "El banco contiene estado posterior al cursor confirmado");
        for (const auto& record : bank_state.at("records")) {
            const auto id = integer(record.at("id"));
            require(integer(record.at("decision_at"), 2 * maximum_steps) == 2 * id &&
                        integer(record.at("maturity_at"), 2 * maximum_steps + maximum_delay) ==
                            2 * id + config().delay &&
                        (config().invalid_every == 0 || id % config().invalid_every != 0),
                    "El banco no conserva el calendario del escenario");
        }
        next->banks.at(index)->restore(bank_state.dump());
        next->errors.at(index) = Errors::read(value.at("errors").at(index));
        next->prediction_digests.at(index) =
            integer(value.at("prediction_digests").at(index), maximum_seed);
        next->examined.at(index) =
            integer(value.at("examined").at(index), maximum_steps * episodic_memory_capacity);
        require(next->banks.at(index)->writes() == expected &&
                    next->errors.at(index).count == expected,
                "Las escrituras y las evaluaciones no coinciden");
        const auto& phases = value.at("phase_errors").at(index);
        require(phases.is_array() && phases.size() == phase_count, "Falta una fase del escenario");
        uint64_t phase_samples = 0;
        for (std::size_t phase = 0; phase < phase_count; ++phase) {
            next->phase_errors.at(index).at(phase) = Errors::read(phases.at(phase));
            phase_samples += next->phase_errors.at(index).at(phase).count;
        }
        require(phase_samples == expected, "Los errores por fase no conservan las muestras");
    }
    const auto& measurements = value.at("measurements");
    next->measurements.seconds = number(measurements.at("seconds"));
    next->measurements.peak_pending = integer(measurements.at("peak_pending"), config().delay + 1);
    for (std::string_view field : {"query_seconds", "write_seconds"}) {
        require(measurements.at(field).is_array() && measurements.at(field).size() == policy_count,
                "Faltan tiempos por política");
    }
    for (std::size_t index = 0; index < policy_count; ++index) {
        next->measurements.query_seconds.at(index) =
            number(measurements.at("query_seconds").at(index));
        next->measurements.write_seconds.at(index) =
            number(measurements.at("write_seconds").at(index));
        require(next->measurements.query_seconds.at(index) >= 0 &&
                    next->measurements.write_seconds.at(index) >= 0,
                "Los tiempos no pueden ser negativos");
    }
    const auto& histogram = measurements.at("histogram");
    require(histogram.is_array() && histogram.size() == next->measurements.histogram.size() &&
                next->measurements.seconds >= 0,
            "La medición no tiene un formato válido");
    uint64_t measured_ticks = 0;
    for (std::size_t index = 0; index < histogram.size(); ++index) {
        next->measurements.histogram.at(index) = integer(histogram.at(index));
        measured_ticks += next->measurements.histogram.at(index);
    }
    require(measured_ticks == next->cursor, "El histograma no conserva el número de pasos");
    impl_ = std::move(next);
}

std::string Experiment::report() const {
    Json policy_results = Json::array();
    constexpr std::size_t bytes_per_write =
        sizeof(MemoryRecord) + sizeof(double) * (episodic_memory_width + 1);
    for (std::size_t index = 0; index < policy_count; ++index) {
        const auto& bank = *impl_->banks.at(index);
        auto result = impl_->errors.at(index).report();
        result["policy"] = policy_names.at(index);
        result["size"] = bank.size();
        result["saturated"] = bank.size() == config().capacity;
        result["capacity_exceeded_by_candidates"] = bank.writes() > config().capacity;
        result["writes"] = bank.writes();
        result["evictions"] = bank.writes() - bank.size();
        result["bytes_written"] = bank.writes() * bytes_per_write;
        result["reserved_bank_payload_bytes"] = config().capacity * bytes_per_write;
        result["query_candidates_examined"] = impl_->examined.at(index);
        result["priority_candidates_examined"] =
            index == 2 ? (bank.writes() - bank.size()) * config().capacity : 0;
        result["prediction_digest"] = std::to_string(impl_->prediction_digests.at(index));
        result["retained_digest"] = std::to_string(bank_digest(parse(bank.checkpoint())));
        result["phases_evaluation_only"] = Json::array();
        for (const auto& phase : impl_->phase_errors.at(index)) {
            result["phases_evaluation_only"].push_back(phase.report());
        }
        policy_results.push_back(std::move(result));
    }
    const auto& measured = impl_->measurements;
    return Json{
        {"schema_version", 1},
        {"build", build_identity()},
        {"device", "cpu"},
        {"config", config_json(config())},
        {"cursor", impl_->cursor},
        {"completed", impl_->cursor == config().steps},
        {"pending", impl_->pending.size()},
        {"invalid_observations", impl_->invalid_observations},
        {"data_digest", std::to_string(impl_->data_digest)},
        {"baseline", impl_->baseline_error.report()},
        {"policies", std::move(policy_results)},
        {"measurements",
         {{"engine_seconds", measured.seconds},
          {"query_seconds_by_policy", measured.query_seconds},
          {"write_seconds_by_policy", measured.write_seconds},
          {"steps_per_second",
           measured.seconds > 0 ? static_cast<double>(impl_->cursor) / measured.seconds : 0},
          {"latency_log2_upper_ns",
           {{"p50", measured.quantile(median_probability)},
            {"p95", measured.quantile(tail_probability)},
            {"p99", measured.quantile(extreme_probability)}}},
          {"peak_pending", measured.peak_pending},
          {"peak_pending_payload_bytes", measured.peak_pending * sizeof(Pending)},
          {"input_payload_bytes", impl_->cursor * (sizeof(MemoryVector) + sizeof(double))},
          {"host_device_transfer_bytes", 0},
          {"vram_bytes", 0},
          {"energy_joules", nullptr},
          {"monetary_cost", nullptr}}}}
        .dump(2);
}
} // namespace mars_titan::stress
