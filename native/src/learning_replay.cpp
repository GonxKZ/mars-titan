#include "mars_titan/learning_replay.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/core/grad_mode.h>
#include <torch/serialize/archive.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <mutex>
#include <span>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace mars_titan::learning {
namespace {
constexpr std::size_t maximum_width = 32768;
constexpr std::size_t archive_overhead = std::size_t{64} * 1024;
constexpr uint64_t maximum_seen = uint64_t{1} << 32;
constexpr uint64_t selection_seed_offset = 0xd1b54a32d192ed03ULL;
constexpr int64_t archive_version = 1;
constexpr int64_t action_count = 6;
constexpr std::size_t field_count = 5;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::invalid_argument(message);
}
void validate_config(std::size_t width, std::size_t capacity, ReplayMode mode, bool store_next,
                     std::size_t max_bytes) {
    const auto maximum_capacity = store_next ? dqn_replay_capacity : auxiliary_replay_capacity;
    const auto maximum_bytes = store_next ? dqn_replay_bytes : auxiliary_replay_bytes;
    require(width > 0 && width <= maximum_width && capacity > 0 && capacity <= maximum_capacity &&
                (mode == ReplayMode::recent || mode == ReplayMode::reservoir) &&
                max_bytes > archive_overhead && max_bytes <= maximum_bytes,
            "El replay supera su capacidad, anchura o presupuesto admitidos");
    const auto row_bytes = width * sizeof(float) * (store_next ? 2 : 1) + sizeof(int64_t) +
                           sizeof(float) + sizeof(bool);
    require(capacity <= (max_bytes - archive_overhead) / row_bytes,
            "El presupuesto del replay no cubre sus tensores y metadatos");
}
ReplaySample allocate(std::size_t count, bool store_next, std::size_t width) {
    const auto rows = static_cast<int64_t>(count);
    const auto columns = static_cast<int64_t>(width);
    ReplaySample result;
    result.observations = at::zeros({rows, columns}, at::kFloat);
    if (store_next)
        result.next_observations = at::zeros({rows, columns}, at::kFloat);
    result.actions = at::zeros({rows}, at::kLong);
    result.rewards = at::zeros({rows}, at::kFloat);
    result.terminated = at::zeros({rows}, at::kBool);
    return result;
}
template <class Data> auto fields(Data& data) {
    using Pointer = decltype(&data.observations);
    return std::array<std::pair<const char*, Pointer>, field_count>{
        {{"observations", &data.observations},
         {"next_observations", &data.next_observations},
         {"actions", &data.actions},
         {"rewards", &data.rewards},
         {"terminated", &data.terminated}}};
}
void check_tensor(const at::Tensor& tensor, at::ScalarType type, at::IntArrayRef shape) {
    require(tensor.defined() && tensor.device().is_cpu() && tensor.layout() == at::kStrided &&
                tensor.scalar_type() == type && tensor.sizes() == shape && tensor.is_contiguous() &&
                !tensor.requires_grad(),
            "El replay necesita tensores CPU contiguos y sin gradiente con el esquema acordado");
    if (tensor.is_floating_point())
        require(at::isfinite(tensor).all().item<bool>(), "El replay contiene NaN o infinito");
}
at::Tensor rng_state(at::Generator generator) {
    const std::lock_guard lock(generator.mutex());
    return generator.get_state();
}
at::Generator restored_rng(const at::Tensor& state, uint64_t seed) {
    auto result = at::detail::createCPUGenerator(seed);
    const auto expected = rng_state(result);
    check_tensor(state, at::kByte, expected.sizes());
    const std::lock_guard lock(result.mutex());
    result.set_state(state);
    require(result.current_seed() == seed, "El generador de replay no conserva su semilla");
    return result;
}
void validate_data(const ReplaySnapshot& state) {
    const auto size = static_cast<int64_t>(state.size);
    const auto width = static_cast<int64_t>(state.width);
    check_tensor(state.data.observations, at::kFloat, {size, width});
    if (state.store_next)
        check_tensor(state.data.next_observations, at::kFloat, {size, width});
    else
        require(!state.data.next_observations.defined(),
                "El replay auxiliar no admite estados siguientes");
    check_tensor(state.data.actions, at::kLong, {size});
    check_tensor(state.data.rewards, at::kFloat, {size});
    check_tensor(state.data.terminated, at::kBool, {size});
    require(((state.data.actions >= 0) & (state.data.actions < action_count)).all().item<bool>(),
            "El replay contiene acciones fuera del contrato financiero");
}
int64_t read_integer(torch::serialize::InputArchive& archive, const char* name) {
    c10::IValue value;
    archive.read(name, value);
    require(value.isInt(), "El archivo de replay necesita contadores enteros");
    return value.toInt();
}
std::size_t read_size(torch::serialize::InputArchive& archive, const char* name) {
    const auto value = read_integer(archive, name);
    require(value >= 0, "Un contador de replay no puede ser negativo");
    return static_cast<std::size_t>(value);
}
} // namespace

struct LearningReplay::Impl {
    struct Settings {
        std::size_t width;
        std::size_t capacity;
        ReplayMode mode;
        bool store_next;
        uint64_t seed;
        std::size_t max_bytes;
    };
    explicit Impl(Settings settings)
        : width(settings.width), capacity(settings.capacity), max_bytes(settings.max_bytes),
          mode(settings.mode), store_next(settings.store_next), seed(settings.seed),
          sampling(at::detail::createCPUGenerator(seed)),
          selection(at::detail::createCPUGenerator(seed ^ selection_seed_offset)),
          data(allocate(capacity, store_next, width)) {}
    std::size_t width;
    std::size_t capacity;
    std::size_t max_bytes;
    ReplayMode mode;
    bool store_next;
    uint64_t seed;
    uint64_t seen = 0;
    std::size_t size = 0;
    std::size_t position = 0;
    at::Generator sampling;
    at::Generator selection;
    ReplaySample data;
};
LearningReplay::LearningReplay(std::size_t width, std::size_t capacity, ReplayMode mode,
                               bool store_next, uint64_t seed, std::size_t max_bytes) {
    validate_config(width, capacity, mode, store_next, max_bytes);
    impl_ = std::make_unique<Impl>(Impl::Settings{.width = width,
                                                  .capacity = capacity,
                                                  .mode = mode,
                                                  .store_next = store_next,
                                                  .seed = seed,
                                                  .max_bytes = max_bytes});
}
LearningReplay::LearningReplay(LearningReplay&&) noexcept = default;
LearningReplay& LearningReplay::operator=(LearningReplay&&) noexcept = default;
LearningReplay::~LearningReplay() = default;

void LearningReplay::add(const at::Tensor& observation, int64_t action, double reward,
                         bool terminated, const at::Tensor& next_observation, bool reward_valid) {
    require(reward_valid && action >= 0 && action < action_count && std::isfinite(reward) &&
                std::abs(reward) <= static_cast<double>(std::numeric_limits<float>::max()) &&
                impl_->seen < maximum_seen,
            "El replay necesita una recompensa madura finita, una acción válida y un contador "
            "disponible");
    const auto width = static_cast<int64_t>(impl_->width);
    check_tensor(observation, at::kFloat, {width});
    if (impl_->store_next)
        check_tensor(next_observation, at::kFloat, {width});
    else
        require(!next_observation.defined(),
                "El replay auxiliar no almacena una observación siguiente");
    const at::NoGradGuard no_grad;
    // Copia solo las filas entrantes antes de escribir, también si son vistas del propio replay.
    const auto current = observation.clone();
    const auto following = impl_->store_next ? next_observation.clone() : at::Tensor{};
    auto selection = impl_->selection.clone();
    auto slot = impl_->position;
    if (impl_->mode == ReplayMode::reservoir) {
        slot = impl_->size < impl_->capacity
                   ? impl_->size
                   : static_cast<std::size_t>(at::randint(static_cast<int64_t>(impl_->seen + 1),
                                                          {1}, selection, at::kLong)
                                                  .item<int64_t>());
    }
    if (slot < impl_->capacity) {
        const auto destination =
            std::span(impl_->data.observations.data_ptr<float>(), impl_->capacity * impl_->width)
                .subspan(slot * impl_->width, impl_->width);
        const auto source = std::span(current.const_data_ptr<float>(), impl_->width);
        std::copy(source.begin(), source.end(), destination.begin());
        if (impl_->store_next) {
            const auto target = std::span(impl_->data.next_observations.data_ptr<float>(),
                                          impl_->capacity * impl_->width)
                                    .subspan(slot * impl_->width, impl_->width);
            const auto next = std::span(following.const_data_ptr<float>(), impl_->width);
            std::copy(next.begin(), next.end(), target.begin());
        }
        std::span(impl_->data.actions.data_ptr<int64_t>(), impl_->capacity)[slot] = action;
        std::span(impl_->data.rewards.data_ptr<float>(), impl_->capacity)[slot] =
            static_cast<float>(reward);
        std::span(impl_->data.terminated.data_ptr<bool>(), impl_->capacity)[slot] = terminated;
    }
    impl_->selection = std::move(selection);
    ++impl_->seen;
    impl_->size =
        static_cast<std::size_t>(std::min(impl_->seen, static_cast<uint64_t>(impl_->capacity)));
    impl_->position = static_cast<std::size_t>(impl_->seen % impl_->capacity);
}
ReplaySample LearningReplay::sample(std::size_t count) {
    require(impl_->size > 0 && count > 0 && count <= learning_replay_batch &&
                count <= impl_->capacity,
            "El replay no tiene un lote de entre una y 64 filas disponible");
    const at::NoGradGuard no_grad;
    auto sampling = impl_->sampling.clone();
    const auto indices = at::randint(static_cast<int64_t>(impl_->size),
                                     {static_cast<int64_t>(count)}, sampling, at::kLong);
    ReplaySample result;
    const auto origin = fields(impl_->data);
    const auto destination = fields(result);
    for (std::size_t field = 0; field < origin.size(); ++field) {
        if (origin.at(field).second->defined())
            *destination.at(field).second = origin.at(field).second->index_select(0, indices);
    }
    impl_->sampling = std::move(sampling);
    return result;
}
at::Tensor LearningReplay::observations() const {
    return impl_->data.observations.narrow(0, 0, static_cast<int64_t>(impl_->size));
}
at::Tensor LearningReplay::rewards() const {
    return impl_->data.rewards.narrow(0, 0, static_cast<int64_t>(impl_->size));
}
ReplaySnapshot LearningReplay::snapshot() const {
    ReplaySnapshot result;
    result.width = impl_->width;
    result.capacity = impl_->capacity;
    result.max_bytes = impl_->max_bytes;
    result.mode = impl_->mode;
    result.store_next = impl_->store_next;
    result.seed = impl_->seed;
    result.seen = impl_->seen;
    result.size = impl_->size;
    result.position = impl_->position;
    const auto origin = fields(impl_->data);
    const auto destination = fields(result.data);
    for (std::size_t field = 0; field < origin.size(); ++field) {
        if (origin.at(field).second->defined())
            *destination.at(field).second =
                origin.at(field).second->narrow(0, 0, static_cast<int64_t>(impl_->size)).clone();
    }
    result.sampling_rng = rng_state(impl_->sampling);
    result.selection_rng = rng_state(impl_->selection);
    return result;
}
void LearningReplay::restore(const ReplaySnapshot& state) {
    require(state.width == impl_->width && state.capacity == impl_->capacity &&
                state.mode == impl_->mode && state.store_next == impl_->store_next &&
                state.seed == impl_->seed && state.max_bytes == impl_->max_bytes &&
                state.seen <= maximum_seen &&
                state.size == std::min(state.seen, static_cast<uint64_t>(impl_->capacity)) &&
                state.position == state.seen % impl_->capacity,
            "El replay guardado no conserva configuración, tamaño o cursor");
    validate_data(state);
    auto sampling = restored_rng(state.sampling_rng, impl_->seed);
    auto selection = restored_rng(state.selection_rng, impl_->seed ^ selection_seed_offset);
    auto candidate = std::make_unique<Impl>(Impl::Settings{.width = impl_->width,
                                                           .capacity = impl_->capacity,
                                                           .mode = impl_->mode,
                                                           .store_next = impl_->store_next,
                                                           .seed = impl_->seed,
                                                           .max_bytes = impl_->max_bytes});
    if (state.mode == ReplayMode::recent || state.seen <= state.capacity) {
        require(at::equal(state.selection_rng, rng_state(candidate->selection)),
                "El replay sin reemplazos aleatorios alteró su RNG de selección");
    }
    if (state.seen == 0)
        require(at::equal(state.sampling_rng, rng_state(candidate->sampling)),
                "Un replay vacío consumió el RNG de muestreo");
    const auto origin = fields(state.data);
    const auto destination = fields(candidate->data);
    for (std::size_t field = 0; field < origin.size(); ++field) {
        if (origin.at(field).second->defined())
            destination.at(field)
                .second->narrow(0, 0, static_cast<int64_t>(state.size))
                .copy_(*origin.at(field).second);
    }
    candidate->sampling = std::move(sampling);
    candidate->selection = std::move(selection);
    candidate->seen = state.seen;
    candidate->size = state.size;
    candidate->position = state.position;
    impl_.swap(candidate);
}
uint64_t LearningReplay::seen() const noexcept { return impl_->seen; }
std::size_t LearningReplay::size() const noexcept { return impl_->size; }
std::size_t LearningReplay::capacity() const noexcept { return impl_->capacity; }

std::string serialize_replay(const ReplaySnapshot& snapshot) {
    LearningReplay validated(snapshot.width, snapshot.capacity, snapshot.mode, snapshot.store_next,
                             snapshot.seed, snapshot.max_bytes);
    validated.restore(snapshot);
    const auto state = validated.snapshot();
    torch::serialize::OutputArchive archive;
    archive.write("schema_version", c10::IValue(archive_version));
    archive.write("width", c10::IValue(static_cast<int64_t>(state.width)));
    archive.write("capacity", c10::IValue(static_cast<int64_t>(state.capacity)));
    archive.write("max_bytes", c10::IValue(static_cast<int64_t>(state.max_bytes)));
    archive.write("mode", c10::IValue(static_cast<int64_t>(state.mode)));
    archive.write("store_next", c10::IValue(state.store_next));
    archive.write("seed", c10::IValue(std::bit_cast<int64_t>(state.seed)));
    archive.write("seen", c10::IValue(static_cast<int64_t>(state.seen)));
    archive.write("size", c10::IValue(static_cast<int64_t>(state.size)));
    archive.write("position", c10::IValue(static_cast<int64_t>(state.position)));
    archive.write("sampling_rng", state.sampling_rng, true);
    archive.write("selection_rng", state.selection_rng, true);
    for (const auto [name, tensor] : fields(state.data))
        if (tensor->defined())
            archive.write(name, *tensor, true);
    std::ostringstream stream;
    archive.save_to(stream);
    auto result = std::move(stream).str();
    require(result.size() <= state.max_bytes, "El archivo de replay excede el presupuesto");
    return result;
}
ReplaySnapshot deserialize_replay(std::string_view bytes) {
    constexpr std::size_t footer_bytes = 22;
    require(bytes.size() >= footer_bytes && bytes.size() <= dqn_replay_bytes &&
                bytes.starts_with("PK\003\004") &&
                bytes.substr(bytes.size() - footer_bytes).starts_with("PK\005\006") &&
                bytes[bytes.size() - 1] == '\0' && bytes[bytes.size() - 2] == '\0',
            "El archivo de replay está truncado o supera 128 MiB");
    std::istringstream stream{std::string(bytes)};
    torch::serialize::InputArchive archive;
    archive.load_from(stream, at::Device(at::kCPU));
    require(read_integer(archive, "schema_version") == archive_version,
            "La versión del replay no está admitida");
    ReplaySnapshot result;
    result.width = read_size(archive, "width");
    result.capacity = read_size(archive, "capacity");
    result.max_bytes = read_size(archive, "max_bytes");
    const auto mode = read_integer(archive, "mode");
    require(mode == static_cast<int64_t>(ReplayMode::recent) ||
                mode == static_cast<int64_t>(ReplayMode::reservoir),
            "El modo de replay no está admitido");
    result.mode = static_cast<ReplayMode>(mode);
    c10::IValue store_next;
    archive.read("store_next", store_next);
    require(store_next.isBool(), "El replay necesita una bandera booleana de estado siguiente");
    result.store_next = store_next.toBool();
    result.seed = std::bit_cast<uint64_t>(read_integer(archive, "seed"));
    result.seen = read_size(archive, "seen");
    result.size = read_size(archive, "size");
    result.position = read_size(archive, "position");
    validate_config(result.width, result.capacity, result.mode, result.store_next,
                    result.max_bytes);
    require(bytes.size() <= result.max_bytes,
            "El archivo supera el presupuesto declarado del replay");
    archive.read("sampling_rng", result.sampling_rng, true);
    archive.read("selection_rng", result.selection_rng, true);
    for (auto [name, tensor] : fields(result.data)) {
        if (result.store_next || std::string_view(name) != "next_observations")
            archive.read(name, *tensor, true);
    }
    LearningReplay validated(result.width, result.capacity, result.mode, result.store_next,
                             result.seed, result.max_bytes);
    validated.restore(result);
    return validated.snapshot();
}
} // namespace mars_titan::learning
