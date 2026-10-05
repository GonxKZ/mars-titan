#include "mars_titan/replay_schedule.hpp"

#include <algorithm>
#include <iomanip>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace mars_titan::controls {
namespace {
constexpr std::size_t maximum_episodes = 1U << 20;
constexpr std::size_t maximum_exposures = 1U << 24;
constexpr std::size_t maximum_bytes = 512U << 20;
constexpr std::size_t maximum_archive_bytes = 128U << 20;
constexpr std::size_t temporary_bytes_per_episode = 128;
constexpr std::size_t fixed_overhead = 4096;
constexpr uint64_t format_version = 1;

void require(bool condition, const char* message) {
    if (!condition)
        throw std::invalid_argument(message);
}
std::size_t validate(const ReplayScheduleConfig& config, std::span<const MatureEpisode> episodes) {
    static_cast<void>(replay_order_name(config.order));
    require(
        !config.run_id.empty() && config.run_id.size() <= 128 &&
            config.run_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                            "0123456789_-.:") == std::string::npos,
        "El replay necesita una identidad breve y explícita, distinta del feedback");
    require(config.max_exposures > 0 && config.max_exposures <= maximum_exposures &&
                config.max_bytes >= fixed_overhead && config.max_bytes <= maximum_bytes &&
                config.batch_size > 0 && config.batch_size <= config.max_exposures &&
                config.minimum_distance > 0 && config.minimum_distance <= config.max_exposures,
            "El calendario supera sus límites de lotes, espaciado, exposiciones o memoria");
    require(!episodes.empty() && episodes.size() <= maximum_episodes &&
                episodes.size() <= config.max_exposures &&
                episodes.size() <=
                    (config.max_bytes - fixed_overhead) / temporary_bytes_per_episode,
            "El número de episodios no cabe en el presupuesto del calendario");
    std::size_t total = 0;
    std::vector<uint64_t> identities;
    identities.reserve(episodes.size());
    for (const auto& episode : episodes) {
        require(
            episode.id > 0 && episode.observed_at <= episode.available_at &&
                episode.available_at <= config.cutoff && episode.exposures > 0,
            "El calendario necesita episodios identificados, maduros y con exposiciones positivas");
        require(episode.exposures <= config.max_exposures - total,
                "El multiconjunto excede el presupuesto de exposiciones");
        total += episode.exposures;
        identities.push_back(episode.id);
    }
    const auto metadata_bytes = fixed_overhead + episodes.size() * temporary_bytes_per_episode;
    require(total <= (config.max_bytes - metadata_bytes) / sizeof(uint32_t),
            "Los índices y los buffers del calendario exceden el presupuesto de memoria");
    std::sort(identities.begin(), identities.end());
    require(std::adjacent_find(identities.begin(), identities.end()) == identities.end(),
            "Una identidad de episodio aparece más de una vez");
    return total;
}
uint64_t bounded(std::mt19937_64& rng, uint64_t bound) {
    const auto threshold = (uint64_t{0} - bound) % bound;
    auto value = rng();
    while (value < threshold)
        value = rng();
    return value % bound;
}
struct WaitingEpisode {
    std::size_t remaining = 0;
    std::size_t ready_at = 0;
    uint64_t tie = 0;
    uint32_t index = 0;
};
struct Priority {
    bool operator()(const WaitingEpisode& left, const WaitingEpisode& right) const noexcept {
        if (left.remaining != right.remaining)
            return left.remaining < right.remaining;
        if (left.tie != right.tie)
            return left.tie < right.tie;
        return left.index < right.index;
    }
};
void spaced_indices(std::vector<uint32_t>& indices, std::span<const MatureEpisode> episodes,
                    std::size_t distance, std::mt19937_64& rng, std::size_t total) {
    const auto largest =
        std::max_element(episodes.begin(), episodes.end(), [](const auto& left, const auto& right) {
            return left.exposures < right.exposures;
        })->exposures;
    const auto tied = static_cast<std::size_t>(
        std::count_if(episodes.begin(), episodes.end(),
                      [largest](const auto& row) { return row.exposures == largest; }));
    // La separación se mide en exposiciones, sin insertar huecos ni actualizaciones adicionales.
    require(static_cast<std::size_t>(largest) - 1 <= (total - tied) / distance,
            "El espaciado solicitado es imposible con este multiconjunto y sin añadir huecos");
    std::vector<WaitingEpisode> heap;
    heap.reserve(episodes.size());
    for (std::size_t index = 0; index < episodes.size(); ++index)
        heap.push_back({episodes[index].exposures, 0, rng(), static_cast<uint32_t>(index)});
    std::priority_queue<WaitingEpisode, std::vector<WaitingEpisode>, Priority> ready(
        Priority{}, std::move(heap));
    std::vector<WaitingEpisode> cooling(episodes.size());
    std::size_t head = 0;
    std::size_t count = 0;
    while (indices.size() < total) {
        while (count > 0 && cooling[head].ready_at <= indices.size()) {
            ready.push(cooling[head]);
            head = (head + 1) % cooling.size();
            --count;
        }
        require(!ready.empty(), "No se puede completar el espaciado dentro del presupuesto");
        auto selected = ready.top();
        ready.pop();
        selected.ready_at = indices.size() + distance;
        indices.push_back(selected.index);
        if (--selected.remaining > 0) {
            cooling[(head + count) % cooling.size()] = selected;
            ++count;
        }
    }
}
} // namespace

std::string_view replay_order_name(ReplayOrder order) {
    switch (order) {
    case ReplayOrder::uniform:
        return "uniform";
    case ReplayOrder::recent:
        return "recent";
    case ReplayOrder::spaced:
        return "spaced";
    }
    throw std::invalid_argument("El orden de replay no existe");
}
ReplaySchedule::ReplaySchedule(ReplayScheduleConfig config, std::span<const MatureEpisode> episodes)
    : config_(std::move(config)) {
    const auto total = validate(config_, episodes);
    episodes_.assign(episodes.begin(), episodes.end());
    indices_.reserve(total);
    std::mt19937_64 rng(config_.order_seed);
    if (config_.order == ReplayOrder::spaced) {
        spaced_indices(indices_, episodes_, config_.minimum_distance, rng, total);
        return;
    }
    std::vector<uint32_t> episode_order(episodes_.size());
    std::iota(episode_order.begin(), episode_order.end(), uint32_t{0});
    if (config_.order == ReplayOrder::recent)
        std::sort(episode_order.begin(), episode_order.end(), [this](auto left, auto right) {
            const auto& first = episodes_[left];
            const auto& second = episodes_[right];
            if (first.available_at != second.available_at)
                return first.available_at > second.available_at;
            return first.id < second.id;
        });
    for (const auto index : episode_order)
        indices_.insert(indices_.end(), episodes_[index].exposures, index);
    if (config_.order == ReplayOrder::uniform)
        for (std::size_t end = indices_.size(); end > 1; --end)
            std::swap(indices_[end - 1], indices_[static_cast<std::size_t>(bounded(rng, end))]);
}
PreparedReplay ReplaySchedule::prepare() const {
    require(cursor_ < indices_.size(), "El calendario ya ha confirmado todas sus exposiciones");
    const auto count = std::min(config_.batch_size, indices_.size() - cursor_);
    return {{config_.run_id, config_.order, config_.order_seed, cursor_, count, updates_},
            std::span(indices_).subspan(cursor_, count)};
}
void ReplaySchedule::commit(const ReplayToken& token) {
    require(token == prepare().token,
            "El token de replay no pertenece al siguiente lote pendiente");
    cursor_ += token.count;
    ++updates_;
}
ReplayScheduleSnapshot ReplaySchedule::snapshot() const {
    return {config_, episodes_, cursor_, updates_};
}
void ReplaySchedule::restore(const ReplayScheduleSnapshot& state) {
    require(state.config == config_ && state.episodes == episodes_,
            "El checkpoint cambia la identidad, el orden o el multiconjunto de replay");
    require(state.cursor <= size() &&
                (state.cursor == size() || state.cursor % config_.batch_size == 0) &&
                state.updates == state.cursor / config_.batch_size +
                                     (state.cursor % config_.batch_size != 0 ? 1 : 0),
            "El cursor no corresponde a un número entero de lotes confirmados");
    cursor_ = state.cursor;
    updates_ = state.updates;
}
std::size_t ReplaySchedule::size() const noexcept { return indices_.size(); }
std::size_t ReplaySchedule::cursor() const noexcept { return cursor_; }
std::size_t ReplaySchedule::updates() const noexcept { return updates_; }
std::size_t ReplaySchedule::index_bytes() const noexcept {
    return indices_.size() * sizeof(uint32_t);
}

std::string serialize_schedule(const ReplayScheduleSnapshot& state) {
    ReplaySchedule checked(state.config, state.episodes);
    checked.restore(state);
    const auto& config = state.config;
    std::ostringstream output;
    output << format_version << ' ' << std::quoted(config.run_id) << ' '
           << static_cast<unsigned int>(config.order) << ' ' << config.cutoff << ' '
           << config.order_seed << ' ' << config.batch_size << ' ' << config.minimum_distance << ' '
           << config.max_exposures << ' ' << config.max_bytes << ' ' << state.cursor << ' '
           << state.updates << ' ' << state.episodes.size() << '\n';
    for (const auto& episode : state.episodes)
        output << episode.id << ' ' << episode.observed_at << ' ' << episode.available_at << ' '
               << episode.exposures << '\n';
    return output.str();
}
ReplayScheduleSnapshot deserialize_schedule(std::string_view archive) {
    require(!archive.empty() && archive.size() <= maximum_archive_bytes,
            "El archivo de calendario está vacío o excede el límite de lectura");
    std::istringstream input{std::string(archive)};
    ReplayScheduleSnapshot result;
    auto& config = result.config;
    uint64_t version = 0;
    unsigned int order = 0;
    std::size_t count = 0;
    input >> version >> std::quoted(config.run_id) >> order >> config.cutoff >> config.order_seed >>
        config.batch_size >> config.minimum_distance >> config.max_exposures >> config.max_bytes >>
        result.cursor >> result.updates >> count;
    require(static_cast<bool>(input) && version == format_version && order <= 2 && count > 0 &&
                count <= maximum_episodes && count <= archive.size() / 8,
            "La cabecera o el número de episodios del checkpoint no son válidos");
    config.order = static_cast<ReplayOrder>(order);
    result.episodes.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        MatureEpisode episode;
        input >> episode.id >> episode.observed_at >> episode.available_at >> episode.exposures;
        require(static_cast<bool>(input), "El checkpoint de replay está truncado");
        result.episodes.push_back(episode);
    }
    input >> std::ws;
    require(input.eof(), "El checkpoint de replay contiene datos sobrantes");
    ReplaySchedule checked(config, result.episodes);
    checked.restore(result);
    return result;
}
} // namespace mars_titan::controls
