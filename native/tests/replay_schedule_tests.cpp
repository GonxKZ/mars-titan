#include "mars_titan/replay_schedule.hpp"

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <utility>

namespace {
using namespace mars_titan::controls;
// Las constantes de estos casos expresan los datos y la solución esperada del calendario.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
template <class Function> void rejected(Function&& action) {
    try {
        std::forward<Function>(action)();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Se aceptó un calendario o un cursor inválido");
}
ReplayScheduleConfig config(ReplayOrder order = ReplayOrder::uniform) {
    return {.run_id = "replay-control-v1",
            .order = order,
            .cutoff = 30,
            .order_seed = 19,
            .batch_size = 2,
            .minimum_distance = 2};
}
std::vector<MatureEpisode> episodes() { return {{11, 0, 10, 4}, {22, 1, 20, 2}, {33, 2, 30, 1}}; }
std::vector<uint32_t> consume(ReplaySchedule& schedule) {
    std::vector<uint32_t> result;
    while (schedule.cursor() < schedule.size()) {
        const auto prepared = schedule.prepare();
        require(prepared.token.cursor == result.size(), "El cursor de exposición no es continuo");
        result.insert(result.end(), prepared.indices.begin(), prepared.indices.end());
        schedule.commit(prepared.token);
    }
    return result;
}
void exact_multisets_and_spacing() {
    const std::vector<uint32_t> expected{0, 0, 0, 0, 1, 1, 2};
    for (const auto order : {ReplayOrder::uniform, ReplayOrder::recent, ReplayOrder::spaced}) {
        ReplaySchedule schedule(config(order), episodes());
        auto observed = consume(schedule);
        require(schedule.updates() == 4 && schedule.cursor() == 7 && schedule.index_bytes() == 28,
                "El calendario cambia exposiciones, actualizaciones o tamaño del índice");
        if (order == ReplayOrder::recent)
            require(observed == std::vector<uint32_t>({2, 1, 1, 0, 0, 0, 0}),
                    "El calendario reciente no da prioridad a la evidencia más nueva");
        if (order == ReplayOrder::spaced)
            for (std::size_t index = 1; index < observed.size(); ++index)
                require(observed[index] != observed[index - 1],
                        "El espaciado repite sin separación");
        std::sort(observed.begin(), observed.end());
        require(observed == expected, "Los controles no comparten el multiconjunto exacto");
        rejected([&] { static_cast<void>(schedule.prepare()); });
    }
    auto impossible = config(ReplayOrder::spaced);
    impossible.minimum_distance = 3;
    rejected([&] { ReplaySchedule schedule(impossible, episodes()); });
}
void recovered_cursor_is_transactional() {
    for (const auto order : {ReplayOrder::uniform, ReplayOrder::recent, ReplayOrder::spaced}) {
        ReplaySchedule original(config(order), episodes());
        const auto first = original.prepare();
        const auto before = original.snapshot();
        original.commit(first.token);
        const auto committed = deserialize_schedule(serialize_schedule(original.snapshot()));
        ReplaySchedule resumed(config(order), episodes());
        resumed.restore(before);
        require(resumed.prepare().token == first.token, "Una interrupción previa pierde el lote");
        resumed.restore(committed);
        rejected([&] { resumed.commit(first.token); });
        while (original.cursor() < original.size()) {
            const auto expected = original.prepare();
            const auto actual = resumed.prepare();
            require(expected.token == actual.token &&
                        std::equal(expected.indices.begin(), expected.indices.end(),
                                   actual.indices.begin()),
                    "La recuperación repite o altera una exposición");
            original.commit(expected.token);
            resumed.commit(actual.token);
        }
        resumed.restore(deserialize_schedule(serialize_schedule(original.snapshot())));
        require(resumed.cursor() == 7 && resumed.updates() == 4,
                "Se pierde el último lote parcial");
        rejected([&] { static_cast<void>(resumed.prepare()); });
    }
}
void invalid_inputs_leave_state_unchanged() {
    auto data = episodes();
    auto settings = config();
    rejected([&] { ReplaySchedule schedule(settings, {}); });
    data[0].available_at = 31;
    rejected([&] { ReplaySchedule schedule(settings, data); });
    data = episodes();
    data[1].observed_at = 21;
    rejected([&] { ReplaySchedule schedule(settings, data); });
    data = episodes();
    data[1].id = data[0].id;
    rejected([&] { ReplaySchedule schedule(settings, data); });
    data = episodes();
    data[1].exposures = 0;
    rejected([&] { ReplaySchedule schedule(settings, data); });
    data[1].exposures = std::numeric_limits<uint32_t>::max();
    rejected([&] { ReplaySchedule schedule(settings, data); });
    settings.batch_size = 0;
    rejected([&] { ReplaySchedule schedule(settings, episodes()); });
    settings = config();
    settings.max_exposures = 6;
    rejected([&] { ReplaySchedule schedule(settings, episodes()); });
    settings = config();
    settings.max_bytes = 16;
    rejected([&] { ReplaySchedule schedule(settings, episodes()); });
    settings = config();
    settings.run_id.clear();
    rejected([&] { ReplaySchedule schedule(settings, episodes()); });
    settings = config();
    settings.order = static_cast<ReplayOrder>(255);
    rejected([&] { ReplaySchedule schedule(settings, episodes()); });

    ReplaySchedule schedule(config(), episodes());
    const auto original = serialize_schedule(schedule.snapshot());
    auto token = schedule.prepare().token;
    ++token.count;
    rejected([&] { schedule.commit(token); });
    auto state = schedule.snapshot();
    state.cursor = 1;
    rejected([&] { schedule.restore(state); });
    state = schedule.snapshot();
    state.updates = 1;
    rejected([&] { schedule.restore(state); });
    state = schedule.snapshot();
    state.cursor = 8;
    rejected([&] { schedule.restore(state); });
    state = schedule.snapshot();
    state.config.order_seed++;
    rejected([&] { schedule.restore(state); });
    state = schedule.snapshot();
    state.episodes[0].id = 100;
    rejected([&] { schedule.restore(state); });
    require(serialize_schedule(schedule.snapshot()) == original,
            "Un fallo modifica el estado confirmado");
    rejected([&] { static_cast<void>(deserialize_schedule(original + " basura")); });
    rejected(
        [&] { static_cast<void>(deserialize_schedule(original.substr(0, original.size() / 2))); });
    rejected([&] { static_cast<void>(deserialize_schedule("999 0")); });
}
void independent_rng_and_many_shapes() {
    std::mt19937_64 external(9);
    const auto external_before = external;
    for (std::size_t count = 1; count <= 32; ++count) {
        std::vector<MatureEpisode> data;
        data.reserve(count);
        for (std::size_t index = 0; index < count; ++index)
            data.push_back({index + 1, 0, 1, 3});
        auto settings = config(ReplayOrder::spaced);
        settings.minimum_distance = count;
        ReplaySchedule schedule(settings, data);
        const auto spaced = consume(schedule);
        for (std::size_t index = count; index < spaced.size(); ++index)
            require(spaced[index] == spaced[index - count],
                    "Se pierde la separación de una ronda completa");
        settings.order = ReplayOrder::uniform;
        ReplaySchedule left(settings, data);
        ReplaySchedule right(settings, data);
        require(consume(left) == consume(right), "La misma semilla no reproduce el calendario");
    }
    require(external == external_before, "El calendario consume un stream ajeno");
    auto left_settings = config();
    auto right_settings = config();
    right_settings.order_seed++;
    ReplaySchedule left(left_settings, episodes());
    ReplaySchedule right(right_settings, episodes());
    require(consume(left) != consume(right), "La semilla de orden no tiene efecto");
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
} // namespace
int main() {
    try {
        exact_multisets_and_spacing();
        recovered_cursor_is_transactional();
        invalid_inputs_leave_state_unchanged();
        independent_rng_and_many_shapes();
        std::cout << "Calendarios de replay comprobados\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
