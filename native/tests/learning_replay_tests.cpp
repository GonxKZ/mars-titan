#include "mars_titan/learning_replay.hpp"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {
using namespace mars_titan::learning;
constexpr std::size_t width = 2;
constexpr std::size_t capacity = 3;
constexpr uint64_t seed = 71;
constexpr int64_t total_records = 30;
constexpr std::size_t repetitions = 256;
constexpr int64_t action_count = 6;
constexpr int64_t initial_records = 5;
constexpr int64_t sampling_start = 10;
constexpr float replacement_value = 9;
constexpr std::size_t maximum_width = 32768;
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
template <class Function> void rejected(Function&& action) {
    try {
        std::forward<Function>(action)();
    } catch (const std::exception&) {
        return;
    }
    throw std::runtime_error("Se aceptó una entrada de replay inválida");
}
at::Tensor observation(float value) { return at::tensor({value, value + 1}, at::kFloat); }
void add(LearningReplay& replay, int64_t value, bool next = false) {
    replay.add(observation(static_cast<float>(value)), value % action_count,
               static_cast<double>(value), value % 2 == 0,
               next ? observation(static_cast<float>(value + 1)) : at::Tensor{});
}
void same_data(const ReplaySample& left, const ReplaySample& right) {
    require(at::equal(left.observations, right.observations) &&
                at::equal(left.actions, right.actions) && at::equal(left.rewards, right.rewards) &&
                at::equal(left.terminated, right.terminated) &&
                left.next_observations.defined() == right.next_observations.defined() &&
                (!left.next_observations.defined() ||
                 at::equal(left.next_observations, right.next_observations)),
            "El replay no conserva sus datos y máscaras");
}
void fifo_matches_the_explicit_reference_and_owns_inputs() {
    LearningReplay replay(width, capacity, ReplayMode::recent, true, seed, dqn_replay_bytes);
    for (int64_t value = 0; value < initial_records; ++value)
        add(replay, value, true);
    const auto state = replay.snapshot();
    constexpr std::array<float, 6> expected_observations{3, 4, 4, 5, 2, 3};
    constexpr std::array<float, 6> expected_following{4, 5, 5, 6, 3, 4};
    require(replay.seen() == initial_records && replay.size() == capacity &&
                replay.capacity() == capacity && state.position == 2,
            "El FIFO no conserva tamaño y cursor");
    require(at::equal(state.data.observations,
                      at::tensor(at::ArrayRef<float>(expected_observations.data(),
                                                     expected_observations.size()),
                                 at::kFloat)
                          .view({3, 2})) &&
                at::equal(state.data.next_observations,
                          at::tensor(at::ArrayRef<float>(expected_following.data(),
                                                         expected_following.size()),
                                     at::kFloat)
                              .view({3, 2})) &&
                at::equal(state.data.actions, at::tensor({3, 4, 2}, at::kLong)) &&
                at::equal(state.data.terminated, at::tensor({0, 1, 1}, at::kLong).to(at::kBool)),
            "El FIFO difiere de la referencia de cinco escrituras en tres slots");
    auto original = observation(replacement_value);
    replay.add(original, 1, 1, false, original);
    original.zero_();
    require(replay.observations()[2][0].item<float>() == replacement_value,
            "El replay conserva una vista mutable de la entrada");
    const auto following = replay.snapshot();
    require(at::equal(following.data.observations[2], following.data.next_observations[2]),
            "La escritura con entradas compartidas pierde el estado siguiente");
    const auto views = replay.observations();
    const auto expected_next = views[0].clone();
    replay.add(views[1], 1, 1, false, views[0]);
    require(at::equal(replay.snapshot().data.next_observations[0], expected_next),
            "La escritura sobrescribió una entrada que era una vista del mismo replay");
}
void rng_recovery_and_sampling_are_independent_of_replacement() {
    for (const auto mode : {ReplayMode::recent, ReplayMode::reservoir}) {
        LearningReplay first(width, capacity, mode, false, seed, auxiliary_replay_bytes);
        for (int64_t value = 0; value < sampling_start; ++value)
            add(first, value);
        LearningReplay second(width, capacity, mode, false, seed, auxiliary_replay_bytes);
        second.restore(deserialize_replay(serialize_replay(first.snapshot())));
        const auto selection_before = first.snapshot().selection_rng;
        same_data(first.sample(capacity), second.sample(capacity));
        require(at::equal(selection_before, first.snapshot().selection_rng),
                "El muestreo avanzó el generador de sustitución del reservorio");
        static_cast<void>(first.sample(capacity));
        for (int64_t value = sampling_start; value < total_records; ++value) {
            add(first, value);
            add(second, value);
        }
        same_data(first.snapshot().data, second.snapshot().data);
        const auto saved = first.snapshot();
        const auto expected = first.sample(capacity);
        first.restore(saved);
        same_data(expected, first.sample(capacity));
    }
    LearningReplay sparse(width, learning_replay_batch, ReplayMode::recent, false, seed,
                          auxiliary_replay_bytes);
    add(sparse, 1);
    const auto repeated = sparse.sample();
    require(repeated.observations.size(0) == static_cast<int64_t>(learning_replay_batch) &&
                (repeated.observations.select(1, 0) == 1).all().item<bool>(),
            "El muestreo no utiliza reemplazo cuando solo hay una fila");
}
void invalid_add_restore_and_limits_preserve_the_confirmed_replay() {
    LearningReplay replay(width, capacity, ReplayMode::recent, true, seed, dqn_replay_bytes);
    add(replay, 1, true);
    const auto before = replay.snapshot();
    rejected([&] { replay.add(observation(2), 1, 1, false); });
    rejected([&] { replay.add(observation(2), action_count, 1, false, observation(3)); });
    rejected([&] {
        replay.add(observation(2), 1, std::numeric_limits<double>::infinity(), false,
                   observation(3));
    });
    rejected([&] { replay.add(observation(2), 1, 1, false, observation(3), false); });
    auto with_grad = observation(2).set_requires_grad(true);
    rejected([&] { replay.add(with_grad, 1, 1, false, observation(3)); });
    auto missing = observation(2);
    missing[0] = std::numeric_limits<float>::quiet_NaN();
    rejected([&] { replay.add(missing, 1, 1, false, observation(3)); });
    same_data(before.data, replay.snapshot().data);
    require(at::equal(before.sampling_rng, replay.snapshot().sampling_rng) &&
                at::equal(before.selection_rng, replay.snapshot().selection_rng),
            "Una entrada inválida consume un generador");
    auto wrong = replay.snapshot();
    wrong.position = 0;
    rejected([&] { replay.restore(wrong); });
    wrong = replay.snapshot();
    wrong.data.actions[0] = -1;
    rejected([&] { replay.restore(wrong); });
    wrong = replay.snapshot();
    wrong.sampling_rng = at::zeros({1}, at::kByte);
    rejected([&] { replay.restore(wrong); });
    same_data(before.data, replay.snapshot().data);
    rejected([&] { static_cast<void>(replay.sample(0)); });
    rejected([&] {
        LearningReplay invalid(0, capacity, ReplayMode::recent, false, seed,
                               auxiliary_replay_bytes);
    });
    rejected([&] {
        LearningReplay invalid(width, auxiliary_replay_capacity + 1, ReplayMode::recent, false,
                               seed, auxiliary_replay_bytes);
    });
    rejected([&] {
        LearningReplay invalid(maximum_width, dqn_replay_capacity, ReplayMode::recent, true, seed,
                               dqn_replay_bytes);
    });
    const auto bytes = serialize_replay(before);
    rejected([&] {
        static_cast<void>(deserialize_replay(std::string_view(bytes).substr(0, bytes.size() / 2)));
    });
}
void reservoir_retains_old_and_recent_rows_and_counters() {
    constexpr std::size_t population = 12;
    std::array<std::size_t, population> included{};
    for (std::size_t trial = 0; trial < repetitions; ++trial) {
        LearningReplay replay(1, capacity, ReplayMode::reservoir, false, trial,
                              auxiliary_replay_bytes);
        for (std::size_t id = 0; id < population; ++id)
            replay.add(at::tensor({static_cast<float>(id)}), 0, 1, false);
        const auto state = replay.snapshot();
        require(state.seen == population && state.size == capacity,
                "El reservorio no cuenta los descartes");
        for (int64_t row = 0; row < static_cast<int64_t>(capacity); ++row)
            ++included.at(static_cast<std::size_t>(state.data.observations[row][0].item<float>()));
    }
    constexpr std::size_t minimum = 35;
    constexpr std::size_t maximum = 95;
    require(std::all_of(included.begin(), included.end(),
                        [](auto count) { return count >= minimum && count <= maximum; }),
            "La inclusión del reservorio no es compatible con una muestra uniforme");
}

void sampling_covers_every_stored_row_without_mutating_the_buffer() {
    LearningReplay replay(width, learning_replay_batch, ReplayMode::recent, true, seed,
                          dqn_replay_bytes);
    for (int64_t id = 0; id < static_cast<int64_t>(capacity); ++id)
        add(replay, id, true);
    const auto before = replay.snapshot();
    std::array<std::size_t, capacity> counts{};
    constexpr std::size_t draws = 16;
    constexpr std::size_t minimum = 240;
    constexpr std::size_t maximum = 440;
    for (std::size_t repetition = 0; repetition < draws; ++repetition) {
        const auto batch = replay.sample();
        for (int64_t row = 0; row < static_cast<int64_t>(learning_replay_batch); ++row) {
            const auto id = static_cast<std::size_t>(batch.observations[row][0].item<float>());
            ++counts.at(id);
            require(batch.rewards[row].item<float>() == static_cast<float>(id) &&
                        batch.next_observations[row][0].item<float>() == static_cast<float>(id + 1),
                    "El muestreo desalineó los campos de una transición");
        }
    }
    require(std::all_of(counts.begin(), counts.end(),
                        [](auto count) { return count >= minimum && count <= maximum; }),
            "El muestreo no reparte las filas con reemplazo");
    same_data(before.data, replay.snapshot().data);
    require(!at::equal(before.sampling_rng, replay.snapshot().sampling_rng),
            "Un lote muestreado no avanzó su generador");
}
} // namespace
int main() {
    try {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        fifo_matches_the_explicit_reference_and_owns_inputs();
        rng_recovery_and_sampling_are_independent_of_replacement();
        invalid_add_restore_and_limits_preserve_the_confirmed_replay();
        reservoir_retains_old_and_recent_rows_and_counters();
        sampling_covers_every_stored_row_without_mutating_the_buffer();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Replay FIFO, reservorio, muestreo y recuperación comprobados\n";
}
