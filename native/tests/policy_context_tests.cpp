#include "mars_titan/policy_context.hpp"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include <array>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
using namespace mars_titan::learning;
using namespace mars_titan::simulation;
constexpr std::size_t lanes = 2;
constexpr std::size_t sessions = 5;
constexpr std::size_t raw_width = 14;
constexpr std::size_t feedback_width = 8;
constexpr std::size_t extra_width = 76;
constexpr std::size_t feature_width = raw_width + extra_width;
constexpr std::size_t memory_offset = raw_width + feedback_width;
constexpr std::size_t memory_present = memory_offset + episodic_memory_width;
constexpr std::size_t markov_offset = feature_width - 2;
constexpr uint64_t seed = 71;
constexpr double price = 10;
constexpr double volume = 1000;
constexpr std::size_t digest_size = 64;
constexpr double tolerance = 1e-6;
constexpr double equal_prior = 0.5;
constexpr double persistence = 0.8;
constexpr double switching = 0.2;
constexpr double fixture_reward = 0.25;

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
    throw std::runtime_error("Se aceptó un contexto inválido");
}
std::vector<BatchInput> inputs() {
    auto tape = std::make_shared<MarketTape>();
    tape->assets = {"FIC0"};
    tape->currency = "USD";
    tape->domain = "synthetic";
    tape->partition = "train";
    tape->parent_id = "fixed";
    tape->source_sha256.assign(digest_size, 'a');
    ContextTape context;
    context.source_sha256.assign(digest_size, 'b');
    context.fields = {{"trade", "1"}, {"signal", "1"}};
    for (std::size_t cursor = 0; cursor < sessions; ++cursor) {
        const auto time = static_cast<int64_t>(cursor) * 2;
        tape->open_times.push_back(time);
        tape->close_times.push_back(time + 1);
        tape->prediction_times.push_back(time + 1);
        tape->prices.insert(tape->prices.end(), {price, price, price, price, volume});
        tape->scores.push_back(default_score_scale);
        context.values.push_back({cursor == 0 ? 0.0F : 1.0F, true, time + 1});
        context.values.push_back({static_cast<float>(cursor), true, time + 1});
    }
    return {{tape, {}, context}, {tape, {}, context}};
}
std::array<float, lanes * raw_width> observations(float value = 1) {
    std::array<float, lanes * raw_width> result{};
    result.fill(value);
    return result;
}
PpoLearningOptions options(std::string variant = "ppo_episodic_hmm") {
    PpoLearningOptions result;
    result.enabled = true;
    result.variant = std::move(variant);
    result.environments = lanes;
    result.trading_field = 0;
    result.markov = MarkovParameters{
        2,      1,     {equal_prior, equal_prior}, {persistence, switching, switching, persistence},
        {0, 2}, {1, 1}};
    result.markov_fields = {1};
    return result;
}
BatchTransition transition() { return {{fixture_reward, -fixture_reward}, {1, 1}, {0, 0}, {0, 0}}; }

void staging_preserves_state_and_bootstrap_sees_mature_memory() {
    const auto sources = inputs();
    PolicyContext context(sources, options(), seed, observations());
    const auto initial = context.observations().clone();
    require(context.observation_width() == feature_width &&
                context.training_mask() == std::vector<uint8_t>({0, 0}),
            "El contexto inicial no conserva dimensiones y máscara");
    const std::array<uint8_t, lanes> actions{2, 3};
    const auto prepared = context.prepare(observations(2), actions, transition()).clone();
    require(at::equal(initial, context.observations()) && context.retrieved().at(0).count == 0,
            "La preparación modifica el contexto confirmado");
    require(prepared[0][static_cast<int64_t>(memory_present)].item<float>() == 1,
            "El bootstrap no consulta el recuerdo madurado en este paso");
    context.cancel();
    require(at::equal(initial, context.observations()), "Cancelar modificó el estado");
    const auto repeated = context.prepare(observations(2), actions, transition()).clone();
    require(at::equal(prepared, repeated), "El candidato no es reproducible después de cancelar");
    context.commit();
    require(at::equal(prepared, context.observations()) && context.retrieved().at(0).count == 1 &&
                context.retrieved().at(0).neighbors[0].record.reward == fixture_reward &&
                context.training_mask() == std::vector<uint8_t>({1, 1}),
            "La confirmación no conserva el candidato y su resultado");
}

void inactive_lanes_and_failed_candidates_leave_state_unchanged() {
    const auto sources = inputs();
    PolicyContext context(sources, options(), seed, observations());
    const auto inactive_before = context.observations()[1].clone();
    const std::array<uint8_t, lanes> actions{2, 3};
    static_cast<void>(
        context.prepare(observations(2), actions, transition(), std::array<uint8_t, lanes>{1, 0}));
    context.commit();
    require(at::equal(inactive_before, context.observations()[1]) &&
                context.retrieved().at(1).count == 0 &&
                context.training_mask() == std::vector<uint8_t>({1, 0}),
            "Una lane inactiva avanzó contexto, filtro o cursor");
    const auto before = context.observations().clone();
    auto invalid = transition();
    invalid.rewards[1] = std::numeric_limits<double>::quiet_NaN();
    rejected([&] { static_cast<void>(context.prepare(observations(3), actions, invalid)); });
    require(at::equal(before, context.observations()),
            "El candidato inválido modificó el contexto");
    rejected([&] { context.commit(); });
}

void snapshot_resumes_exactly_and_reset_starts_a_new_world() {
    const auto sources = inputs();
    PolicyContext first(sources, options(), seed, observations());
    const std::array<uint8_t, lanes> actions{2, 3};
    static_cast<void>(first.prepare(observations(2), actions, transition()));
    first.commit();
    const auto saved = first.snapshot();
    PolicyContext restored(sources, options(), seed, observations());
    restored.restore(saved);
    require(at::equal(first.observations(), restored.observations()),
            "El snapshot no recuperó las observaciones");
    const auto expected = first.prepare(observations(3), actions, transition()).clone();
    const auto actual = restored.prepare(observations(3), actions, transition()).clone();
    require(at::equal(expected, actual), "La continuación no recuperó memoria y filtro");
    first.commit();
    restored.commit();
    const auto previous_scope = restored.memory_scope(0);
    const auto reset_observation = observations();
    restored.reset(0, sources.at(0), std::span(reset_observation).first(raw_width));
    require(restored.retrieved().at(0).count == 0 && restored.retrieved().at(1).count == 2 &&
                restored.training_mask().at(0) == 0 &&
                restored.memory_scope(0).world != previous_scope.world && restored.cursor(0) == 0 &&
                restored.episode(0) == 1 && restored.episode(1) == 0,
            "El reset reutiliza recuerdos futuros o borra otra lane");
    const auto reset_saved = restored.snapshot();
    PolicyContext reset_resumed(sources, options(), seed, observations());
    reset_resumed.restore(reset_saved);
    require(at::equal(restored.observations(), reset_resumed.observations()),
            "La recuperación pierde el nonce y estado de un reset");
    auto different_options = options();
    different_options.fold = "other-fold";
    PolicyContext different(sources, different_options, seed, observations());
    rejected([&] { different.restore(saved); });
    const auto before = restored.observations().clone();
    rejected([&] { restored.restore({saved.archive.substr(0, saved.archive.size() / 2)}); });
    require(at::equal(before, restored.observations()),
            "La recuperación corrupta modificó el contexto");
}

void window_history_is_ordered_and_controls_keep_reserved_slots_empty() {
    const auto sources = inputs();
    PolicyContext context(sources, options("ppo_window"), seed, observations());
    require(context.observation_width() == policy_window * feature_width,
            "La variante de ventana no conserva sus 16 pasos");
    const auto initial = context.observations().clone();
    require(at::count_nonzero(
                initial.narrow(1, 0, static_cast<int64_t>((policy_window - 1) * feature_width)))
                    .item<int64_t>() == 0,
            "La ventana inicial inventa una historia anterior");
    static_cast<void>(
        context.prepare(observations(2), std::array<uint8_t, lanes>{2, 3}, transition()));
    context.commit();
    const auto history = context.observations().view({static_cast<int64_t>(lanes),
                                                      static_cast<int64_t>(policy_window),
                                                      static_cast<int64_t>(feature_width)});
    require(history[0][static_cast<int64_t>(policy_window - 2)][0].item<float>() == 1 &&
                history[0][static_cast<int64_t>(policy_window - 1)][0].item<float>() == 2,
            "La ventana invierte o pierde sus observaciones");
    require(at::count_nonzero(history.select(1, static_cast<int64_t>(policy_window - 1))
                                  .narrow(1, static_cast<int64_t>(memory_offset),
                                          static_cast<int64_t>(episodic_memory_width + 4)))
                    .item<int64_t>() == 0,
            "El control sin memoria utiliza slots de memoria o HMM");
    PpoLearningOptions disabled;
    PolicyContext plain(sources, disabled, seed, observations());
    require(plain.observation_width() == raw_width,
            "El contrato antiguo añade características nuevas");
}

void hmm_uses_published_source_values_and_unknown_variants_fail() {
    const auto sources = inputs();
    PolicyContext context(sources, options(), seed, observations());
    MarkovFilter reference(options().markov.value_or(MarkovParameters{}));
    const std::array<double, 1> value{0};
    const std::array<uint8_t, 1> present{1};
    const std::array<int64_t, 1> available{1};
    reference.step({value, present, available, 1});
    require(
        std::abs(static_cast<double>(
                     context.observations()[0][static_cast<int64_t>(markov_offset)].item<float>()) -
                 reference.probabilities()[0]) < tolerance,
        "El contexto no utiliza las unidades y fechas de la fuente HMM");
    auto invalid = options("unknown");
    rejected([&] { PolicyContext bad(sources, invalid, seed, observations()); });
    invalid = options();
    auto invalid_model = invalid.markov.value_or(MarkovParameters{});
    invalid_model.states = 3;
    invalid.markov = invalid_model;
    rejected([&] { PolicyContext bad(sources, invalid, seed, observations()); });
}

void variants_keep_their_declared_components_and_reject_future_context() {
    const auto sources = inputs();
    const std::array<std::string_view, 7> variants{"ppo",           "double_dqn", "ppo_gru",
                                                   "ppo_episodic",  "ppo_hmm",    "ppo_recent_aux",
                                                   "ppo_replay_aux"};
    for (const auto variant : variants) {
        PolicyContext context(sources, options(std::string(variant)), seed, observations());
        static_cast<void>(
            context.prepare(observations(2), std::array<uint8_t, lanes>{2, 3}, transition()));
        context.commit();
        const bool has_memory =
            variant == "ppo_episodic" || variant == "ppo_recent_aux" || variant == "ppo_replay_aux";
        const bool has_hmm =
            variant == "ppo_hmm" || variant == "ppo_recent_aux" || variant == "ppo_replay_aux";
        require(context.observations()[0][static_cast<int64_t>(memory_present)].item<float>() ==
                    (has_memory ? 1.0F : 0.0F),
                "La variante no conserva su componente episódico declarado");
        const auto posterior =
            context.observations().narrow(1, static_cast<int64_t>(markov_offset), 2);
        require((at::count_nonzero(posterior).item<int64_t>() != 0) == has_hmm,
                "La variante no conserva su componente HMM declarado");
    }
    auto future = sources;
    constexpr int64_t future_date = 100;
    auto future_context = future.at(0).context.value_or(ContextTape{});
    future_context.values.at(1).available_at = future_date;
    future.at(0).context = std::move(future_context);
    rejected([&] { PolicyContext wrong(future, options(), seed, observations()); });
    PolicyContext unchanged(sources, options(), seed, observations());
    const auto before = unchanged.observations().clone();
    rejected([&] {
        const auto raw = observations();
        unchanged.reset(0, future.at(0), std::span(raw).first(raw_width));
    });
    require(at::equal(before, unchanged.observations()),
            "Un reset con datos futuros modificó el contexto confirmado");
    constexpr std::size_t excessive_lanes = 64;
    std::vector<BatchInput> oversized(excessive_lanes, sources.front());
    std::vector<float> raw(excessive_lanes * raw_width, 1);
    auto oversized_options = options();
    oversized_options.environments = excessive_lanes;
    rejected([&] { PolicyContext wrong(oversized, oversized_options, seed, raw); });
}

void memory_audit_preserves_the_full_checkpoint_and_other_features() {
    const auto sources = inputs();
    PolicyContext context(sources, options(), seed, observations());
    static_cast<void>(
        context.prepare(observations(2), std::array<uint8_t, lanes>{2, 3}, transition()));
    context.commit();
    const auto before = context.snapshot();
    const auto original = context.observations().clone();
    const auto audited = context.without_retrieved_memory();
    require(at::count_nonzero(audited.narrow(1, static_cast<int64_t>(memory_offset),
                                             static_cast<int64_t>(episodic_memory_width + 2)))
                    .item<int64_t>() == 0,
            "La auditoría conserva características de la memoria recuperada");
    require(at::equal(original.narrow(1, 0, static_cast<int64_t>(memory_offset)),
                      audited.narrow(1, 0, static_cast<int64_t>(memory_offset))) &&
                at::equal(original.narrow(1, static_cast<int64_t>(markov_offset), 2),
                          audited.narrow(1, static_cast<int64_t>(markov_offset), 2)) &&
                at::equal(original, context.observations()) &&
                before.archive == context.snapshot().archive,
            "La auditoría modificó datos, HMM, estado o RNG");
    PolicyContext control(sources, options("ppo_window"), seed, observations());
    require(at::equal(control.observations(), control.without_retrieved_memory()),
            "La auditoría altera un control con memoria vacía");
}
} // namespace

int main() {
    try {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        staging_preserves_state_and_bootstrap_sees_mature_memory();
        inactive_lanes_and_failed_candidates_leave_state_unchanged();
        snapshot_resumes_exactly_and_reset_starts_a_new_world();
        window_history_is_ordered_and_controls_keep_reserved_slots_empty();
        hmm_uses_published_source_values_and_unknown_variants_fail();
        variants_keep_their_declared_components_and_reject_future_context();
        memory_audit_preserves_the_full_checkpoint_and_other_features();
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    std::cout << "Contexto de política, memoria, ventanas y filtros comprobados\n";
}
