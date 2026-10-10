#include "mars_titan/group_waves.hpp"
#include "mars_titan/klpo_collection.hpp"

#include <ATen/ATen.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>

namespace mars_titan::learning {
namespace {
void require(bool value, std::string_view message) {
    if (!value) {
        throw std::invalid_argument(std::string(message));
    }
}

// Clave del estado inicial de un episodio. Dos episodios con la misma cinta, el mismo contexto,
// el mismo prefijo forzado y el mismo primer cierre empiezan en el mismo estado del entorno.
using StartKey = std::tuple<std::string, std::string, std::size_t, int64_t>;

StartKey start_key(const KlpoEpisodeRecord& episode) {
    require(!episode.steps.empty() && !episode.spec.close_times.empty(),
            "Un episodio de grupo necesita al menos un paso");
    return {episode.spec.source_sha256, episode.spec.context_sha256, episode.spec.forced_prefix,
            episode.spec.close_times.front()};
}

std::size_t sampled(const KlpoEpisodeRecord& episode) {
    return static_cast<std::size_t>(
        std::ranges::count_if(episode.steps, [](const auto& step) { return step.sampled; }));
}
} // namespace

GroupWave group_wave(const KlpoEpisodeBatch& wave, const GroupObjectiveConfig& config) {
    config.validate();
    // Con gamma distinto de uno el retorno descontado ya no es el resultado del episodio, que
    // es la recompensa de resultado que suponen las cuatro fuentes.
    require(wave.gamma == 1, "Los objetivos de grupo necesitan el retorno sin descontar");
    const auto returns = klpo_terminal_returns(wave);
    GroupWave result;
    std::map<StartKey, int64_t> labels;
    std::vector<const KlpoEpisodeRecord*> first;
    std::vector<int64_t> decisions;
    decisions.reserve(wave.episodes.size());
    for (const auto& episode : wave.episodes) {
        const auto [entry, inserted] =
            labels.try_emplace(start_key(episode), static_cast<int64_t>(labels.size()));
        if (inserted) {
            first.push_back(&episode);
            result.sizes.push_back(0);
        }
        const auto label = entry->second;
        // Defensa adversarial: un grupo con observaciones iniciales distintas compararía
        // estados diferentes y la media del grupo dejaría de ser una línea base válida.
        require(first[static_cast<std::size_t>(label)]->steps.front().observation ==
                    episode.steps.front().observation,
                "Un grupo contiene episodios que no parten de la misma observación");
        result.labels.push_back(label);
        ++result.sizes[static_cast<std::size_t>(label)];
        const auto count = sampled(episode);
        result.sampled_decisions += count;
        decisions.push_back(static_cast<int64_t>(count));
    }
    result.returns = at::tensor(returns, at::kDouble);
    result.advantages = group_advantages(result.returns, at::tensor(result.labels, at::kLong),
                                         config.advantage(), config.advantage_epsilon);
    result.weights = group_episode_weights(at::tensor(decisions, at::kLong), config);
    return result;
}

at::Tensor group_block_loss(const PpoPolicy& actor, const KlpoEpisodeBatch& block,
                            std::size_t begin, const GroupWave& wave,
                            const GroupObjectiveConfig& config, GroupObjectiveTrace* trace) {
    require(begin + block.episodes.size() <= wave.labels.size(),
            "El bloque sale de la oleada de grupo");
    const auto decisions = terminal_decisions(actor, block);
    if (decisions.active.empty()) {
        return {};
    }
    std::vector<int64_t> rows;
    rows.reserve(decisions.active.size());
    for (const auto index : decisions.active) {
        rows.push_back(static_cast<int64_t>(begin + index));
    }
    const auto selected = at::tensor(rows, at::kLong);
    const at::Device device(actor.device());
    const auto advantages = wave.advantages.index_select(0, selected).to(device);
    const auto weights = wave.weights.index_select(0, selected).to(device);
    return group_relative_loss(decisions.logp, decisions.logq, decisions.actions, advantages,
                               weights, decisions.mask, config, trace)
        .sum();
}

GroupWaveTrace group_wave_trace(const GroupWave& wave) {
    GroupWaveTrace result;
    result.groups = wave.sizes.size();
    result.smallest_group = *std::ranges::min_element(wave.sizes);
    result.largest_group = *std::ranges::max_element(wave.sizes);
    const auto returns = wave.returns.contiguous();
    const std::span<const double> values(returns.const_data_ptr<double>(), wave.labels.size());
    std::vector<double> totals(result.groups, 0);
    for (std::size_t index = 0; index < wave.labels.size(); ++index) {
        totals[static_cast<std::size_t>(wave.labels[index])] += values[index];
    }
    std::vector<double> squares(result.groups, 0);
    for (std::size_t index = 0; index < wave.labels.size(); ++index) {
        const auto label = static_cast<std::size_t>(wave.labels[index]);
        const auto centered =
            values[index] - totals[label] / static_cast<double>(wave.sizes[label]);
        squares[label] += centered * centered;
    }
    for (std::size_t label = 0; label < result.groups; ++label) {
        const auto spread = std::sqrt(squares[label] / static_cast<double>(wave.sizes[label] - 1));
        result.return_spread += spread / static_cast<double>(result.groups);
        result.flat_groups += spread == 0 ? 1 : 0;
    }
    const auto advantages = wave.advantages;
    result.objective.episodes = wave.labels.size();
    result.objective.decisions = wave.sampled_decisions;
    result.objective.advantage_mean = advantages.mean().item<double>();
    result.objective.advantage_std = advantages.std(/*unbiased=*/false).item<double>();
    result.objective.advantage_min = advantages.min().item<double>();
    result.objective.advantage_max = advantages.max().item<double>();
    return result;
}
} // namespace mars_titan::learning
