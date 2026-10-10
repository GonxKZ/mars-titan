#include "mars_titan/candidate.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>
#include <torch/serialize/archive.h>

#include <array>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>

// Valores manuales para verificar la política, sin estimar objetivos ni ajustar pesos.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
namespace {
using namespace mars_titan::candidate;
void require(bool value, std::string_view reason) {
    if (!value) {
        throw std::runtime_error(std::string(reason));
    }
}
template <class F> void rejected(F action) {
    bool failed = false;
    try {
        action();
    } catch (const std::exception&) {
        failed = true;
    }
    require(failed, "Se aceptó un contrato incompatible");
}
Config config() {
    Config value;
    value.dimensions = {5, 3, 4, 6, 9};
    value.normalization_id = "manual-historical-inputs-v1";
    value.max_batch = 4;
    value.input_policy = "historical_masked_2000_v1";
    return value;
}
Inputs inputs(bool complete = false) {
    Inputs result{at::linspace(-1., 1., 2 * price_window * 5, at::kDouble).reshape({2, 64, 5}),
                  at::zeros({2, 3}, at::kDouble),
                  at::ones({2, 4}, at::kDouble),
                  at::zeros({2, 6}, at::kDouble),
                  at::zeros({2, 9}, at::kDouble),
                  at::ones({2, 5}, at::kBool)};
    result.fundamentals.narrow(1, 2, 2).fill_(1);
    result.macro.narrow(1, 3, 3).fill_(1);
    if (!complete) {
        result.presence.select(0, 1).select(0, 1).fill_(false);
        result.presence.select(0, 1).select(0, 4).fill_(false);
        result.macro.select(0, 1).zero_();
    }
    return result;
}
void legitimate_absences_and_new_codec() {
    const Candidate model(config(), at::kDouble);
    const auto data = inputs();
    const auto result = model.forward(data, model.empty_memory());
    require(at::isfinite(result.quantiles).all().item<bool>(), "Cuantiles no finitos");
    require(model.named_parameters()["fusion_weight"].size(1) == 645,
            "La fusión no recibe los cinco bits");
    auto present_zero = inputs();
    present_zero.presence.select(0, 1).select(0, 1).fill_(true);
    require(
        !at::equal(result.encoded.episode_features, model.encode(present_zero).episode_features),
        "El codec confunde ausencia con un cero observado");
}
void invalid_concepts_are_rejected() {
    const Candidate model(config(), at::kDouble);
    auto data = inputs(true);
    data.fundamentals.select(1, 2).fill_(0.5);
    rejected([&] { (void)model.encode(data); });
    data = inputs(true);
    data.macro.select(1, 6).fill_(-1e-50);
    rejected([&] { (void)model.encode(data); });
    data = inputs(true);
    data.fundamentals.select(1, 2).zero_();
    data.fundamentals.select(1, 0).fill_(1e-50);
    rejected([&] { (void)model.encode(data); });
}
void historical_archive_requires_opt_in() {
    const Candidate model(config(), at::kDouble);
    std::stringstream archive;
    model.save_state(archive);
    torch::serialize::InputArchive metadata;
    metadata.load_from(archive, at::Device(at::kCPU));
    at::Tensor schema;
    metadata.read("schema", schema, true);
    require(schema[0].item<int64_t>() == 3, "La edición histórica no tiene archivo propio");
    archive.clear();
    archive.seekg(0);
    rejected([&] { (void)Candidate::load_state(archive); });
    archive.clear();
    archive.seekg(0);
    const auto restored =
        Candidate::load_state(archive, at::Device(at::kCPU), "historical_masked_2000_v1");
    require(restored->config() == model.config(), "La carga altera la política histórica");
    require(at::equal(model.forward(inputs(), model.empty_memory()).quantiles,
                      restored->forward(inputs(), restored->empty_memory()).quantiles),
            "La recuperación cambia la siguiente predicción");
}
void gates_follow_projection_and_exclude_absent_gradients() {
    Candidate model(config(), at::kDouble);
    auto data = inputs();
    data.news.set_requires_grad(true);
    const auto before = model.encode(data);
    {
        const at::NoGradGuard guard;
        model.named_parameters()["news_bias"].add_(5);
    }
    const auto after = model.encode(data);
    require(at::equal(before.fused[1], after.fused[1]), "Se filtró el sesgo de una ausencia");
    require(!at::equal(before.fused[0], after.fused[0]), "Se ocultó la noticia presente");
    after.fused[1].sum().backward();
    require(data.news.grad().abs().sum().item<double>() == 0,
            "La ausencia recibe gradiente de su proyección");
    require(!after.episode_keys.requires_grad() && !after.episode_features.requires_grad(),
            "El codec retiene el grafo del predictor");
}
void invalid_presence_and_padding_fail_before_masking() {
    const Candidate model(config(), at::kDouble);
    for (const double value : {1e-50, std::numeric_limits<double>::infinity(),
                               std::numeric_limits<double>::quiet_NaN()}) {
        auto data = inputs();
        data.news[1][0].fill_(value);
        rejected([&] { (void)model.encode(data); });
    }
    for (const int64_t required : {0, 2}) {
        auto data = inputs();
        data.presence.select(1, required).fill_(false);
        rejected([&] { (void)model.encode(data); });
    }
    auto data = inputs();
    data.presence = data.presence.to(at::kDouble);
    rejected([&] { (void)model.encode(data); });
    data = inputs();
    data.presence[1][4].fill_(true);
    rejected([&] { (void)model.encode(data); });
    data = inputs(true);
    data.fundamentals[0][2].fill_(1.00000001);
    rejected([&] { (void)model.encode(data); });
}
void transfer_preserves_full_rows_and_fixed_destination_codec() {
    auto strict_config = config();
    strict_config.input_policy = "strict_inputs_v1";
    Candidate strict(strict_config, at::kDouble);
    Candidate masked(config(), at::kDouble);
    const auto source_id = strict.parameter_fingerprint();
    const auto codec_id = masked.representation_id();
    const auto codec = masked.encode(inputs(true)).episode_features;
    const auto before_rng = at::detail::getDefaultCPUGenerator().get_state().clone();
    const auto copied = masked.transfer_strict_parameters(strict);
    require(copied.size() == strict.parameters().size(), "Faltan parámetros en el recibo");
    require(strict.parameter_fingerprint() == source_id, "La transferencia cambia el origen");
    require(masked.representation_id() == codec_id &&
                at::equal(codec, masked.encode(inputs(true)).episode_features),
            "La transferencia sustituyó el codec histórico");
    require(at::equal(before_rng, at::detail::getDefaultCPUGenerator().get_state()),
            "La transferencia altera el RNG global");
    for (const int64_t count : {1, 2, 4}) {
        require(at::allclose(strict.forward(inputs(true), strict.empty_memory(), count).quantiles,
                             masked.forward(inputs(true), masked.empty_memory(), count).quantiles,
                             1e-12, 1e-12),
                "La transferencia cambia los cuantiles con todas las entradas presentes");
    }
    {
        const at::NoGradGuard guard;
        masked.named_parameters()["head_bias"].add_(1);
    }
    require(strict.parameter_fingerprint() == source_id, "Los modelos comparten almacenamiento");
    rejected([&] { (void)strict.encode(inputs()); });
    std::stringstream archive;
    strict.save_state(archive);
    rejected([&] {
        (void)Candidate::load_state(archive, at::Device(at::kCPU), "historical_masked_2000_v1");
    });
}
void incompatible_transfer_is_atomic() {
    Candidate target(config(), at::kDouble);
    const auto before = target.parameter_fingerprint();
    auto altered = config();
    altered.input_policy = "strict_inputs_v1";
    for (const int variant : {0, 1, 2, 3}) {
        auto source_config = altered;
        if (variant == 0)
            source_config.normalization_id = "another-catalog";
        if (variant == 1)
            source_config.dimensions[1] += 1;
        if (variant == 2)
            source_config.input_policy = "historical_masked_2000_v1";
        Candidate source(source_config, variant == 3 ? at::kFloat : at::kDouble);
        rejected([&] { (void)target.transfer_strict_parameters(source); });
        require(before == target.parameter_fingerprint(), "Un rechazo modificó el destino");
    }
}
void historical_configuration_is_bounded() {
    auto invalid = config();
    invalid.input_policy = "unknown";
    rejected([&] { (void)Candidate(invalid); });
    invalid = config();
    invalid.dimensions[3] = 5;
    rejected([&] { (void)Candidate(invalid); });
    invalid = config();
    invalid.dimensions[0] = 4;
    rejected([&] { (void)Candidate(invalid); });
    invalid = config();
    invalid.dimensions = {5, 4096, 4096, 4095, 3777};
    rejected([&] { (void)Candidate(invalid); });
}
void price_presence_channel_is_validated() {
    auto with_presence = config();
    with_presence.dimensions[0] = 6;
    const Candidate model(with_presence, at::kDouble);
    auto data = inputs();
    const auto ones = at::ones({2, price_window, 1}, at::kDouble);
    data.prices = at::cat({data.prices, ones}, 2);
    require(at::isfinite(model.forward(data, model.empty_memory()).quantiles).all().item<bool>(),
            "La ventana con bit de presencia no produce cuantiles finitos");
    auto gap = data;
    gap.prices = data.prices.clone();
    gap.prices.select(1, 10).zero_();
    require(at::isfinite(model.encode(gap).fused).all().item<bool>(),
            "Un hueco con relleno cero y bit nulo debe aceptarse");
    auto filled = gap;
    filled.prices = gap.prices.clone();
    filled.prices.select(1, 10).select(1, 3).fill_(0.25);
    rejected([&] { (void)model.encode(filled); });
    auto negative = gap;
    negative.prices = gap.prices.clone();
    negative.prices.select(1, 10).select(1, 3).fill_(-0.0);
    rejected([&] { (void)model.encode(negative); });
    auto single = data;
    single.prices = data.prices.clone();
    single.prices.narrow(1, 0, price_window - 1).zero_();
    rejected([&] { (void)model.encode(single); });
    auto last = data;
    last.prices = data.prices.clone();
    last.prices.select(1, price_window - 1).zero_();
    rejected([&] { (void)model.encode(last); });
    auto fraction = data;
    fraction.prices = data.prices.clone();
    // El paso se vacía para que solo falle la regla del bit.
    fraction.prices.select(1, 5).zero_();
    fraction.prices.select(1, 5).select(1, 5).fill_(0.5);
    rejected([&] { (void)model.encode(fraction); });
    auto strict = config();
    strict.input_policy = "strict_inputs_v1";
    strict.dimensions[0] = 6;
    rejected([&] { (void)Candidate(strict); });
    auto seven = config();
    seven.dimensions[0] = 7;
    rejected([&] { (void)Candidate(seven); });
}
void fusion_consumes_presence_separately_from_projection() {
    Candidate model(config(), at::kDouble);
    {
        const at::NoGradGuard guard;
        for (auto& parameter : model.parameters())
            parameter.zero_();
        model.named_parameters()["fusion_weight"][0][641].fill_(1);
    }
    const auto result = model.encode(inputs());
    require(result.fused[0][0].item<double>() > 0 && result.fused[1][0].item<double>() == 0,
            "La fusión no conserva el bit de presencia de noticias");
}
} // namespace
int main() {
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    const std::array<std::pair<std::string_view, void (*)()>, 10> tests{
        {{"ausencias y codec", legitimate_absences_and_new_codec},
         {"conceptos", invalid_concepts_are_rejected},
         {"archivo histórico", historical_archive_requires_opt_in},
         {"gates y gradientes", gates_follow_projection_and_exclude_absent_gradients},
         {"presencia y rellenos", invalid_presence_and_padding_fail_before_masking},
         {"transferencia", transfer_preserves_full_rows_and_fixed_destination_codec},
         {"transferencia incompatible", incompatible_transfer_is_atomic},
         {"configuración", historical_configuration_is_bounded},
         {"bits de fusión", fusion_consumes_presence_separately_from_projection},
         {"presencia de precios", price_presence_channel_is_validated}}};
    int failures = 0;
    for (const auto& [name, test] : tests) {
        try {
            test();
            std::cout << "OK " << name << '\n';
        } catch (const std::exception& error) {
            ++failures;
            std::cerr << name << ": " << error.what() << '\n';
        }
    }
    return failures == 0 ? 0 : 1;
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
