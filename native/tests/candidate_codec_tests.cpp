#include "mars_titan/candidate.hpp"

#include <ATen/ATen.h>
#include <ATen/CPUGeneratorImpl.h>
#include <ATen/Parallel.h>

#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

// Entradas manuales para propiedad, normalización y presupuesto del codec.
// NOLINTBEGIN(cppcoreguidelines-avoid-magic-numbers)
namespace {
using namespace mars_titan::candidate;
void require(bool condition, const char* message) {
    if (!condition)
        throw std::runtime_error(message);
}
// Cada rechazo debe proceder de la guarda indicada, no de cualquier excepción.
template <class F> void rejected(F action, std::string_view reason) {
    try {
        action();
    } catch (const std::invalid_argument& error) {
        if (std::string_view(error.what()).find(reason) == std::string_view::npos) {
            throw std::runtime_error("Rechazo inesperado: " + std::string(error.what()));
        }
        return;
    }
    throw std::runtime_error("El codec aceptó una entrada incompatible: " + std::string(reason));
}
Config config() {
    Config result;
    result.dimensions = {5, 3, 4, 3, 3};
    result.normalization_id = "manual-cpu-codec-v1";
    return result;
}
Inputs inputs(at::ScalarType dtype, int64_t rows = 1) {
    const auto options = at::TensorOptions().dtype(dtype).device(at::kCPU);
    return {at::zeros({rows, 64, 5}, options), at::zeros({rows, 3}, options),
            at::zeros({rows, 4}, options),     at::zeros({rows, 3}, options),
            at::zeros({rows, 3}, options),     at::ones({rows, 5}, options.dtype(at::kBool))};
}
void boundaries(const CpuEpisodeCodec& codec, at::ScalarType dtype) {
    rejected([&] { (void)codec.encode(inputs(dtype, 0)); }, "El lote supera el presupuesto");
    rejected([&] { (void)codec.encode(inputs(dtype, maximum_batch + 1)); },
             "El lote supera el presupuesto");
    const auto other = dtype == at::kDouble ? at::kFloat : at::kDouble;
    rejected([&] { (void)codec.encode(inputs(other)); }, "Forma, tipo o dispositivo");
    auto data = inputs(dtype);
    data.macro[0][0].fill_(std::numeric_limits<double>::infinity());
    rejected([&] { (void)codec.encode(data); }, "NaN o infinito");
    data = inputs(dtype);
    data.prices[0][0][0].fill_(std::numeric_limits<double>::quiet_NaN());
    rejected([&] { (void)codec.encode(data); }, "NaN o infinito");
    data = inputs(dtype);
    data.presence[0][1].fill_(false);
    rejected([&] { (void)codec.encode(data); }, "Se requieren las cuatro modalidades");
}
void run(at::ScalarType dtype) {
    auto data = inputs(dtype);
    data.prices[0][0][0].fill_(1);
    std::shared_ptr<CpuEpisodeCodec> codec;
    EpisodeEncoding reference;
    {
        const Candidate model(config(), dtype);
        const auto before_rng = at::detail::getDefaultCPUGenerator().get_state().clone();
        codec = model.cpu_episode_codec();
        const auto encoded = model.encode(data);
        reference = {encoded.episode_keys, encoded.episode_features};
        require(at::equal(codec->encode(data).keys, reference.keys),
                "El codec difiere de la fórmula original de una fila");
        require(at::equal(model.encode_context(data), encoded.fused),
                "La extracción cambia el contexto de la GRU");
        require(at::equal(before_rng, at::detail::getDefaultCPUGenerator().get_state()),
                "El codec consumió el RNG global");
        const auto tight = model.cpu_episode_codec(codec->estimated_bytes(1) - 1);
        rejected([&] { (void)tight->encode(data); }, "El codec supera el presupuesto de trabajo");
        rejected([&] { (void)model.cpu_episode_codec(1); }, "Las proyecciones exceden");
        rejected([&] { (void)model.cpu_episode_codec(maximum_codec_bytes + 1); },
                 "Las proyecciones exceden");
    }
    auto encoded = codec->encode(data);
    require(at::equal(encoded.values, reference.values), "El codec dependía del propietario");
    require(!encoded.keys.requires_grad() && !encoded.values.requires_grad() &&
                encoded.keys.scalar_type() == dtype && encoded.values.scalar_type() == dtype,
            "La codificación conserva grafo o cambia la precisión");
    encoded.values.zero_();
    encoded.keys.zero_();
    require(at::equal(codec->encode(data).keys, reference.keys), "Un output modificó el codec");
    data.prices.zero_();
    require((codec->encode(data).keys == 0).all().item<bool>(), "El cero no se conserva");
    for (double value : {1e-30, dtype == at::kDouble ? 1e-200 : 1e-40}) {
        data.prices[0][0][0].fill_(value);
        rejected([&] { (void)codec->encode(data); }, "no es unitaria ni exactamente nula");
    }
    boundaries(*codec, dtype);
}
} // namespace
int main() {
    at::set_num_threads(2);
    at::set_num_interop_threads(2);
    try {
        run(at::kFloat);
        run(at::kDouble);
        std::cout << "Codec CPU, propiedad, epsilon y presupuestos correctos\n";
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
// NOLINTEND(cppcoreguidelines-avoid-magic-numbers)
