#include "mars_titan/candidate.hpp"
#include <ATen/ATen.h>
#include <ATen/Parallel.h>

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {
void check_projection_ownership(mars_titan::candidate::Candidate& model) {
    using namespace mars_titan::candidate;
    const auto& dimensions = model.config().dimensions;
    const auto options = model.parameters().front().options().requires_grad(false);
    const Inputs inputs{at::ones({1, price_window, dimensions.at(0)}, options),
        at::ones({1, dimensions.at(1)}, options), at::ones({1, dimensions.at(2)}, options),
        at::ones({1, dimensions.at(3)}, options), at::ones({1, dimensions.at(4)}, options),
        at::ones({1, modality_count}, options.dtype(at::kBool))};
    const auto keys = model.encode(inputs).episode_keys.clone();
    model.named_parameters()["initial_weight"].data().zero_();
    if (!at::equal(keys, model.encode(inputs).episode_keys)) {
        throw std::runtime_error("Una proyección comparte almacenamiento con un parámetro público");
    }
}
}
int main(int argc, char* argv[]) {
    at::set_num_threads(1);
    if (argc != 3) { return 2; }
    // Los dos argumentos existen después de comprobar argc.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    const std::string expected(argv[1]);
    // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
    std::ifstream source(argv[2], std::ios::binary);
    try {
        const auto model = mars_titan::candidate::Candidate::load_state(source);
        if (expected == "alias") {
            check_projection_ownership(*model);
        } else if (expected != "valid") {
            std::cerr << "Se aceptó el archivo inválido: " << expected << '\n';
            return 1;
        }
        std::cout << model->representation_id() << '\n';
    } catch (const std::exception& error) {
        const std::string_view reason(error.what());
        if (expected == "valid" || expected == "alias" || reason.find(expected) == std::string_view::npos) {
            std::cerr << "Rechazo distinto del esperado: " << error.what() << '\n';
            return 1;
        }
        std::cout << error.what() << '\n';
    }
}
