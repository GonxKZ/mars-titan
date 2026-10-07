#include "mars_titan/candidate.hpp"

#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <ATen/core/grad_mode.h>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {
using namespace mars_titan::candidate;
constexpr int64_t sample_rows = 2;
void smoke(const std::string& device_name, const std::filesystem::path& destination) {
    if (device_name != "cpu" && device_name != "cuda:0") {
        throw std::invalid_argument("El dispositivo debe indicarse como cpu o cuda:0");
    }
    Config config;
    config.normalization_id = "synthetic-contract-v1";
    const at::Device device(device_name);
    Candidate model(config, at::kDouble, device);
    model.eval();
    const at::NoGradGuard guard;
    const auto options = at::TensorOptions().dtype(at::kDouble).device(device);
    const Inputs inputs{
        at::zeros({sample_rows, price_window, config.dimensions.at(0)}, options),
        at::ones({sample_rows, config.dimensions.at(1)}, options),
        at::ones({sample_rows, config.dimensions.at(2)}, options),
        at::ones({sample_rows, config.dimensions.at(3)}, options),
        at::ones({sample_rows, config.dimensions.at(4)}, options),
        at::ones({sample_rows, modality_count}, options.dtype(at::kBool))};
    const auto before = model.forward(inputs, model.empty_memory());
    std::stringstream buffer;
    model.save_state(buffer);
    const auto restored = Candidate::load_state(buffer, device);
    const auto after = restored->forward(inputs, restored->empty_memory());
    if (!at::equal(before.quantiles, after.quantiles)) {
        throw std::runtime_error("La recuperación altera los cuantiles");
    }
    if (!destination.empty()) {
        if (std::filesystem::symlink_status(destination).type() != std::filesystem::file_type::not_found) {
            throw std::invalid_argument("El destino del archivo ya existe");
        }
        std::ofstream file(destination, std::ios::binary);
        model.save_state(file);
        file.close();
        if (!file) { throw std::runtime_error("No se pudo cerrar el archivo candidato"); }
    }
    std::cout << "{\"schema\":\"candidate-core-smoke-v1\",\"device\":\"" << device_name
              << "\",\"trained\":false,\"rows\":" << sample_rows
              << ",\"quantiles\":" << quantile_count
              << ",\"archive_roundtrip\":true,\"archive_bytes\":" << buffer.str().size() << "}\n";
}
}
int main(int argc, char* argv[]) {
    at::set_num_threads(1);
    at::set_num_interop_threads(1);
    try {
        if (argc < 2 || argc > 3) {
            throw std::invalid_argument("Uso: mars-titan-candidate cpu|cuda:0 [archivo_nuevo]");
        }
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        smoke(argv[1], argc == 3 ? argv[2] : "");
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
