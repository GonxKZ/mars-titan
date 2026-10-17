#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <ATen/DeviceAccelerator.h>
#include <ATen/Parallel.h>

#include <exception>
#include <iostream>

// Control de asignaciones con LibTorch, sin enlazar los modelos de MARS-TITAN. Los tensores
// de cada producto salen de ámbito antes de consultar el asignador. No vacía sus cachés.
int main() {
    constexpr int repetitions = 30;
    constexpr int rows = 32;
    constexpr int features = 64;
    try {
        at::set_num_threads(1);
        at::set_num_interop_threads(1);
        TORCH_CHECK(at::globalContext().hasCUDA(), "CUDA no disponible");
        at::globalContext().setAllowTF32CuBLAS(false);
        for (int iteration = 0; iteration < repetitions; ++iteration) {
            for (const auto dtype : {at::kFloat, at::kDouble}) {
                const auto options = at::TensorOptions().dtype(dtype).device(at::Device("cuda:0"));
                auto left = at::ones({rows, features}, options).set_requires_grad(true);
                auto right = at::ones({features, rows}, options).set_requires_grad(true);
                auto result = at::mm(left, right);
                result.square().sum().backward();
            }
            at::accelerator::synchronizeDevice(0);
            const auto stats = at::accelerator::getDeviceStats(0);
            std::cout << "{\"iteration\":" << iteration + 1
                      << ",\"allocated_bytes\":" << stats.allocated_bytes[0].current
                      << ",\"reserved_bytes\":" << stats.reserved_bytes[0].current << "}\n";
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
