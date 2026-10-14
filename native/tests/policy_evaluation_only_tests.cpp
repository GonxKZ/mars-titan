#include "mars_titan/klpo_collection.hpp"
#include "mars_titan/klpo_learning.hpp"
#include "mars_titan/learning_hold.hpp"
#include "mars_titan/policy_evaluation.hpp"
#include "mars_titan/ppo_training.hpp"

#include <nlohmann/json.hpp>
#include <torch/cuda.h>

#include <array>
#include <cstddef>
#include <filesystem>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include <unistd.h>

// Modo de solo evaluación de las políticas. La evaluación sin entrenador debe construir la misma
// política inicial que el primer paso de un ajuste, y el cerrojo debe rechazar entrenadores,
// controladores y Adam terminal sin impedir la evaluación congelada. Ninguna prueba avanza un
// entrenador ni da un paso de optimizador. El dispositivo llega como argumento: `cpu` en la
// prueba ordinaria y `cuda:0` en la que CMake registra para una ventana GPU exclusiva.
namespace {
using namespace mars_titan::learning;
using namespace mars_titan::simulation;
constexpr std::size_t digest_size = 64;
constexpr std::size_t sessions = 9;
constexpr double price = 10;
constexpr double volume = 1e6;
constexpr double prediction = 0.01;
constexpr double capital = 10'000;
constexpr double cost_bps = 10;

void require(bool condition, const char* reason) {
    if (!condition) {
        throw std::runtime_error(reason);
    }
}

// La orden debe fallar con el error del cerrojo, no con otro anterior.
template <typename Action> void refused(Action&& action, const char* reason) {
    try {
        std::forward<Action>(action)();
    } catch (const std::logic_error& error) {
        if (std::string_view(error.what()).starts_with("Modo de solo evaluación")) {
            return;
        }
        throw;
    }
    throw std::runtime_error(reason);
}

std::shared_ptr<MarketTape> tape(std::string partition, char hash) {
    auto data = std::make_shared<MarketTape>();
    data->assets = {"FIC0", "FIC1"};
    data->currency = "USD";
    data->domain = "synthetic";
    data->partition = std::move(partition);
    data->source_sha256.assign(digest_size, hash);
    data->parent_id = "analytic-frozen";
    for (std::size_t at = 0; at < sessions; ++at) {
        data->open_times.push_back(static_cast<int64_t>(2 * at));
        data->close_times.push_back(static_cast<int64_t>(2 * at + 1));
        data->prediction_times.push_back(static_cast<int64_t>(2 * at + 1));
        for (std::size_t asset = 0; asset < 2; ++asset) {
            const auto value = price + static_cast<double>(at + asset);
            data->prices.insert(data->prices.end(), {value, value, value, value, volume});
            data->scores.push_back(prediction * static_cast<double>(asset + 1));
        }
    }
    return data;
}

Parameters parameters() {
    Parameters result;
    result.capital = capital;
    result.cost_bps = cost_bps;
    result.participation = 1;
    return result;
}

std::vector<BatchInput> sources(std::string partition) {
    return {{tape(partition, 'a'), parameters(), {}}, {tape(partition, 'b'), parameters(), {}}};
}

PpoTrainingConfig training() {
    PpoTrainingConfig result;
    result.total_transitions = 32;
    result.rollout_transitions = 16;
    return result;
}

PpoHyperparameters hyperparameters() {
    PpoHyperparameters result;
    result.epochs = 2;
    result.minibatch_size = 4;
    return result;
}

// El esquema 4 ajusta con el contexto adaptativo, sin máscara ni HMM, y estas variantes.
PpoLearningOptions adaptive(std::string_view variant) {
    PpoLearningOptions result;
    result.enabled = true;
    result.environments = 2;
    result.variant = variant;
    return result;
}

void the_initial_ppo_policy_is_the_one_of_the_trainer(const std::string& device) {
    const auto config = training();
    const auto fit = sources("train");
    const auto validation = sources("validation");
    // El diagnóstico de 32 transiciones solo existe en CPU.
    const bool diagnostic = device == "cpu";
    std::vector<PpoLearningOptions> catalogue{PpoLearningOptions{}};
    for (const auto* variant : {"ppo", "double_dqn", "qr_dqn", "qr_dqn_cvar"}) {
        catalogue.push_back(adaptive(variant));
    }
    for (const auto& learning : catalogue) {
        const PpoTrainer trainer(fit, config, hyperparameters(), device, diagnostic, learning);
        const auto width = policy_observation_width(fit, learning, config.seed, config.workers);
        const auto initial =
            initial_ppo_policy(width, hyperparameters(), config.seed, device, learning, {});
        require(width == trainer.policy().observation_width() &&
                    initial.parameter_fingerprint() == trainer.policy().parameter_fingerprint() &&
                    initial.architecture() == trainer.policy().architecture() &&
                    initial.optimizer_steps() == 0,
                "La política inicial sin entrenador no es la del primer paso del ajuste");
        // Los pesos salen del generador de CPU, así que cualquier dispositivo evalúa los mismos.
        require(initial.parameter_fingerprint() ==
                    initial_ppo_policy(width, hyperparameters(), config.seed, "cpu", learning, {})
                        .parameter_fingerprint(),
                "La política inicial cambia con el dispositivo");
        // Las cintas de validación o evaluación con los mismos activos dan la misma anchura.
        require(policy_observation_width(validation, learning, config.seed, config.workers) == width,
                "La anchura de observación depende del papel de la cinta");
    }
}

KlpoCollectionOptions collection(const std::string& device) {
    KlpoCollectionOptions result;
    result.device = device;
    result.fold = "fixed";
    result.context.enabled = true;
    result.context.variant = "ppo";
    result.context.fold = result.fold;
    result.context.environments = 2;
    return result;
}

void the_initial_klpo_actor_is_the_one_of_the_collector(const std::string& device) {
    const auto options = collection(device);
    const auto fit = sources("train");
    const KlpoTerminalCollector collector(fit, options);
    const auto width = policy_observation_width(fit, options.context, options.seed, options.workers);
    const auto actor = initial_klpo_actor(width, options);
    require(width == collector.policy().observation_width() &&
                actor.parameter_fingerprint() == collector.records().reference_sha256 &&
                actor.optimizer_steps() == 0,
            "El actor inicial sin controlador no es el de la recogida de un ajuste");
}

nlohmann::json frozen_identity(const std::vector<double>& costs) {
    return nlohmann::json{{"final_test_opened", false}, {"cost_bps", costs}, {"decisions", false}};
}

// Va al final: el cerrojo no se desactiva y afecta a todo el proceso.
void the_latch_refuses_learners_and_keeps_the_frozen_evaluation(const std::string& device) {
    require(!evaluation_only(), "El cerrojo no debe estar activo antes de pedirlo");
    require_optimizer_steps_allowed("una comprobación previa");
    enter_evaluation_only();
    require(evaluation_only(), "El cerrojo no se activó");
    refused([] { require_optimizer_steps_allowed("un paso"); }, "El cerrojo admitió un paso");
    const auto config = training();
    const auto fit = sources("train");
    const auto learning = adaptive("ppo");
    refused(
        [&] { const PpoTrainer trainer(fit, config, hyperparameters(), device, true, learning); },
            "Se creó un entrenador PPO en modo de solo evaluación");
    const auto width = policy_observation_width(fit, learning, config.seed, config.workers);
    auto policy = initial_ppo_policy(width, hyperparameters(), config.seed, device, learning, {});
    refused([&] { policy.enable_terminal_adam({}); }, "Se creó un Adam terminal");
    // Cada entrada que da pasos se detiene en el cerrojo antes de validar sus argumentos, que
    // aquí están vacíos. Sin la comprobación, fallaría con otro error y la prueba lo vería.
    refused([&] { static_cast<void>(policy.update(PpoRollout{})); }, "PPO actualizó");
    refused([&] { static_cast<void>(policy.update_from_cpu(PpoRollout{})); }, "PPO desde CPU");
    refused([&] { static_cast<void>(policy.terminal_step()); }, "Adam terminal dio un paso");
    refused([&] { static_cast<void>(policy.consolidate({}, {}, 1)); }, "Consolidación auxiliar");
    refused([&] { static_cast<void>(policy.update_double_dqn({}, 0, 1)); }, "Double DQN");
    // La política creada con el cerrojo no construye optimizadores, así que no se guarda, y
    // tampoco se carga un archivo de política, que restauraría su optimizador.
    std::ostringstream bytes;
    bool saved = true;
    try {
        policy.save(bytes);
    } catch (const std::invalid_argument&) {
        saved = false;
    }
    require(!saved && policy.optimizer_steps() == 0, "La política de solo evaluación se guardó");
    refused(
        [] {
            std::istringstream empty;
            static_cast<void>(PpoPolicy::load(empty));
        },
        "Se cargó una política con su optimizador");
    refused([&] { const KlpoLearningController controller(fit, KlpoLearningConfig{}); },
            "Se creó un controlador KLPO en modo de solo evaluación");
    // La evaluación congelada no aprende y sigue disponible.
    const auto root = std::filesystem::temp_directory_path() /
                      ("mars-titan-evaluation-only-" + std::to_string(::getpid()) + "-" +
                       (device == "cpu" ? "cpu" : "cuda"));
    std::filesystem::remove_all(root);
    FrozenEvaluationRequest request;
    request.output = root / "evaluation";
    request.learning = learning;
    request.costs = {0.0, cost_bps};
    request.identity = frozen_identity(request.costs);
    for (const auto& input : sources("validation")) {
        request.tapes.push_back(
            PolicyTape{input, nlohmann::json{{"manifest_sha256", input.tape->source_sha256}}});
    }
    const auto report = run_frozen_evaluation(policy, request);
    require(report.at("status") == "completed" && report.at("metrics").size() == 4 &&
                policy.optimizer_steps() == 0 && evaluation_only(),
            "La evaluación congelada no terminó en modo de solo evaluación");
    std::filesystem::remove_all(root);
}
} // namespace

int main(int argc, char** argv) {
    try {
        const std::string device = argc > 1 ? argv[1] : "cpu";
        require(argc <= 2 && (device == "cpu" || device == "cuda:0"),
                "La prueba recibe como mucho un dispositivo, cpu o cuda:0");
        // La prueba CUDA solo se registra para una ventana GPU, así que sin GPU falla.
        require(device == "cpu" || torch::cuda::is_available(), "cuda:0 no está disponible");
        the_initial_ppo_policy_is_the_one_of_the_trainer(device);
        the_initial_klpo_actor_is_the_one_of_the_collector(device);
        the_latch_refuses_learners_and_keeps_the_frozen_evaluation(device);
        std::cout << "Políticas iniciales y cerrojo de solo evaluación comprobados sin pasos en "
                  << device << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
    return 0;
}
