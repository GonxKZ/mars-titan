#include "mars_titan/policy_command.hpp"
#include "mars_titan/ppo_experiment.hpp"

int main(int argc, char** argv) {
    const mars_titan::learning::PolicyCommand command{
        "mars-titan-ppo",
        "mars-titan-ppo --config ARCHIVO --output DIR --train-tape DIR "
        "--validation-tape DIR [--resume] [--stop-after N]\n"
        "  Diagnóstico: --device cpu --diagnostic\n"
        "  CUDA: --device cuda:0 --gpu-lease-fd FD "
        "--vram-budget-bytes N --vram-total-bytes N\n"
        "  Esquema 2: hasta 512 fuentes por partición y 16 entornos activos.\n"
        "  Esquema 4: cintas reconstruidas por ventana walk-forward y una validación.\n"
        "  Auditoría separada: --audit-run RUN --audit-tape DIR "
        "(sin --train-tape ni --validation-tape).\n"
        "  Vigilancia Linux: --parent-pid PID detiene el hijo si muere su padre.\n"
        "  Capacidades: --capabilities, sin leer datos.\n"
        "Entrena y valida PPO en escenarios sintéticos o cintas reconstruidas, sin abrir el "
        "test.\n",
        {"native_policy_reconstructed_tapes"},
        [](const auto& options, const auto& stop) {
            return mars_titan::learning::run_ppo_experiment(options, stop);
        }};
    return mars_titan::learning::run_policy_command(argc, argv, command);
}
