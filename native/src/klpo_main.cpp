#include "mars_titan/klpo_experiment.hpp"
#include "mars_titan/policy_command.hpp"

int main(int argc, char** argv) {
    const mars_titan::learning::PolicyCommand command{
        "mars-titan-klpo",
        "mars-titan-klpo --config ARCHIVO --output DIR --train-tape DIR [--train-tape DIR ...] "
        "--validation-tape DIR [--resume] [--stop-after N]\n"
        "  Diagnóstico: --device cpu --diagnostic\n"
        "  CUDA: --device cuda:0 --gpu-lease-fd FD "
        "--vram-budget-bytes N --vram-total-bytes N\n"
        "  Evaluación separada: --audit-run RUN --audit-tape DIR [--evaluation-cost PB ...] "
        "(sin --train-tape ni --validation-tape). Publica el patrimonio por sesión.\n"
        "  Vigilancia Linux: --parent-pid PID detiene el hijo si muere su padre.\n"
        "  Capacidades: --capabilities, sin leer datos.\n"
        "KLPO terminal sobre cintas reconstruidas por ventana, con selección en validación y "
        "sin abrir el test.\n",
        {"native_policy_reconstructed_tapes", "native_klpo_financial_runner",
         "native_policy_equity_and_costs"},
        [](const auto& options, const auto& stop) {
            return mars_titan::learning::run_klpo_experiment(options, stop);
        }};
    return mars_titan::learning::run_policy_command(argc, argv, command);
}
