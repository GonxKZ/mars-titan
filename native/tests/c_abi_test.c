#include "mars_titan/simulation.h"

#include <stdio.h>

int main(void) {
    const struct mt_layout_v1 layout = mt_simulation_layout_v1();
    char error[256] = {0};
    if (layout.abi_version != 1 || layout.position_size != sizeof(struct mt_position_v1)
        || layout.account_size != sizeof(struct mt_account_v1)
        || layout.trade_size != sizeof(struct mt_trade_v1)) {
        fputs("El consumidor C no comparte el contrato binario\n", stderr);
        return 1;
    }
    if (mt_simulation_step_v1(0, 0, NULL, NULL, NULL, NULL, NULL, NULL, 0, 0,
                             0, 0, 0, NULL, NULL, NULL, error, sizeof(error)) != MT_SIM_INVALID_ARGUMENT) {
        fputs("El consumidor C debe recibir un error para dimensiones vacías\n", stderr);
        return 1;
    }
    return 0;
}
