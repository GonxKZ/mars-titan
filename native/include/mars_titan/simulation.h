#ifndef MARS_TITAN_SIMULATION_H
#define MARS_TITAN_SIMULATION_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(MARS_TITAN_SIMULATION_BUILD) || defined(mars_titan_simulation_EXPORTS)
#define MT_SIM_API __declspec(dllexport)
#else
#define MT_SIM_API __declspec(dllimport)
#endif
#else
#define MT_SIM_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* C17 no admite enum class ni tipos base de enumeración. Son constantes de la interfaz C. */
// NOLINTNEXTLINE(cppcoreguidelines-use-enum-class,performance-enum-size)
enum mt_sim_status_v1 {
    MT_SIM_OK = 0,
    MT_SIM_INVALID_ARGUMENT = 1,
    MT_SIM_INVALID_STATE = 2,
    MT_SIM_NUMERICAL_ERROR = 3
};

/* Mantener estas constantes disponibles para los consumidores C17. */
// NOLINTNEXTLINE(cppcoreguidelines-use-enum-class,performance-enum-size)
enum mt_order_reason_v1 {
    MT_ORDER_COMPLETE = 0,
    MT_ORDER_MISSING_OPEN = 1,
    MT_ORDER_UNKNOWN_LIQUIDITY = 2,
    MT_ORDER_RESTRICTED = 3,
    MT_ORDER_LIMIT_UP = 4,
    MT_ORDER_LIMIT_DOWN = 5
};

/* NaN en target indica ausencia de orden, en capacity indica liquidez desconocida. */
struct mt_position_v1 {
    double quantity;
    double target;
    double capacity;
    int64_t decision_at;
};

struct mt_account_v1 {
    double cash;
    double costs;
    double turnover;
    double receivable;
    double nav;
};

struct mt_trade_v1 {
    double quantity;
    double price;
    double cost;
    int32_t reason;
    uint32_t reserved;
};

/*
 * Reglas de un activo vigentes en la apertura de una ejecución. reference es el cierre anterior
 * ajustado por los eventos de esa apertura, NaN o no positivo sin límite diario, y band 0 indica
 * que no hay banda en esa fecha. Los impuestos se cobran sobre el efectivo negociado además de
 * cost_rate. odd_lot_exit admite vender de una vez el resto inferior al lote.
 */
struct mt_rules_v1 {
    double minimum_order;
    double reference;
    double band;
    double buy_tax;
    double sell_tax;
    uint32_t odd_lot_exit;
    uint32_t reserved;
};

/* Contrato binario para contrastar tamaños antes de compartir memoria. */
struct mt_layout_v1 {
    uint32_t abi_version;
    uint32_t position_size;
    uint32_t account_size;
    uint32_t trade_size;
};

MT_SIM_API struct mt_layout_v1 mt_simulation_layout_v1(void);

/*
 * Los arrays se prestan durante la llamada y deben permanecer vivos y sin cambios concurrentes.
 * prices contiene asset_count filas OHLCV contiguas. Los demás arrays de activos tienen esa
 * longitud. Las cuentas tienen account_count filas. Los destinos son buffers separados de todas las
 * entradas. Se admiten hasta 4096 activos y 32 monedas. No se conserva ningún puntero ni hay estado
 * global mutable. Un error conserva las entradas. Los destinos solo son válidos si se devuelve
 * MT_SIM_OK. Las acciones corporativas y los derechos de cobro deben estar conciliados antes de
 * esta operación.
 */
MT_SIM_API int mt_simulation_step_v1(
    uint32_t asset_count, uint32_t account_count, const uint32_t *currencies, const double *lots,
    const uint8_t *retired, const double *prices, const struct mt_position_v1 *previous_positions,
    const struct mt_account_v1 *previous_accounts, double cost_rate, double participation,
    int64_t previous_close, int64_t open_at, int64_t close_at,
    struct mt_position_v1 *next_positions, struct mt_account_v1 *next_accounts,
    struct mt_trade_v1 *trades, char *error, size_t error_capacity);

MT_SIM_API uint32_t mt_simulation_rules_size_v1(void);

/*
 * Igual que v1 con reglas de mercado por activo. rules contiene asset_count filas o es NULL, y
 * entonces el resultado es idéntico al de v1. Con reglas, una compra no se ejecuta si la apertura
 * alcanza el límite superior y una venta tampoco si alcanza el inferior. Las compras se redondean
 * al lote y descartan las inferiores al mínimo. Las ventas admiten el resto impar si se declara.
 */
MT_SIM_API int mt_simulation_step_v2(
    uint32_t asset_count, uint32_t account_count, const uint32_t *currencies, const double *lots,
    const struct mt_rules_v1 *rules, const uint8_t *retired, const double *prices,
    const struct mt_position_v1 *previous_positions,
    const struct mt_account_v1 *previous_accounts, double cost_rate, double participation,
    int64_t previous_close, int64_t open_at, int64_t close_at,
    struct mt_position_v1 *next_positions, struct mt_account_v1 *next_accounts,
    struct mt_trade_v1 *trades, char *error, size_t error_capacity);

/*
 * Precios límite con la aritmética decimal de la referencia Python: la representación decimal
 * más corta de reference por 1 ± band, redondeada a 0,01 por la mitad hacia arriba. Devuelve NaN
 * en ambos límites si band es 0 o reference es NaN o no positiva.
 */
MT_SIM_API int mt_simulation_price_limits_v1(double reference, double band, double *upper,
                                             double *lower, char *error, size_t error_capacity);

/* Observación de una cuenta: seis valores por activo, seguidos de efectivo y derechos de cobro. */
MT_SIM_API int mt_simulation_observation_v1(uint32_t asset_count, const double *prices,
                                            const double *previous_prices, const double *scores,
                                            const uint8_t *retired,
                                            const struct mt_position_v1 *positions, double nav,
                                            double cash, double score_scale, float *observation,
                                            char *error, size_t error_capacity);

#ifdef __cplusplus
}
#endif

#endif
