# Ejemplo del objetivo residual y su reloj

El objetivo principal es el retorno simple entre la apertura y el cierre de la
siguiente sesión, ajustado por el retorno del mercado en ese mismo intervalo.
La fórmula está fijada en el [protocolo](protocol.md). La sección `target` de
[study.toml](../../configs/study.toml) declara una ventana de 252 sesiones y un
mínimo de 126 pares válidos. Este ejemplo reutiliza esa configuración y
[el cálculo residual vigente](../../src/mars_titan/data/residual_arrays.py).

Los precios y retornos siguientes son técnicos, construidos para comprobar la
aritmética. No proceden de FinMultiTime ni describen un resultado predictivo. Las
fechas de sesión, apertura y cierre sí se obtienen del calendario XNYS que usa
[MarketClock](../../src/mars_titan/data/temporal.py).

## Información conocida y etiqueta futura

Para la decisión de la sesión $t$, el objetivo es

$$
r^{OC}_{i,t+1}=\frac{Close_{i,t+1}}{Open_{i,t+1}}-1,
\qquad
y_{i,t}=r^{OC}_{i,t+1}-\hat\alpha_{i,t}
-\hat\beta_{i,t}r^{OC}_{m,t+1}.
$$

$i$ identifica al activo y $m$ al factor de mercado. $t+1$ es la próxima sesión
del mercado, no el día natural siguiente. El intercepto $\hat\alpha_{i,t}$ y la
pendiente $\hat\beta_{i,t}$ se ajustan por mínimos cuadrados con pares de retornos
cuyas dos disponibilidades son anteriores o iguales a `prediction_at`. La
ventana termina en $t$. Los dos retornos de $t+1$ solo se usan al formar la
etiqueta, nunca para estimar esos coeficientes ni como entradas conocidas en $t$.

En la historia técnica, cada par cumple $r_i=0{,}001+1{,}2r_m$. Los 252 pares
válidos disponibles en la decisión producen $\hat\alpha=0{,}001$ y
$\hat\beta=1{,}2$, dentro de la tolerancia numérica del cálculo. En la sesión
siguiente, ambos instrumentos abren a 100. El activo cierra a 102 y el factor a
101. Por tanto,

$$
y=0{,}020-0{,}001-1{,}2\times0{,}010=0{,}007.
$$

Los retornos se expresan como fracciones. El resultado equivale a 0,7 puntos
porcentuales de retorno residual. No es el retorno bruto del activo ni el
beneficio de una cartera con costes. La pendiente es adimensional y el
intercepto comparte las unidades del retorno.

La decisión es la del 22 de noviembre de 2023. El 23 no es sesión de XNYS y el
24 tiene cierre anticipado. `MarketClock` sitúa cada decisión cinco minutos
después del cierre. Para ilustrar la maduración se supone que el precio de
cierre del activo está disponible con ese margen y el del factor dos minutos
después. Esos retardos son parte del ejemplo, no evidencia de un proveedor.

| Campo o hecho | Instante UTC | Papel en el cálculo |
| --- | --- | --- |
| `prediction_at` | 2023-11-22 21:05 | Corte de las entradas y de los pares históricos del ajuste. |
| `entry_at` | 2023-11-24 14:30 | Apertura de la siguiente sesión real. |
| `exit_at` | 2023-11-24 18:00 | Cierre anticipado de esa misma sesión. |
| Cierre del activo disponible | 2023-11-24 18:05 | Permite conocer su retorno apertura-cierre. |
| Cierre del factor disponible | 2023-11-24 18:07 | Permite conocer el segundo retorno necesario. |
| `label_available_at` | 2023-11-24 18:07 | Máximo de las disponibilidades de las componentes. |

La implementación guarda la última marca como `target_available_at`, equivalente
al `label_available_at` del [contrato de datos](../data/data-contract.md).
No debe confundirse con el instante de descarga o de cálculo retrospectivo.
La etiqueta del día 22 todavía no está madura en la decisión del día 24 a las
18:05 UTC, aunque ambas sesiones ya hayan cerrado.

El horizonte apertura-cierre es el principal. Cierre-cierre y ajuste sectorial
siguen siendo variantes distintas, con las condiciones del protocolo. El ejemplo
no activa esas sensibilidades ni cambia el test final reservado de 2024.

## Comprobación reproducible

Desde la raíz del repositorio, el bloque siguiente crea únicamente precios
técnicos en memoria. Usa el entorno existente, lee la configuración del objetivo
y ejecuta el residualizador sin GPU. Comprueba el calendario, la maduración y
el ajuste con pasado. También modifica precios posteriores para verificar que
no cambian los coeficientes de la decisión anterior.

```bash
CUDA_VISIBLE_DEVICES=-1 PYTHONPATH=src uv run --no-sync python - <<'PY'
import tomllib
from datetime import date, timedelta
from decimal import Decimal
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from mars_titan.data.residual_arrays import residual_targets_array
from mars_titan.data.temporal import MarketClock

settings = tomllib.loads(Path("configs/study.toml").read_text())["target"]
history = settings["coefficient_window_sessions"]
minimum = settings["minimum_history_sessions"]
assert (history, minimum) == (252, 126)
assert (settings["entry"], settings["exit"], settings["horizon_sessions"]) == (
    "next_session_open",
    "next_session_close",
    1,
)

clock = MarketClock("US", "2022-01-01", "2023-11-30")
index = clock.days.index(date(2023, 11, 22))
following = index + 1
assert date(2023, 11, 23) not in clock.days
assert clock.days[following] == date(2023, 11, 24)
calendar = xcals.get_calendar("XNYS", start="2022-01-01", end="2023-11-30")
entry = calendar.session_open(clock.days[following]).to_pydatetime()
exit_at = calendar.session_close(clock.days[following]).to_pydatetime()

# Precios técnicos, sin observaciones del corpus ni identificadores de empresas.
x = np.resize(np.array([-0.02, -0.01, 0.01, 0.02]), len(clock.days))
common = dict(
    session=[day.isoformat() for day in clock.days], open=100.0, available_at=clock.decisions
)
factor = pd.DataFrame(dict(common, close=100 * (1 + x)))
asset = pd.DataFrame(dict(common, close=100 * (1 + 0.001 + 1.2 * x)))
asset.loc[following, "close"] = 102.0
factor.loc[following, "close"] = 101.0
factor.loc[following, "available_at"] += timedelta(minutes=2)

arguments = dict(clock=clock, cutoff="2023-11-30", history=history, minimum=minimum)
row = residual_targets_array(asset, factor, **arguments).iloc[index]
prediction = clock.decisions[index]
known = slice(index - history + 1, index + 1)
assert all(at <= prediction for at in asset.iloc[known].available_at)
assert all(at <= prediction for at in factor.iloc[known].available_at)
assert row.reason == "accepted" and row.history_pairs == history
np.testing.assert_allclose([row.alpha, row.beta], [0.001, 1.2], rtol=0, atol=1e-12)
exact = Decimal("0.020") - Decimal("0.001") - Decimal("1.2") * Decimal("0.010")
assert exact == Decimal("0.007")
np.testing.assert_allclose(row.target, float(exact), rtol=0, atol=1e-12)
assert row.target_available_at == max(
    asset.loc[following, "available_at"], factor.loc[following, "available_at"]
)
assert prediction < entry < exit_at < row.target_available_at

# Cambiar observaciones posteriores al objetivo no modifica la fila ya formada.
later = asset.copy()
later.loc[following + 1 :, "close"] *= 1.5
pd.testing.assert_series_equal(row, residual_targets_array(later, factor, **arguments).iloc[index])
# El retorno objetivo siguiente tampoco interviene en los coeficientes de t.
shock = asset.copy()
shock.loc[following, "close"] = 105.0
changed = residual_targets_array(shock, factor, **arguments).iloc[index]
np.testing.assert_array_equal([row.alpha, row.beta], [changed.alpha, changed.beta])
np.testing.assert_allclose(changed.target, 0.037, rtol=0, atol=1e-12)

print("prediction_at:", prediction.isoformat())
print("entry_at:", entry.isoformat())
print("exit_at:", exit_at.isoformat())
print("label_available_at:", row.target_available_at.isoformat())
print("history_pairs:", row.history_pairs)
print("alpha, beta:", f"{row.alpha:.3f}", f"{row.beta:.1f}")
print("residual:", f"{row.target:.3f}")
PY
```

El resultado comprobado es:

```text
prediction_at: 2023-11-22T21:05:00+00:00
entry_at: 2023-11-24T14:30:00+00:00
exit_at: 2023-11-24T18:00:00+00:00
label_available_at: 2023-11-24T18:07:00+00:00
history_pairs: 252
alpha, beta: 0.001 1.2
residual: 0.007
```

La aritmética decimal da `0.007` exactamente. El cálculo OLS en `float64` se
contrasta con tolerancia absoluta de `1e-12`, sin tolerancia relativa. Es una
verificación técnica de este ejemplo, no una estimación de precisión predictiva.
