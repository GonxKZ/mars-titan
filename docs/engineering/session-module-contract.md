# Contrato de un módulo de sesión

Cómo entra una familia nueva de modelos en el motor de campaña para ajustarse con los datos
multimodales completos y medirse con el mismo examen que el núcleo Titans. Está escrito para
los adaptadores de las familias 2 (cerebro holográfico), 3 (MMARRS-TITAN) y 4 (núcleo
agéntico) que hoy solo se ajustan sobre la tabla técnica del laboratorio.

La referencia que funciona es el holograma de sesión:
[`session_hologram.py`](../../src/mars_titan/models/titans/session_hologram.py) y sus pruebas
en `tests/models/titans/test_session_hologram.py`, en la rama `v12-criba3-parada`
(commit 22c95322 o posterior). Para un módulo con estado entre sesiones, la referencia es
`cross_sectional_rank_memory.py`.

## Dónde entra

El predictor financiero codifica cada modalidad, las funde, pasa por la memoria Titans (MAC)
y entrega una representación por activo. Un módulo de sesión lee las representaciones de todos
los activos de una sesión (mismo mercado y mismo instante de decisión) y devuelve otra del
mismo tamaño, antes de la rama técnica y de la cabeza:

```text
modalidades -> fusión -> MAC -> [módulo de sesión] -> rama técnica -> cabeza
```

En `financial.py` es una línea junto a la del holograma:

```python
encoded = module(encoded, markets, batch.prediction_at)
```

`encoded` tiene forma `[filas, hidden_size]`. `markets` y `prediction_at` tienen una entrada
por fila. El bloque es la sesión entera: hasta 4.198 filas en EE. UU. y 810 en China, y el
número cambia de una sesión a otra.

## Qué debe traer el módulo

| Pieza | Requisito |
| --- | --- |
| Configuración | Una `dataclass` congelada con `kind`, `mode`, `position` y sus tamaños, validada en `__post_init__`, con `identity()` y una función `parse_<nombre>(raw)` que admite `None`, un `dict` o la propia configuración |
| Módulo | `nn.Module(hidden_size, config, *, seed, dtype)` con `forward(encoded, markets, prediction_at)`. Sus pesos salen de un generador propio derivado de `seed`: no consume el generador global |
| Controles | En el mismo módulo, con los mismos parámetros y la misma inicialización: `self_only` (cada fila se lee solo a sí misma) y `shuffled_peers` (la lectura de los demás, permutada dentro de la sesión con una semilla derivada). Sin ellos no se puede decir de dónde sale una ganancia |
| Registro | El campo en `FinancialConfig` (ausente por defecto), su identidad solo cuando está declarado y su nombre en `PREDICTOR_CHANGES` de `search_cases.py`. Cada bloque añadido va entre marcas `# --- <nombre> ---` para que las fusiones no choquen |

## Reglas que comprueban las pruebas

1. **Ausente es idéntico.** Sin declarar el módulo, la huella de identidad, los parámetros y las
   predicciones del predictor son los de antes, bit a bit.
2. **Arranque neutro.** Declarado, el predictor empieza bit a bit igual que su control: el
   residuo entra con una ganancia que arranca en cero y recibe gradiente en el primer paso (no
   dos ceros encadenados).
3. **Solo su sesión.** Cada fila lee únicamente filas de su mercado y de su instante. Cambiar
   cualquier fila de otra sesión no cambia ni un bit de su salida.
4. **Sin etiquetas.** `forward` no recibe objetivos. Lo que aprenda de resultados pasados entra
   por el patrón de estado de más abajo, con etiquetas ya maduras.
5. **Equivariante.** Permutar las filas del bloque permuta la salida igual.
6. **FP32 estricto y determinista.** Dos ejecuciones dan la misma salida y el mismo gradiente
   bit a bit en FP32 y en FP64. Sin `autocast`, sin TF32 y sin BF16: el módulo rechaza
   `autocast` activo. La paridad de la campaña está hecha en FP32.
7. **Sin esperar a la GPU.** Nada de `.item()`, `.tolist()` ni comprobaciones que sincronicen
   dentro de `forward`. Las comprobaciones de finitud usan `check_finite` y `require_true`,
   que se difieren en CUDA; `linalg.solve` y `cholesky` van en su forma `_ex` con
   `check_errors=False` y `require_true` sobre `info`.
8. **Coste lineal o casi.** Con 4.198 filas, el paso con gradiente no puede crecer con el
   cuadrado de las filas (ese fue el fallo FM-13 de la memoria de rango). Una prueba mide que
   el grafo no crece con el número de filas.
9. **Estado.** `state_dict` va y vuelve, y rechaza cargar otro `mode` u otra `position`.

## Si el módulo guarda memoria entre sesiones

El estado no vive en atributos del módulo: se pasa y se devuelve, como en la memoria de rango.
El módulo ofrece `initial_state`, `validate_state`, `stage` (anota lo emitido en la sesión),
`settle` (recibe las etiquetas cuando maduran y actualiza), `detach_state` (corte de
truncamiento), `export_state` y `restore_state` (puntos de control y reanudación exacta). El
entrenador ya llama a esos ganchos en el orden cronológico correcto. Una primera versión sin
estado es bienvenida: la memoria se añade después como segundo cambio, con su propio control.

## Cómo se declara y se mide

En la declaración de la campaña es un cambio de una sola variable sobre el núcleo:

```json
"titans_mac_f3_looped": {
  "machine": "mars_titan", "recipe": "core", "runner": "fit", "variant": "mac_online",
  "tier": "s4_family3",
  "predictor_changes": {"session_module_f3": {"kind": "...", "mode": "full", "position": "after_mac"}}
}
```

con sus dos controles como modelos hermanos (`mode` `self_only` y `shuffled_peers`). Se
ajusta en el origen 2013 con la semilla 42, y lo que supere a sus controles se confirma con
tres semillas. La regla es la de la tercera criba (S-3 del cuaderno): entra si mejora el IC de
rangos por sesión del núcleo en al menos +0,002 con t de Newey-West de 2 o más y mejora a su
placebo barajado en al menos +0,0015. El autor puede fijar antes un umbral más exigente.

## Entrega

- Una rama desde `v12-criba3-parada`, un fichero por módulo y su fichero de pruebas.
- `ruff` limpio y las pruebas del módulo y de `tests/models/titans` en verde en CPU.
- Una nota corta con la hipótesis, el umbral de refutación y lo que el autor no ha podido
  comprobar.

El coordinador repite las pruebas, añade mutaciones propias (cada mutante debe hacer fallar
alguna prueba) y una prueba de humo en GPU antes de autorizar el ajuste.
