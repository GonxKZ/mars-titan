# Controlador optativo del actor KLPO

[`KlpoLearningController`](../../native/include/mars_titan/klpo_learning.hpp)
reutiliza el [colector terminal](terminal-klpo.md), el objetivo Full-KL,
`PpoPolicy`, `FinancialBatch` y `PpoCheckpointStore`. Implementa el ciclo de
actualización con una identidad distinta, `klpo_full_fresh_waves_v1`. No activa
campañas ni modifica la ruta PPO. Bajo el bloqueo histórico se comprueban
gradientes, archivos etiquetados y transiciones sin decisiones. No se ejecutan
pasos de Adam ni se acredita una recuperación posterior a una actualización real.

## Referencia fija y oleadas nuevas

`confirmed_updates_per_reference` cuenta actualizaciones confirmadas, no oleadas.
El valor inicial es dos. Cada oleada nueva usa una referencia q fija y conserva
sus seis probabilidades FP32, acciones, recompensas y reloj original. Los logits
del actor actual se recalculan sobre la historia completa al obtener el gradiente.
No se reutiliza una oleada consumida ni se heredan las épocas o el recorte PPO.

Una oleada formada solo por pasos forzados incrementa `consumed_waves`, conserva
su registro y no incrementa `confirmed_updates`. Tampoco adelanta el refresco de
q. Una valoración inválida bloquea toda la oleada. No se retira el episodio ni
se renormaliza la población. La identidad de un episodio entre oleadas es la
tupla de identidad de ejecución, ordinal de oleada e ID del registro fuente.

La cadencia uno es un control explícito. Si p=q y la ejecución coincide, el
primer gradiente no depende de beta porque log(p/q)=0. La cadencia dos permite
que una segunda oleada nueva utilice el actor modificado y la referencia anterior.
Es una elección de integración, no una exigencia ni una ventaja demostrada del
artículo. El [pin secuencial 304e5ac7](https://github.com/yifanzhang-pro/KLPO/blob/304e5ac7ca573d45a0e42e6eaea95203260402bc/klpo/loss.py)
y [arXiv 2610.08963v1](https://arxiv.org/abs/2610.08963v1) se conservan separados
del pin predictivo. El apéndice N.1 permite reutilización bajo una referencia
fija. Esta integración usa oleadas nuevas y no declara que las épocas PPO sean
una implementación equivalente.

## Adam y acumulación

El actor utiliza un Adam propio con betas 0,9 y 0,999, epsilon 10⁻⁸,
`weight_decay=0` y `amsgrad=false`. La tasa de aprendizaje es obligatoria.
No se importan momentos PPO. Se conserva la cabeza de siete filas existente,
pero la pérdida utiliza solo las seis filas de acciones. La fila del crítico
debe mantener gradiente y momentos cero. Sus bits se fijan por huella y se
comprueban en recuperación y alrededor del paso. El tronco compartido puede
cambiar, por lo que esto no significa que la función de valor permanezca fija.

`gradient_norm=0` desactiva el clipping. Un valor positivo aplica clipping una
sola vez al gradiente global. No es recorte de cocientes ni una garantía KL.
Esta configuración Adam sin decaimiento es propia y no reproduce la receta
AdamW del repositorio de referencia.

La acumulación divide por el número total de episodios de la oleada. Un bloque
solo forzado aporta cero y conserva sus episodios en ese denominador. Los
bloques dividen el eje de episodios, nunca el tiempo. Cada GRU conserva hasta
256 pasos, sin desacoplar el prefijo cada 16 pasos. Se libera cada grafo después
de backward y solo se permite un paso tras reunir todos los bloques.

El bloque configurable admite de 1 a 128 episodios. Las guardas heredadas
limitan registros, padding, observaciones y memoria lógica por política.
El presupuesto de 512 MiB por política no es una cota del RSS conjunto ni del
contexto CUDA. Los archivos conjuntos de actor y referencia y el archivo de
registros tienen cada uno un máximo de 128 MiB. Los metadatos admiten 32 MiB.
El escritor mantiene la retención acotada existente, sin crear una carpeta por
oleada o referencia.

## Publicación y recuperación

El ciclo es `collecting → ready → consumed → collecting`. Antes de modificar
pesos, `update_ready` confirma `ready`. El paso de Adam, su contador y el consumo
se publican juntos. Ese estado `consumed` conserva q de los registros consumidos.
No puede recoger otra transición ni volver a optimizar ese registro.

`start_next_wave` hace una segunda publicación. Si se alcanzó la cadencia,
copia los pesos actuales a q e incrementa `reference_version`. En otro caso,
conserva q. Ambas rutas continúan el RNG del sampler y publican el nuevo cursor
vacío antes de recoger. No hay recogida entre las dos publicaciones. Un corte
después de confirmar `consumed` recupera ese estado y completa la segunda
publicación. Un corte después de publicar la nueva oleada recupera `collecting`.

El checkpoint contiene actor, Adam, referencia, RNG inicial y confirmado,
carteras, contexto, hidden, registros y contadores. La recuperación construye
candidatos separados y reproduce el prefijo con su referencia y RNG inicial.
Solo adopta los candidatos después de conciliar identidad, contratos y estado.
Un error tras entrar en una publicación o modificar parámetros deja el objeto
inutilizable hasta recuperar un checkpoint confirmado. No se serializan grafos
ni acumulaciones parciales de gradientes.

Quedan pendientes la ejecución de Adam CPU/CUDA, la invariancia efectiva del
crítico tras ese paso, la recuperación de una actualización ejecutada, el coste
del ciclo completo y cualquier comparación científica. Los contadores y momentos
escritos a mano en las pruebas no acreditan esas comprobaciones.

## Comprobaciones técnicas

El [recibo técnico](../../reports/engineering/terminal-klpo-updates-verification-20261009.json)
separa diez ejecutables CPU en Release, ASan/UBSan y cobertura del delta CUDA.
Las diez mutaciones finales se detectan. Dos mutaciones iniciales descubrieron
aserciones insuficientes para un prefijo solo forzado y para el RNG recién
copiado. Los intentos y las regresiones se conservan.

El recibo identifica `be88f4d9` como revisión medida. ASan/UBSan, cobertura,
análisis estático, mutaciones y CUDA se ejecutaron sobre ese código. El commit
`603156f7` añade la aserción del RNG copiado, que se ejecutó antes por separado
en Release, ASan/UBSan y cobertura. También reorganiza una llamada de
`klpo_learning.cpp` sin cambiar sus 3.242 tokens C++. Los hashes de fuentes del
recibo ya corresponden a `603156f7`, y `format_only_correspondence` enlaza el
hash anterior con el final. Sobre `603156f7` se repitieron los diez ejecutables
CPU en Release y ASan/UBSan, con los resultados de `head_reconciliation`.
Cobertura, mutaciones, análisis estático y CUDA no se repitieron sobre ese commit.

LLVM cubre 346 de 377 líneas del controlador. El CCN máximo de sus funciones
es 17. La complejidad máxima de las funciones modificadas en los tres módulos
es 35, en el lector compartido. El CRAP máximo es 147,15, usando regiones LLVM
de código por función y Lizard 1.24.1. `update_ready` y `terminal_step` tienen
cobertura cero porque requieren actualizaciones reales. Estas cifras son
diagnósticos, no una demostración de corrección.

En CUDA, MLP y GRU conservan pérdidas y gradientes dentro de `rtol=1e-5` y
`atol=1e-6`. El máximo error de gradiente es 1,073×10⁻⁶ y el de pérdida
8,115×10⁻⁸. El gradiente posterior a recuperar el actor es exacto. La referencia
importada y la publicación de una oleada forzada también se recuperan, con RNG
global y parámetros intactos. Ese tramo alcanzó 42.671.616 bytes asignados y
56.623.104 reservados. El máximo asignado del delta es 76.267.008 bytes, en el
tramo de actores, con 90.177.536 reservados y cuota Torch de 128 MiB. Un intento
previo rechazó un perfil MLP usado por error con el fixture GRU y se conserva
por separado.

El perfil previo H=256 utiliza ocho episodios, entrada de 128 números y una
GRU de 64 unidades. Los bloques de uno y ocho episodios alcanzan 5.024.376 y
8.170.176 bytes de almacenamiento ATen CPU. En CUDA alcanzan 78.547.456 y
94.834.176 bytes asignados. Son medidas instrumentadas anteriores al
controlador. No miden el ciclo de actualización ni acreditan una aceleración.

La selección reproducible evita los ejecutables anteriores que optimizan:

```bash
CUDA_VISIBLE_DEVICES=-1 UV_OFFLINE=1 cmake --preset native-ppo-release -S native \
  -B build/native/terminal-actor -DMARS_TITAN_BUILD_RUNNER=OFF \
  -DMARS_TITAN_LIBTORCH_ENABLE_CUDA=OFF
cmake --build build/native/terminal-actor --target klpo_actor_tests \
  klpo_reference_tests klpo_learning_tests klpo_episodes_tests klpo_terminal_tests \
  klpo_collection_tests klpo_policy_tests ppo_variant_state_tests \
  ppo_variant_objective_tests ppo_checkpoint_tests -j 1
CUDA_VISIBLE_DEVICES=-1 ctest --test-dir build/native/terminal-actor \
  -R '^(klpo_actor|klpo_reference|klpo_learning|klpo_episodes|klpo_terminal|klpo_collection|klpo_policy|ppo_variant_state|ppo_variant_objective|ppo_checkpoint)$' \
  -LE '^optimizer-steps$' --output-on-failure
```

La selección original incluía `ppo_variant_state`, que aplica un paso de Adam con `update_from_cpu`. Desde el 10 de octubre de 2026 lleva la etiqueta `optimizer-steps` y `-LE` la deja fuera mientras rija el bloqueo.
