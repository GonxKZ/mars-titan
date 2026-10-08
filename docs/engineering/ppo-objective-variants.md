# Variantes del objetivo PPO

La opción `policy_objective` distingue tres variantes sobre las seis acciones
del motor financiero existente. Comparte actor, crítico, GAE, observaciones,
recompensa y reglas de ejecución. Su ausencia conserva el camino PPO anterior.
Los archivos de las campañas previas no se migran ni se modifican.

| Identidad | Actor | Control entre épocas |
| --- | --- | --- |
| `ppo_clip_full_kl_v1` | Objetivo recortado vigente. | Diagnóstico de KL completa. |
| `ppo_kl_penalty_adaptive_v1` | `-mean(r*A) + beta*mean(KL(q||p))`. | Beta fija durante el rollout y adaptada para el siguiente. |
| `ppo_clip_kl_epoch_stop_v1` | Objetivo recortado vigente. | Omite las épocas pendientes si la KL supera el umbral. |

El término del crítico y la entropía conservan sus coeficientes. La parada
detiene la actualización conjunta, incluido el cuerpo compartido con el crítico.
No deshace la época que cruzó el umbral ni impone una cota dura de divergencia.
Estas variantes rechazan Double DQN y las arquitecturas con optimizador
auxiliar. Una actualización auxiliar posterior podría cambiar de nuevo la
política y requiere otra regla de fases.

## Configuración y cálculo

El bloque es opcional en las configuraciones de esquema 2 o 3. El diagnóstico
solo necesita `schema_version: 1` y su `id`. La parada añade `target_kl`. La
penalización añade `target_kl`, `beta_initial`, `beta_min` y `beta_max`. No se
añaden valores a una configuración antigua al leerla.

Por ejemplo, el siguiente bloque declara parámetros técnicos explícitos. No
activa una campaña ni acredita que sean adecuados para datos financieros:

```json
{
  "schema_version": 1,
  "id": "ppo_kl_penalty_adaptive_v1",
  "target_kl": 0.01,
  "beta_initial": 1.0,
  "beta_min": 0.000001,
  "beta_max": 1000000.0
}
```

El rollout conserva los seis pesos FP32 que recibió el muestreador en
`old_action_weights[T,N,6]`. La conversión
`categorical_fp32_weights_normalized_fp64_v1` los normaliza en FP64, manteniendo
los ceros. Se usa la misma conversión para la distribución actual. El logaritmo
seleccionado que utiliza el cociente PPO permanece separado. Las acciones
válidas se concilian con sus pesos históricos. Las filas de calentamiento que
fuerzan efectivo no participan en el objetivo ni en la KL.

La medición usa `categorical_behavior_to_current_kl_v1`, suma las seis acciones
y promedia todas las transiciones válidas después de una época completa. En
GRU reconstruye los prefijos con los parámetros actuales y respeta los
reinicios. No usa el estado oculto producido con los pesos anteriores. La
medición no consume los generadores de acciones o mezcla.

Si la media D es menor que `target_kl/1.5`, la beta siguiente se divide por dos.
Si es mayor que `target_kl*1.5`, se multiplica por dos. Se respetan sus límites
y la igualdad no cambia beta. La adaptación ocurre una vez al cerrar el
rollout. Esta regla procede de la sección 4 de
[PPO v2](https://arxiv.org/pdf/1707.06347v2). Los límites de beta son parte del
contrato local.

La variante con parada omite las épocas restantes cuando D supera
`1.5*target_kl`. Registra las épocas realmente ejecutadas y usa ese denominador
en sus estadísticas. Con cero filas válidas no hay KL ni actualización de
beta. Una KL entre −10⁻¹² y cero conserva su valor en el recibo y se trata como
cero solo al decidir. Una media más negativa, no finita o con soporte
incompatible se rechaza.

`approximate_kl` conserva su significado anterior, calculado con las salidas
previas al paso. No se presenta como el nuevo diagnóstico posterior. La regla
de parada aquí usa suma categórica completa y frontera de época, distinta del
estimador y calendario de
[Spinning Up](https://spinningup.openai.com/en/latest/algorithms/ppo.html).

## Estado y recuperación

El camino anterior escribe los mismos esquemas: política 2 y rollout 1. Las
variantes explícitas usan política 3 y rollout 2. La política guarda identidad,
beta vigente y anterior, rollouts completados, contador Adam y resumen de la
última medición. Los metadatos del entrenador se concilian con ese estado.
No se acepta un checkpoint antiguo como variante nueva inventando q o beta.
Un binario recompilado tiene otra huella de código y build. La compatibilidad
del lector no autoriza a sustituir el runtime de una ejecución histórica.

`advance()` sigue siendo la frontera recuperable. Completa recogida y
actualización antes de admitir snapshot. Un fallo dentro de la actualización,
medición o confirmación obliga a restaurar el último checkpoint confirmado.
No se publica un estado intermedio por minibatch. El escritor conserva dos
checkpoints recientes y el mejor, con los mismos payloads y hashes. La pausa
por KL es distinta de la selección temporal del mejor estado.

La columna nueva añade 24 bytes de contenido por transición, 24.576 bytes en
1.024 filas. El presupuesto de recogida incluye sus copias y temporales FP64,
no solo ese contenido. El cálculo se divide en minibatches. Esta cifra no es
una medida del RSS ni de la VRAM total.

Las pruebas permitidas comprueban álgebra, gradientes, tipos, máscaras,
serialización y medición con parámetros fijos. Las comprobaciones de pasos de
optimización y de recuperación de la siguiente actualización siguen
pendientes. No se ejecutan mientras esté vigente el bloqueo histórico.
La implementación de una variante no acredita mejora predictiva o financiera.

La comparación de [#137](https://github.com/GonxKZ/mars-titan/issues/137)
necesita registrar transiciones, exposiciones, actualizaciones y coste. Una
parada independiente puede producir menos updates, por lo que no demuestra
igualdad de ese presupuesto. El test final permanece cerrado. KLPO terminal
secuencial tiene un contrato de trayectorias distinto y no se activa mediante
esta opción PPO.
