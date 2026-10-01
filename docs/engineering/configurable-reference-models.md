# Referencias multimodales configurables

`MultimodalReference` separa las arquitecturas científicas de `CostProbe`, que
se conserva para las sondas anteriores. RNN, LSTM, GRU y DLinear reciben precios,
noticias, gráficos, fundamentales y contexto macro. No hay una variante que
omita una modalidad para reducir su coste sin identificar esa diferencia.

Las anchuras admitidas son 32, 64 y 128. La profundidad puede ser uno o dos.
En las familias recurrentes determina la pila temporal y las capas de fusión.
DLinear conserva su descomposición y solo cambia la profundidad de la fusión.
Los codificadores de las otras entradas proyectan cada vector a la anchura
elegida. La salida es un escalar por muestra. MSE y MAE buscan aproximar,
respectivamente, la media y la mediana condicionales del objetivo. Una
implementación y un ajuste finitos no garantizan alcanzar esos óptimos.

El estado recurrente empieza de cero en cada ventana. La representación de
`encode` permite utilizar después otra cabeza sin modificar el contrato de
entrada. No implementa memoria persistente ni la arquitectura MARS-TITAN.

## Regularización y recuperación

La regularización usa dropout de 0, 0,1 o 0,2 en las capas de fusión. No se aplica
dropout interno de cuDNN en la pila recurrente. La primera prueba con GRU de dos
capas y ese dropout interno no recuperó pesos idénticos después de una parada.
Al conservar únicamente el dropout de fusión, la misma prueba sí mantuvo la
igualdad. No se amplió la tolerancia para aceptar el primer resultado.

La comprobación se extendió a las cuatro familias, tanto con la sonda histórica
como con una arquitectura explícita de dos capas. Se comparan pesos y predicciones
tras reanudar en mitad de una época. El generador aleatorio, el optimizador y el
cursor forman parte del checkpoint.

La arquitectura se declara dentro de cada caso:

```json
{
  "kind": "gru",
  "loss": "mae",
  "learning_rate": 0.0003,
  "seed": 42,
  "epochs": 30,
  "huber_delta": 0.01,
  "architecture": {
    "hidden_size": 64,
    "layers": 2,
    "dropout": 0.1
  }
}
```

`run_reference_case` recibe este caso junto al manifiesto supervisado y un
directorio de salida nuevo. Los casos históricos sin `architecture` mantienen
la sonda anterior y quedan identificados como tales. Una continuación exige
el mismo origen, semilla, familia, arquitectura, población y ponderación. Puede
cambiar la pérdida y la tasa según el control definido, pero no cambiar la
regularización sin declarar otro diseño.

La comprobación actual cubre formas, gradientes de las cinco entradas, ausencia
de estado arrastrado, regularización y recuperación. No demuestra superioridad
predictiva ni sustituye la campaña sobre el corpus completo. El
[MAE por sesión](../research/metrics.md) y la
[selección de épocas](checkpoint-recovery.md) tienen contratos separados.
La búsqueda de configuraciones debe quedar fijada antes de ejecutar esa campaña.
